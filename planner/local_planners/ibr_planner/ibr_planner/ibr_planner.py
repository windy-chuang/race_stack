"""ROS 2 iterative-best-response planner using Spica et al. (2018).

Paper: "A Real-Time Game Theoretic Planner for Autonomous Two-Player
Drone Racing", Section V, especially Equations (16) and (18).

Keeps the source planner's 2-D single-integrator car approximation, existing
F110 waypoint interfaces and ROS topics. Track geometry is available only as
waypoints, so the continuous centerline derivative in Equation (18) uses an
approximate discrete curvature.

The outer IBR loop alternates players, passing the most recent collision
multipliers (dual variables) to the other player's objective. Each best
response is solved through an inner sequential convex approximation loop.
The dual variables here are those of the final convex approximation; passing
them between sensitivity-augmented OCPs is an implementation choice discussed
in the accompanying explanation, not an exact evaluation of the value-function
derivative for the unmodified problem (10).
"""

import copy
import time

import cvxpy as cp
import numpy as np
import rclpy
from f110_msgs.msg import Wpnt, WpntArray
from nav_msgs.msg import Odometry
from rclpy.node import Node


class PlannerSolveError(RuntimeError):
    """Raised when an OCP fails and no trustworthy primal/dual result exists."""


class IBRPlanner(Node):
    def __init__(self):
        super().__init__("ibr_planner")

        self.rate_hz = float(self.declare_parameter("rate_hz", 10.0).value)
        self.speed_scale = float(self.declare_parameter("speed_scale", 0.2).value)
        # self.n_loc_wpnts = int(self.declare_parameter("n_loc_wpnts", 80).value)
        self.opp_ns = str(self.declare_parameter("opp_ns", "opp").value)

        self.dt = float(self.declare_parameter("dt", 0.1).value)
        self.n_steps = int(self.declare_parameter("n_steps", 15).value)

        # IBR outer iterations and SCP inner iterations are NOT solver iterations.
        self.n_game_iterations = int(
            self.declare_parameter("n_game_iterations", 1).value
        )
        self.n_sqp_iterations = int(
            self.declare_parameter("n_sqp_iterations", 3).value
        )
        self.inner_tolerance = float(
            self.declare_parameter("inner_tolerance", 1e-3).value
        )
        self.alpha = float(self.declare_parameter("alpha", 1.0).value)

        # Separate CLARABEL stopping parameters for EACH convex subproblem.
        self.solver_max_iter = int(
            self.declare_parameter("solver_max_iter", 200).value
        )
        self.solver_tolerance = float(
            self.declare_parameter("solver_tolerance", 1e-7).value
        )

        if (
            self.rate_hz <= 0 or self.dt <= 0 or self.n_steps < 2
            or self.n_game_iterations < 1 or self.n_sqp_iterations < 1
            or self.alpha < 0 or self.inner_tolerance <= 0
        ):
            raise ValueError("Invalid horizon, step size, iteration count, or alpha")

        self.have_ego = False
        self.have_opp = False
        self.state = np.zeros((2, 2), dtype=float)
        self.car_params = [
            {"r_col": 0.29, "v_max": 0.0},
            {"r_col": 0.29, "v_max": 0.0},
        ]
        self.d_coll = sum(p["r_col"] for p in self.car_params)

        self.wpnts = None
        self.wpnt_dist = None
        self.track_centers = None
        self.track_tangents = None
        self.track_normals = None
        self.track_d_left = None
        self.track_d_right = None
        self.track_curvature = None

        # Compile a reusable DPP-compatible conic problem once.
        self.build_problem()

        self.create_subscription(
            WpntArray, "/global_waypoints_scaled", self.gb_cb, 10
        )
        self.create_subscription(
            WpntArray, "/centerline_waypoints", self.centerline_cb, 10
        )
        self.create_subscription(Odometry, "car_state/odom", self.ego_cb, 10)
        self.create_subscription(
            Odometry, f"/{self.opp_ns}/car_state/odom", self.opp_cb, 10
        )
        self.loc_pub = self.create_publisher(WpntArray, "local_waypoints", 10)
        self.create_timer(1.0 / self.rate_hz, self.loop)

    def build_problem(self):
        """Build one convexified player OCP; parameters change for each solve.

        Equivalent to Eq. (16) with the first term linearized per Eq. (18):

            maximize sigma.T @ p[N] + sum_k (alpha*mu_opp[k]*beta[k]).T @ p[k]

        The controls u[k] are eliminated using u[k] = p[k] - p[k-1],
        so ||u[k]|| <= v_max*dt is implemented with norm constraints.
        """
        n = self.n_steps
        self.p = cp.Variable((n, 2), name="predicted_positions")
        p = self.p

        self.par_p0 = cp.Parameter(2, name="initial_position")
        self.par_step = cp.Parameter(nonneg=True, name="maximum_step_length")
        self.par_n = cp.Parameter((n, 2), name="track_normals")
        self.par_nc = cp.Parameter(n, name="normal_dot_center")
        self.par_left = cp.Parameter(n, name="available_left_width")
        self.par_right = cp.Parameter(n, name="available_right_width")
        self.par_beta = cp.Parameter((n, 2), name="own_collision_bearings")
        self.par_beta_opp = cp.Parameter(n, name="bearing_dot_opponent")
        self.par_sigma = cp.Parameter(2, name="progress_gradient")
        self.par_sensitivity = cp.Parameter((n, 2), name="frozen_sensitivity")

        # Discretized holonomic dynamics from Eq. (9b),(9e).
        dynamics = [
            cp.norm(p[0] - self.par_p0, 2) <= self.par_step,
            cp.norm(p[1:] - p[:-1], axis=1) <= self.par_step,
        ]

        # Linearization of track constraints (9d). Original paper assumes
        # symmetric half-width; use left/right widths from F110 waypoints.
        lateral = cp.sum(cp.multiply(self.par_n, p), axis=1) - self.par_nc
        track = [lateral <= self.par_left, -lateral <= self.par_right]

        # Eq. (9c) linearized per Section V-B:
        # beta_ij[k].T @ (p_opp[k] - p_i[k]) >= d_coll.
        # Save THIS specific vector constraint for its dual_value (mu_i).
        signed_separation = self.par_beta_opp - cp.sum(
            cp.multiply(self.par_beta, p), axis=1
        )
        self.collision_constraint = signed_separation >= self.d_coll

        # Combine mu and beta in NumPy BEFORE passing them to the CVXPY
        # parameter. This keeps both fixed through all inner SCP iterations.
        progress_term = self.par_sigma @ p[-1]
        sensitivity_term = cp.sum(cp.multiply(self.par_sensitivity, p))
        objective = cp.Maximize(progress_term + sensitivity_term)

        self.prob = cp.Problem(
            objective, dynamics + track + [self.collision_constraint]
        )
        self.get_logger().info(f"IBR convex subproblem DPP: {self.prob.is_dpp()}")
    
    def diagnose_infeasibility(self, player, guess, opponent_trajectory):
        """Debug only: relax track/collision with slacks to locate the conflict."""
        N = self.n_steps
        p = cp.Variable((N, 2))
        s_left = cp.Variable(N, nonneg=True)
        s_right = cp.Variable(N, nonneg=True)
        s_coll = cp.Variable(N, nonneg=True)

        p0, step = self.par_p0.value, float(self.par_step.value)
        dyn = [cp.norm(p[0] - p0, 2) <= step,
               cp.norm(p[1:] - p[:-1], axis=1) <= step]
        lateral = cp.sum(cp.multiply(self.par_n.value, p), axis=1) - self.par_nc.value
        sep = self.par_beta_opp.value - cp.sum(cp.multiply(self.par_beta.value, p), axis=1)
        track = [lateral <= self.par_left.value + s_left,
                 -lateral <= self.par_right.value + s_right]
        coll = [sep >= self.d_coll - s_coll]

        def feasible(cons):
            pr = cp.Problem(cp.Minimize(0), cons)
            pr.solve(solver=cp.CLARABEL)
            return pr.status

        # Which constraint groups conflict? (slacks fixed to 0 = original constraints)
        zero = [s_left == 0, s_right == 0, s_coll == 0]
        st_track = feasible(dyn + track + zero)
        st_coll = feasible(dyn + coll + zero)

        # Where do they conflict?
        relaxed = cp.Problem(cp.Minimize(cp.sum(s_left + s_right + s_coll)),
                             dyn + track + coll)
        relaxed.solve(solver=cp.CLARABEL)

        log = self.get_logger()
        log.warn(f"[diag] P{player}: dyn+track={st_track}, dyn+coll={st_coll}")
        for k in range(N):
            sl, sr, sc = s_left.value[k], s_right.value[k], s_coll.value[k]
            if max(sl, sr, sc) > 1e-4:
                d_opp = np.linalg.norm(guess[k] - opponent_trajectory[k])
                log.warn(
                    f"[diag]  k={k}: slack L={sl:.3f} R={sr:.3f} coll={sc:.3f} "
                    f"| guess-opp dist={d_opp:.2f} "
                    f"beta={np.round(self.par_beta.value[k], 2)}"
                )

    

    
    def gb_cb(self, msg: WpntArray):
        if not msg.wpnts:
            return
        # Opponent drives the unscaled global line; read its top speed before scaling.
        vmax_raw = max(float(w.vx_mps) for w in msg.wpnts)

        self.wpnts = copy.deepcopy(msg.wpnts)
        for wpnt in self.wpnts:
            wpnt.vx_mps *= self.speed_scale
        if len(self.wpnts) >= 2:
            self.wpnt_dist = self.wpnts[1].s_m - self.wpnts[0].s_m

        self.car_params[0]["v_max"] = vmax_raw * self.speed_scale  # ego
        self.car_params[1]["v_max"] = vmax_raw                     # opponent


    def centerline_cb(self, msg: WpntArray):
        if len(msg.wpnts) < 3:
            self.get_logger().warn("Need at least three centerline waypoints")
            return

        waypoints = msg.wpnts
        centers = np.array([[w.x_m, w.y_m] for w in waypoints], dtype=float)
        psi = np.array([w.psi_rad for w in waypoints], dtype=float)
        d_left = np.array([w.d_left for w in waypoints], dtype=float)
        d_right = np.array([w.d_right for w in waypoints], dtype=float)

        # Some closed-track maps repeat the first waypoint at the end.
        if len(centers) > 3 and np.linalg.norm(centers[0] - centers[-1]) < 1e-7:
            centers = centers[:-1]
            psi = psi[:-1]
            d_left = d_left[:-1]
            d_right = d_right[:-1]

        self.track_centers = centers
        self.track_tangents = np.column_stack([np.cos(psi), np.sin(psi)])
        self.track_normals = np.column_stack([-np.sin(psi), np.cos(psi)])
        self.track_d_left = d_left
        self.track_d_right = d_right

        # Approximate curvature from discrete tangent headings and track
        # centerline chord lengths, needed for sigma in Equation (18).
        self.track_curvature = self.estimate_track_curvature()

    def estimate_track_curvature(self):
        centers = self.track_centers
        tangents = self.track_tangents
        m = len(centers)
        distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        positive = distances[distances > 1e-8]
        step_typical = np.median(positive) if positive.size else 0.0
        closed = (
            step_typical > 0.0
            and np.linalg.norm(centers[0] - centers[-1]) < 2.5 * step_typical
        )

        kappa = np.zeros(m, dtype=float)
        for k in range(m):
            before = (k - 1) % m if closed else max(k - 1, 0)
            after = (k + 1) % m if closed else min(k + 1, m - 1)
            arclength = (
                np.linalg.norm(centers[k] - centers[before])
                + np.linalg.norm(centers[after] - centers[k])
            )
            if arclength < 1e-8:
                continue
            t0, t1 = tangents[before], tangents[after]
            dpsi = np.arctan2(
                t0[0] * t1[1] - t0[1] * t1[0], np.dot(t0, t1)
            )
            kappa[k] = dpsi / arclength
        return kappa

    def track_frame_at(self, pos):
        """Return nearest sampled track center, tangent, normal, left/right width."""
        idx = int(np.argmin(np.linalg.norm(self.track_centers - pos, axis=1)))
        return (
            idx,
            self.track_centers[idx],
            self.track_tangents[idx],
            self.track_normals[idx],
            self.track_d_left[idx],
            self.track_d_right[idx],
        )

    def progress_gradient(self, terminal_position):
        """Discrete-track approximation of Eq. (18): ds/dp = t/(1-kappa*e)."""
        idx, c, t, n, _, _ = self.track_frame_at(terminal_position)
        lateral_error = float(n @ (terminal_position - c))
        denom = 1.0 - self.track_curvature[idx] * lateral_error
        # The exact derivative is singular at denom=0. Do not allow noisy
        # map curvature to produce an enormous artificial progress gradient.
        if not np.isfinite(denom) or abs(denom) < 0.05:
            return t.copy()
        return t / denom

    @staticmethod
    def relative_bearings(player_guess, opponent_traj, initial_relative):
        """Bearings from the optimized player toward its fixed opponent.

        For coincident predicted points, choose the initial measured relative
        bearing. The physical collision constraint may still be infeasible.
        """
        delta = opponent_traj - player_guess
        dist = np.linalg.norm(delta, axis=1)
        beta = np.zeros_like(delta)
        valid = dist > 1e-9
        beta[valid] = delta[valid] / dist[valid, None]

        fallback = np.array(initial_relative, dtype=float)
        length = np.linalg.norm(fallback)
        if length < 1e-9:
            fallback = np.array([1.0, 0.0])
        else:
            fallback /= length
        beta[~valid] = fallback
        return beta

    @staticmethod
    def trajectory_controls(trajectory, initial_position):
        """Recover u[k] = p[k]-p[k-1] from reduced position-only variables."""
        return np.diff(
            np.vstack([np.asarray(initial_position)[None, :], trajectory]),
            axis=0,
        )

    def strategy_change_norm(self, new_positions, old_positions, initial_position):
        """Full-strategy norm ||theta_new - theta_old|| from Section V-B."""
        delta_positions = (new_positions - old_positions).ravel()
        new_u = self.trajectory_controls(new_positions, initial_position)
        old_u = self.trajectory_controls(old_positions, initial_position)
        return float(np.linalg.norm(
            np.concatenate([delta_positions, (new_u - old_u).ravel()])
        ))

    def init_trajectory(self, player, p0):
        """Same tangent-following initial strategy as the source planner."""
        v_max = self.car_params[player]["v_max"]
        p = np.array(p0, dtype=float, copy=True)
        result = np.zeros((self.n_steps, 2), dtype=float)
        for k in range(self.n_steps):
            _, _, tangent, _, _, _ = self.track_frame_at(p)
            p += self.dt * v_max * tangent
            result[k] = p
        return result

    def best_response(self, player, state, trajectories, multipliers):
        """Solve Eq. (16) for one player by sequential convex approximation.

        Frozen through the whole inner loop:
          - opponent predicted trajectory,
          - opponent collision multipliers mu_opp,
          - beta used by the sensitivity objective.

        Updated at each inner step m:
          - sigma_i^m at the current guess's last position,
          - track linearizations,
          - OWN collision half-plane directions.

        Returns (new trajectory, OWN collision dual multipliers).
        """
        other = 1 - player
        initial_position = np.array(state[player], dtype=float, copy=True)
        opponent_trajectory = np.array(trajectories[other], copy=True)
        guess = np.array(trajectories[player], copy=True)
        mu_other = np.array(multipliers[other], dtype=float, copy=True)
        if mu_other.shape != (self.n_steps,):
            raise ValueError("Opponent collision multiplier vector has wrong shape")

        relative_at_start = state[other] - state[player]

        # This beta is evaluated at the beginning of the OCP and is FROZEN
        # for all of its inner SCP iterations (bar-beta in Eq. 16).
        frozen_beta = self.relative_bearings(
            guess, opponent_trajectory, relative_at_start
        )
        frozen_sensitivity = self.alpha * mu_other[:, None] * frozen_beta

        self.par_p0.value = initial_position
        self.par_step.value = self.car_params[player]["v_max"] * self.dt
        self.par_sensitivity.value = frozen_sensitivity

        last_good_trajectory = None
        last_good_mu = None
        for m in range(self.n_sqp_iterations):
            n = np.zeros((self.n_steps, 2), dtype=float)
            normal_dot_center = np.zeros(self.n_steps, dtype=float)
            widths_left = np.zeros(self.n_steps, dtype=float)
            widths_right = np.zeros(self.n_steps, dtype=float)
            for k in range(self.n_steps):
                _, center, _, normal, d_left, d_right = self.track_frame_at(guess[k])
                n[k] = normal
                normal_dot_center[k] = normal @ center
                widths_left[k] = d_left - self.car_params[player]["r_col"]
                widths_right[k] = d_right - self.car_params[player]["r_col"]

            if min(np.min(widths_left), np.min(widths_right)) < 0:
                raise PlannerSolveError("Car radius exceeds available track width")

            # Unlike frozen_beta, beta_collision is updated each SCP iteration.
            beta_collision = self.relative_bearings(
                guess, opponent_trajectory, relative_at_start
            )
            self.par_n.value = n
            self.par_nc.value = normal_dot_center
            self.par_left.value = widths_left
            self.par_right.value = widths_right
            self.par_beta.value = beta_collision
            self.par_beta_opp.value = np.sum(
                beta_collision * opponent_trajectory, axis=1
            )
            self.par_sigma.value = self.progress_gradient(guess[-1])

            # Warm start affects solver initialization, not objective coefficients.
            self.p.value = guess.copy()
            try:
                self.prob.solve(
                    solver=cp.CLARABEL,
                    warm_start=True,
                    max_iter=self.solver_max_iter,
                    tol_gap_abs=self.solver_tolerance,
                    tol_feas=self.solver_tolerance,
                )
            except cp.error.SolverError as exc:
                if last_good_trajectory is not None:
                    self.get_logger().warn(
                        f"Player {player}, SCP {m}: solver error ({exc}); "
                        "keeping last successful subproblem"
                    )
                    break
                raise PlannerSolveError(f"Player {player}: {exc}") from exc

            if (
                self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                or self.p.value is None
                or self.collision_constraint.dual_value is None
            ):

                # --- diagnostics: which k=0 constraint is unreachable? ---
                step = float(self.par_step.value)
                gap0 = float(np.linalg.norm(state[other] - state[player]))
                # Best separation p[0] can achieve along beta within one step
                coll_margin0 = float(
                    beta_collision[0] @ (opponent_trajectory[0] - initial_position)
                ) + step - self.d_coll
                # Lateral offset of p0 in the strip the solver actually used for k=0
                lat0 = float(n[0] @ initial_position - normal_dot_center[0])
                left_margin0 = widths_left[0] - (lat0 - step)
                right_margin0 = widths_right[0] - (-lat0 - step)
                self.get_logger().warn(
                    f"[diag] P{player} SCP {m} status={self.prob.status} "
                    f"gap0={gap0:.2f} d_coll={self.d_coll:.2f} step={step:.3f} "
                    f"coll_margin0={coll_margin0:.3f} "
                    f"lat0={lat0:.2f} left_margin0={left_margin0:.3f} "
                    f"right_margin0={right_margin0:.3f}"
                )

                self.diagnose_infeasibility(player, guess, opponent_trajectory)

                # --- end diagnostics ---

                if last_good_trajectory is not None:
                    self.get_logger().warn(
                        f"Player {player}, SCP {m}: {self.prob.status}; "
                        "keeping last successful subproblem"
                    )
                    break
                raise PlannerSolveError(
                    f"Player {player} has no primal/dual solution: {self.prob.status}"
                )

            new_guess = np.array(self.p.value, dtype=float, copy=True)
            mu_player = np.asarray(
                self.collision_constraint.dual_value, dtype=float
            ).reshape(self.n_steps).copy()
            if not np.all(np.isfinite(new_guess)) or not np.all(np.isfinite(mu_player)):
                raise PlannerSolveError("Non-finite primal or dual solution")
            # Tiny negative entries may result from numerical tolerance.
            mu_player = np.maximum(mu_player, 0.0)

            # The half-plane constraint is conservative for true Euclidean
            # separation; reject solutions with a significant physical violation.
            separation = np.linalg.norm(new_guess - opponent_trajectory, axis=1)
            if np.min(separation) < self.d_coll - 1e-3:
                raise PlannerSolveError(
                    "Solver result violates collision distance; refusing to publish"
                )

            residual = self.strategy_change_norm(new_guess, guess, initial_position)
            last_good_trajectory = new_guess
            last_good_mu = mu_player
            self.get_logger().debug(
                f"Player {player}, SCP {m+1}/{self.n_sqp_iterations}, "
                f"delta_theta={residual:.3e}, "
                f"max_mu={np.max(mu_player):.3g}"
            )
            guess = new_guess
            if residual <= self.inner_tolerance:
                break

        if last_good_trajectory is None:
            raise PlannerSolveError(f"Player {player} produced no solution")
        return last_good_trajectory, last_good_mu

    def iterative_br(self, i_ego, state, n_game_iterations=None, n_sqp_iterations=None):
        """Outer IBR loop, following the ego-first branch of Figure 3.

        One outer cycle: solve ego -> solve opponent. Then always solve ego
        once more so the published strategy uses the final opponent prediction.
        The inner SCP iteration count can optionally be overridden per call.
        """
        if i_ego not in (0, 1):
            raise ValueError("i_ego must be 0 or 1")
        game_count = (
            self.n_game_iterations if n_game_iterations is None
            else int(n_game_iterations)
        )
        saved_inner_count = self.n_sqp_iterations
        if n_sqp_iterations is not None:
            self.n_sqp_iterations = int(n_sqp_iterations)
        if game_count < 1 or self.n_sqp_iterations < 1:
            raise ValueError("Iteration counts must be positive")

        trajectories = [self.init_trajectory(i, state[i]) for i in (0, 1)]
        multipliers = [np.zeros(self.n_steps), np.zeros(self.n_steps)]
        other = 1 - i_ego
        start_time = time.perf_counter()

        try:
            for _ in range(game_count):
                for player in (i_ego, other):
                    trajectories[player], multipliers[player] = self.best_response(
                        player, state, trajectories, multipliers
                    )

            # Extra ego resolution explicitly shown in the paper's Fig. 3.
            trajectories[i_ego], multipliers[i_ego] = self.best_response(
                i_ego, state, trajectories, multipliers
            )
        except PlannerSolveError as exc:
            # Infeasible hard collision constraints cannot be safely papered
            # over by returning the old, potentially colliding trajectory.
            self.get_logger().warn(f"IBR failed: {exc}")
            return None
        finally:
            self.n_sqp_iterations = saved_inner_count
            self.get_logger().info(
                f"Total IBR time: {time.perf_counter() - start_time:.4f} s"
            )
        return trajectories[i_ego]

    def ego_cb(self, msg: Odometry):
        self.state[0] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        self.have_ego = True

    def opp_cb(self, msg: Odometry):
        self.state[1] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        self.have_opp = True

    def loop(self):
        if (
            not self.have_ego or not self.have_opp
            or self.wpnts is None or self.track_centers is None
            or self.car_params[0]["v_max"] <= 0
        ):
            return

        # One measured state snapshot per receding-horizon planning tick.
        state = self.state.copy()
        point_array = self.iterative_br(0, state)
        if point_array is None:
            # No invented feasible trajectory; leave fallback to a supervisor.
            return

        pts = np.vstack([state[0], point_array])
        vx = np.linalg.norm(np.diff(pts, axis=0), axis=1) / self.dt

        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        out.wpnts = [
            Wpnt(id=k, x_m=float(pos[0]), y_m=float(pos[1]), vx_mps=float(speed))
            for k, (pos, speed) in enumerate(zip(point_array, vx))
        ]
        self.loc_pub.publish(out)


def main():
    rclpy.init()
    node = IBRPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

