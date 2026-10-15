import copy

import rclpy
import numpy as np
import time
import cvxpy as cp
from rclpy.node import Node
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray, Wpnt



class IBRPlanner(Node):
    """
    
    """

    def __init__(self):
        super().__init__('ibr_planner')
        self.rate_hz = self.declare_parameter('rate_hz', 10).value
        self.speed_scale = self.declare_parameter('speed_scale', 0.2).value
        self.n_loc_wpnts = self.declare_parameter('n_loc_wpnts', 80).value

        self.have_ego = False
        self.have_opp = False

        self.state = np.zeros((2, 2))
        self.car_params = [{}, {}]
        self.car_params[0]["r_col"] = 0.58  # half car length
        self.car_params[0]["r_safe"] = 1.16 # full car length
        self.car_params[1]["r_col"] = 0.58
        self.car_params[1]["r_safe"] = 1.16

        self.dt = 0.1
        self.n_steps = 15
        self.blocking = True
        
        # These are some parameters that could be tuned.
        # They control how the safety penalty and the relaxed constraints are weighted.
        self.nc_weight = 2.0
        self.nc_relax_weight = 128.0
        self.track_relax_weight = 128.0  # possibly should be the largest of the gains?
        self.blocking_weight = 16.0  # increasing gain increases the aggressiveness in the blocking behavior

        
        self.wpnts = None
        self.wpnt_dist = None
        self.cur_s = None

        self.create_subscription(WpntArray, '/global_waypoints_scaled', self.gb_cb, 10)
        self.create_subscription(WpntArray, '/centerline_waypoints', self.centerline_cb, 10)
        self.opp_ns = self.declare_parameter('opp_ns', 'opp').value
        # own car (relative -> /ego/...)
        self.create_subscription(Odometry, 'car_state/odom', self.ego_cb, 10)
        # other car (absolute -> /opp/...)
        self.create_subscription(Odometry, f'/{self.opp_ns}/car_state/odom', self.opp_cb, 10)

        self.loc_pub = self.create_publisher(WpntArray, 'local_waypoints', 10)
        self.create_timer(1 / self.rate_hz, self.loop)

        self.build_problem()

    def build_problem(self):
        N = self.n_steps
        # both cars use the same radii, so one problem works for i_ego = 0 and 1
        d_coll = self.car_params[0]["r_col"] + self.car_params[1]["r_col"]
        d_safe = self.car_params[0]["r_safe"] + self.car_params[1]["r_safe"]
        w = 0.5 ** np.arange(N)  # exponentially decreasing weights (constant)

        # --- parameters: numbers that change every tick ---
        self.par_p0 = cp.Parameter(2)               # current position
        self.par_step = cp.Parameter(nonneg=True)   # v_max * dt
        self.par_n = cp.Parameter((N, 2))           # track normal n[k]
        self.par_nc = cp.Parameter(N)               # n[k] . c[k]
        self.par_half_w = cp.Parameter(N)           # width[k] - r_col
        self.par_beta = cp.Parameter((N, 2))        # unit vector ego -> opp
        self.par_beta_opp = cp.Parameter(N)         # beta[k] . p_opp[k]
        self.par_t_last = cp.Parameter(2)           # tangent at last point
        self.par_bn = cp.Parameter((N, 2))
        self.par_bn_opp = cp.Parameter(N)

        self.p = cp.Variable((N, 2))
        p = self.p

        # dynamics: ||p0 - p[0]|| <= v*dt, ||p[k+1] - p[k]|| <= v*dt
        dyn = [cp.norm(p[0] - self.par_p0) <= self.par_step,
            cp.norm(p[1:] - p[:-1], axis=1) <= self.par_step]

        # track: |n[k].p[k] - n[k].c[k]| <= width - r_col
        lateral = cp.sum(cp.multiply(self.par_n, p), axis=1) - self.par_nc
        track = [lateral <= self.par_half_w, lateral >= -self.par_half_w]

        # non-collision (linearized half-plane): beta.p_opp - beta.p >= d_coll
        dist = self.par_beta_opp - cp.sum(cp.multiply(self.par_beta, p), axis=1)
        nc = [dist >= d_coll]
        nc_obj = w @ cp.pos(d_safe - dist)

        # blocking: sum_k gamma_k * (n[k].(p[k] - p_opp[k]))^2, gamma folded into par_bn
        block_obj = cp.sum_squares(cp.sum(cp.multiply(self.par_bn, p), axis=1) - self.par_bn_opp)


        obj = -self.par_t_last @ p[-1]

        self.prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.blocking_weight * block_obj),
                            dyn + track + nc)
        self.get_logger().info(f'IBR problem is DPP: {self.prob.is_dpp()}')

        
    def gb_cb(self, msg: WpntArray):
        # copy so scaling never compounds on repeated messages
        wpnts = copy.deepcopy(msg.wpnts)
        for wpnt in wpnts:
            wpnt.vx_mps *= self.speed_scale
        self.wpnts = wpnts
        self.wpnt_dist = wpnts[1].s_m - wpnts[0].s_m
        self.car_params[0]["v_max"] = max([wpnt.vx_mps for wpnt in wpnts])
        self.car_params[1]["v_max"] = max([wpnt.vx_mps for wpnt in wpnts])


    def centerline_cb(self, msg: WpntArray):
        wpnts = msg.wpnts
        psi = np.array([w.psi_rad for w in wpnts])

        self.track_centers = np.array([[w.x_m, w.y_m] for w in wpnts])

        self.track_tangents = np.stack([np.cos(psi), np.sin(psi)], axis=1)
        self.track_normals = np.stack([-np.sin(psi), np.cos(psi)], axis=1)

        self.track_d_left = np.array([w.d_left for w in wpnts])
        self.track_d_right = np.array([w.d_right for w in wpnts])
        self.track_widths = (self.track_d_left + self.track_d_right) / 2.0




    def ego_cb(self, msg: Odometry):
        self.state[0, 0] = msg.pose.pose.position.x
        self.state[0, 1] = msg.pose.pose.position.y
        self.have_ego = True
        


    def opp_cb(self, msg: Odometry):
        self.state[1, 0] = msg.pose.pose.position.x
        self.state[1, 1] = msg.pose.pose.position.y
        self.have_opp = True

    
    
    def loop(self):
        if not (self.have_ego and self.have_opp) or self.wpnts is None or not hasattr(self, 'track_centers'):
            return
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        point_array = self.iterative_br(0, self.state, n_game_iterations=1, n_sqp_iterations=2)

        pts = np.vstack([self.state[0], point_array])
        steps = np.diff(pts, axis=0)
        vx = np.linalg.norm(steps, axis=1) / self.dt

        out.wpnts = [
            Wpnt(id=i, x_m=x, y_m=y, vx_mps=v)
            for i, ((x, y), v) in enumerate(zip(point_array.tolist(), vx.tolist()))
        ]

        self.loc_pub.publish(out)

    def init_trajectory(self, i_0, p_0):
        """Initialize Trajectory along the track tangent
        Based on the start position p_0, return an initial guess for a trajectory
        This is simply a line following the tangent of the track with maximal speed

        :param i_0: The index of the drone to initialize the trajectory for.
        :param p_0: The current position of the drone
        :return: A trajectory of length self.n_steps
        """
        v_ego = self.car_params[i_0]["v_max"]
        trajectory = np.zeros(shape=(self.n_steps, 2))
        p = np.copy(p_0)  # Copy state as it gets modified throughout the loop
        for k in range(self.n_steps):
            idx, c, t, n, width = self.track_frame_at(p)
            p += self.dt * v_ego * t
            trajectory[k, :] = p
        return trajectory

    def iterative_br(self, i_ego, state, n_game_iterations=1, n_sqp_iterations=2):
        trajectories = [
            self.init_trajectory(i, state[i, :]) for i in [0, 1]
        ]
        t0 = time.time()
        for i_game in range(n_game_iterations - 1):
            for i in [i_ego, (i_ego + 1) % 2]:
                for i_sqp in range(n_sqp_iterations - 1):
                    trajectories[i] = self.best_response(i, state, trajectories)
        # one last time for i_ego
        for i_sqp in range(n_sqp_iterations):
            trajectories[i_ego] = self.best_response(i_ego, state, trajectories)
        t1 = time.time()
        self.get_logger().info(f'Total IBR solution time: {t1 - t0:.4f} s')
        return trajectories[i_ego]

    def best_response(self, i_ego, state, trajectories):
        t_build = time.time()
        i_opp = (i_ego + 1) % 2
        N = self.n_steps
        n = np.zeros((N, 2))
        nc = np.zeros(N)
        half_w = np.zeros(N)
        beta = np.zeros((N, 2))
        beta_opp = np.zeros(N)
        gamma = np.zeros(N)

        for k in range(N):
            p_ego = trajectories[i_ego][k, :]
            p_opp = trajectories[i_opp][k, :]
            _, c, t, n[k], width = self.track_frame_at(p_ego)
            lead = (p_ego - p_opp) @ t
            if lead > 0:
                gamma[k] = 0.5 ** k / (lead * lead) / (k + 1)
            if k == 0:
                lead_0 = lead
            nc[k] = n[k] @ c
            half_w[k] = width - self.car_params[i_ego]["r_col"]
            b = p_opp - p_ego
            if np.linalg.norm(b) >= 1e-6:
                b = b / np.linalg.norm(b)
            beta[k] = b
            beta_opp[k] = b @ p_opp

        if not (self.blocking and lead_0 > 0):
            gamma[:] = 0.0
        sg = np.sqrt(gamma)
        bn = sg[:, None] * n
        self.par_bn.value = bn
        self.par_bn_opp.value = np.sum(bn * trajectories[i_opp], axis=1)

        self.par_p0.value = state[i_ego, :].copy()
        self.par_step.value = self.car_params[i_ego]["v_max"] * self.dt
        self.par_n.value = n
        self.par_nc.value = nc
        self.par_half_w.value = half_w
        self.par_beta.value = beta
        self.par_beta_opp.value = beta_opp
        self.par_t_last.value = t  # tangent from the last loop step (k = N-1)

        self.get_logger().info(f'build: {time.time() - t_build:.4f} s')
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            self.get_logger().warn('solver failed: using previous trajectory')
            return trajectories[i_ego]
        self.get_logger().debug(
            f'compile: {self.prob.compilation_time:.4f} s | '
            f'solve: {self.prob.solver_stats.solve_time} s | '
            f'status: {self.prob.status}'
        )

        if self.p.value is None:  # infeasible -> relaxations come in step 3
            return trajectories[i_ego]
        return self.p.value.copy()

    def best_response_old(self, i_ego, state, trajectories):
        """Based on current trajectories, compute the best-response (BR) of player i_ego.

        This is done by solving an optimization problem over the trajectory
            p_ego[0], p_ego[1], ..., p_ego[N]
        for drone i_ego maximizing the progress along the track while also trying to block the opponent. Here t is the tangent vector that points along the track and N is the horizon length. The resulting optimization is implemented as the following:

        minimize -t^T p_ego[N]

        subject to

          - dynamical constraints
                ||p_ego[k+1] - p_ego[k]|| <= v_max*dt

          - stay-within-track constraints
                |n^T (p[k] - c)| <= width,
                |v^T (p[k] - c)| <= height,
            where v is the track vertical (t x n, where x here represents the cross product) and c is the track center

          - non-collision constraints
                ||p_ego[k] - p_opp[k]|| >= r_coll * 2

        The progress along the track is approximated by the progress along the tangent of the last point of the trajectory, i. e., maximize t^T p_ego[k]. The non-collision constraints are non convex and are linearized here. Instead of requiring the ego drone to stay outside a circle, the drone is constrained to be in a half-plane tangential to that circle. In addition to optimizing track progress, the penalty of violating the safety radius is accounted for. 

        If the blocking term defined in the trajectory parameters is non-zero, we add an additional term to the objective function that incentivizes the drones to slightly adjust their trajectories to block the opponent. Now the objective function is 
        
            -t^T p_ego[N] + sum_k( gamma^k p_rel[k]^T n n^T p_rel[k] )

        where the sum here is over the full trajectory (1, ..., k, ..., N), gamma is the blocking coefficient (a term that is positive when the opponent is behind the ego agent and zero otherwise), and p_rel = p_ego - p_opp is the relative pose vector between the two drones. This is just a heuristic for "blocking" behavior that is only activated when the ego drone is in front of the opponent. 

        :param i_ego: The drone index of the ego drone.
        :param state: The current state (positions) of the two drones.
        :param trajectories: The current trajectories
        :return: Optimized trajectory of player i_ego
        """
        t_build = time.time()
        i_opp = (i_ego + 1) % 2
        v_ego = self.car_params[i_ego]["v_max"]
        r_coll_ego = self.car_params[i_ego]["r_col"]
        r_coll_opp = self.car_params[i_opp]["r_col"]
        r_safe_ego = self.car_params[i_ego]["r_safe"]
        r_safe_opp = self.car_params[i_opp]["r_safe"]
        d_coll = r_coll_ego + r_coll_opp
        d_safe = r_safe_ego + r_safe_opp
        p = cp.Variable(shape=(self.n_steps, 2))

        # === Dynamical Constraints ===
        # ||p_0 - p[0]|| <= v*dt
        init_dyn_constraint = cp.SOC(cp.Constant(v_ego * self.dt), cp.Constant(state[i_ego, :]) - p[0, :])
        # ||p[k+1] - p[k]|| <= v*dt
        dyn_constraints = [init_dyn_constraint] + [
            cp.SOC(cp.Constant(v_ego * self.dt), p[k + 1, :] - p[k, :]) for k in range(self.n_steps - 1)]

        # === Track Constraints ===
        track_constraints = []
        track_obj = cp.Constant(0)
        track_objective_exp = 0.5  # exponentially decreasing weight
        t = np.zeros((self.n_steps, 2))
        n = np.zeros((self.n_steps, 2))
        for k in range(self.n_steps):
            # query track indices at ego position
            idx, c, t[k, :], n[k, :], width = self.track_frame_at(trajectories[i_ego][k, :])
            # hortizontal track height constraints
            # |n·(p[k] - c)| ≤ width  - r_coll
            track_constraints.append(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) <= width - r_coll_ego)
            track_constraints.append(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) >= -(width - r_coll_ego))
            # vertical track height constraints
            # |v·(p[k] - c)| ≤ height - r_coll
            # v = np.cross(t[k, :], n[k, :])  # the vertical direction component of the track
            # track_constraints.append(v.T @ p[k, :] - v.dot(c) <= height - r_coll_ego)
            # track_constraints.append(v.T @ p[k, :] - v.dot(c) >= -(height - r_coll_ego))
            # track constraints objective
            # L1 penalty
            track_obj += (track_objective_exp ** k) * (
                    cp.pos(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) - (width - r_coll_ego)) +
                    cp.pos(-(n[k, :].T @ p[k, :] - np.dot(n[k, :], c) + (width - r_coll_ego))))

        # === Non-Collision Constraints ===
        nc_constraints = []
        nc_obj = cp.Constant(0)
        nc_relax_obj = cp.Constant(0)
        non_collision_objective_exp = 0.5  # exponentially decreasing weight
        for k in range(self.n_steps):
            p_opp = trajectories[i_opp][k, :]
            p_ego = trajectories[i_ego][k, :]
            # Compute beta, the normal direction vector pointing from the ego's drone position to the opponent's
            beta = p_opp - p_ego
            if np.linalg.norm(beta) >= 1e-6:
                # Only normalize if norm is large enough
                beta /= np.linalg.norm(beta)
            #     n.T * (p_opp - p_ego) >= d_coll
            nc_constraints.append(beta.dot(p_opp) - beta.T @ p[k, :] >= d_coll)
            # For normal non-collision objective use safety distance
            nc_obj += (non_collision_objective_exp ** k) * cp.pos(d_safe - (beta.dot(p_opp) - beta.T @ p[k, :]))
            # For relaxed non-collision objective use collision distance
            nc_relax_obj += (non_collision_objective_exp ** k) * cp.pos(d_coll - (beta.dot(p_opp) - beta.T @ p[k, :]))

        # === Blocking Heuristic Objective ===
        blocking_obj = cp.Constant(0)
        blocking_objective_exp = 0.5  # exponentially decreasing weight
        leader_term = np.dot((trajectories[i_ego][0, :] - trajectories[i_opp][0, :]), t[0, :])
        if ( self.blocking and (leader_term > 0.0) ):
            for k in range(self.n_steps):
                p_opp = trajectories[i_opp][k, :]
                # scale factor for leading robot
                p_rel = trajectories[i_ego][k, :] - p_opp
                leader_term = np.dot(p_rel, t[k, :])
                gamma = 0.0
                if (leader_term > 0):
                    gamma = 1.0/(leader_term * leader_term)/(k + 1)
                else:
                    gamma = 0.0
                # add blocking cost function
                # gamma * 0.5^k * (n_k · (p[k] - p_opp))^2
                blocking_obj += gamma  * blocking_objective_exp**k * cp.quad_form(p[k, :] - p_opp, np.outer(n[k, :], n[k, :]))
        
        # === "Win the Race" Objective ===
        # Take the tangent t at the last trajectory point
        # This serves as an approximation to the total track progress
        obj = -t[-1, :].T @ p[-1, :]

        # create the problem in cxvpy and solve it
        prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.blocking_weight * blocking_obj), dyn_constraints + track_constraints + nc_constraints)

        # try to solve proposed problem
        trajectory_result = np.array((self.n_steps, 2))
        try:
            
            self.get_logger().info(f'build: {time.time() - t_build:.4f} s')
            prob.solve()
            self.get_logger().info(
                f'compile: {prob.compilation_time:.4f} s | '
                f'solve: {prob.solver_stats.solve_time} s | '
                f'solver: {prob.solver_stats.solver_name}'
            )


            # relax track constraints if problem is infeasible
            if np.isinf(prob.value):    
                self.get_logger().warn("WARN: relaxing track constraints")
                # If the problem is not feasible, relax track constraints
                # Assert it is indeed an infeasible problem and not unbounded (in which case value is -inf).
                # (The dynamical constraints keep the problem bounded.)
                assert prob.value >= 0.0

                # Solve relaxed problem (track constraint -> track objective)
                relaxed_prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.track_relax_weight * track_obj),
                                        dyn_constraints + nc_constraints)
                relaxed_prob.solve()

                # relax non-collision constraints if problem is still  infeasible
                if np.isinf(relaxed_prob.value):
                    self.get_logger().warn("WARN: relaxing non collision constraints")
                    # If the problem is still infeasible, relax non-collision constraints
                    # Again, assert it is indeed an infeasible problem and not unbounded (in which case value is -inf).
                    # (The dynamical constraints keep the problem bounded.)
                    assert relaxed_prob.value >= 0.0

                    # Solve relaxed problem (non-collision constraint -> non-collision objective)
                    relaxed_prob = cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.nc_relax_weight * nc_relax_obj),
                                            dyn_constraints + track_constraints)
                    relaxed_prob.solve()

                    assert not np.isinf(relaxed_prob.value)
            if p.value is None:
                trajectory_result = trajectories[i_ego]
            else:
                trajectory_result = p.value
        except:  # if cvxpy fails, just return the initialized trajectory to do something
            self.get_logger().warn("WARN: cvxpy failre: resorting to initial trajectory (no collision constraints!)")
            trajectory_result = trajectories[i_ego]
        return trajectory_result


    def track_frame_at(self, p):
        """Find closest track frame to a reference point p.
        :param p: Point of reference
        :return: Index of track frame, track center, tangent and normal.
        """
        i = np.linalg.norm(self.track_centers - p, axis=1).argmin()
        return i, self.track_centers[i], self.track_tangents[i], self.track_normals[i], \
               self.track_widths[i]


    


def main():
    rclpy.init()
    node = IBRPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
