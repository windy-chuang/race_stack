"""
Iterative Best Response (IBR) planner for two cars, ported from the AirSim
NeurIPS 2019 drone racing baseline (baselines/gtp.py).

Changes w.r.t. the drone version:
- 2D instead of 3D (no height constraints)
- the track comes from race_stack global waypoints instead of gate poses
- asymmetric track bounds (d_left / d_right) instead of a symmetric gate width
- the per-step speed limit follows the raceline speed profile (capped by v_max)
- infeasibility is detected via the solver status instead of asserts
- the convex problems are built once with cvxpy Parameters and only re-filled at
  every call; rebuilding them took ~50 ms per solve versus ~1 ms for solving

This file has no ROS dependencies so it can be tested on its own.
"""
import time

import cvxpy as cp
import numpy as np


class Track2D:
    """Closed track built from race_stack global waypoints (replaces gtp.SplinedTrack)."""

    def __init__(self, x, y, d_left, d_right, v_ref):
        self.centers = np.column_stack((x, y)).astype(float)
        # tangents by central differences on the closed loop
        diff = np.roll(self.centers, -1, axis=0) - np.roll(self.centers, 1, axis=0)
        norms = np.linalg.norm(diff, axis=1)
        norms[norms < 1e-9] = 1.0
        self.tangents = diff / norms[:, np.newaxis]
        # normal points to the left, same convention as frenet d > 0
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        self.d_left = np.asarray(d_left, dtype=float)
        self.d_right = np.asarray(d_right, dtype=float)
        self.v_ref = np.asarray(v_ref, dtype=float)
        self.n_points = len(self.centers)

    def frame_at(self, p):
        """Closest track frame to point p.
        :return: index, center, tangent, normal, d_left, d_right
        """
        i = int(np.argmin(np.sum((self.centers - p[:2]) ** 2, axis=1)))
        return i, self.centers[i], self.tangents[i], self.normals[i], self.d_left[i], self.d_right[i]


class IBRPlanner:
    """
    Given the positions of both cars, iteratively computes collision-free trajectories
    that stay within the track and respect a maximum speed. Each trajectory is
    'n_steps' points spaced 'dt' seconds apart.

    See gtp.IBRController in the AirSim repo for the full explanation of the method:
    best responses are computed alternately for both players, each one a sequential
    convex program in which the non-collision constraint is linearized around the
    current guess. If a problem is infeasible, the track and then the non-collision
    constraints are relaxed into penalties.

    car_params: list of two dicts with keys
        v_max   maximal speed [m/s]
        r_coll  collision radius [m]
        r_safe  safety radius [m] (penalized, not enforced)
    """

    def __init__(self, track: Track2D, car_params, dt=0.1, n_steps=20, blocking=False,
                 n_game_iters=2, n_sqp_iters=3, solver=None):
        self.track = track
        self.car_params = car_params
        self.dt = dt
        self.n_steps = n_steps
        self.blocking = blocking
        self.n_game_iters = n_game_iters
        self.n_sqp_iters = n_sqp_iters
        self.solver = solver

        # weights, same values as the drone version
        self.nc_weight = 2.0
        self.nc_relax_weight = 128.0
        self.track_relax_weight = 128.0
        self.blocking_weight = 16.0

        # diagnostics of the last call to iterative_br
        self.n_fallbacks = 0
        self.n_relaxations = 0
        self.solve_time = 0.0
        self.last_trajectories = None

        self._build_problems()

    def _build_problems(self):
        """Build the best-response problem and its two relaxations once.

        minimize   -t^T p[N] + safety penalty (+ blocking penalty)
        subject to ||p[k+1] - p[k]|| <= v_k * dt                     (dynamics)
                   -(d_right - r) <= n^T (p[k] - c) <= d_left - r    (track)
                   beta^T (p_opp[k] - p[k]) >= d_coll                (linearized non-collision)
        """
        N = self.n_steps
        p = cp.Variable((N, 2))
        par = {
            "p0": cp.Parameter(2),                 # current position
            "step": cp.Parameter(N, nonneg=True),  # max distance per step, v_k * dt
            "normals": cp.Parameter((N, 2)),       # track normal n_k
            "lat_c": cp.Parameter(N),              # n_k^T c_k
            "upper": cp.Parameter(N),              # d_left - r_coll
            "lower": cp.Parameter(N),              # -(d_right - r_coll)
            "beta": cp.Parameter((N, 2)),          # unit vector ego -> opponent
            "beta_opp": cp.Parameter(N),           # beta_k^T p_opp[k]
            "t_last": cp.Parameter(2),             # tangent at the last point
            "blk_n": cp.Parameter((N, 2)),         # sqrt(w_k) n_k
            "blk_c": cp.Parameter(N),              # sqrt(w_k) n_k^T p_opp[k]
            "d_coll": cp.Parameter(nonneg=True),
            "d_safe": cp.Parameter(nonneg=True),
        }
        decay = 0.5 ** np.arange(N)  # exponentially decreasing weights

        prev = cp.vstack([cp.reshape(par["p0"], (1, 2), order="C"), p[:-1, :]])
        dyn_constraints = [cp.norm(p - prev, 2, axis=1) <= par["step"]]

        lateral = cp.sum(cp.multiply(par["normals"], p), axis=1) - par["lat_c"]
        track_constraints = [lateral <= par["upper"], lateral >= par["lower"]]
        track_obj = decay @ (cp.pos(lateral - par["upper"]) + cp.pos(par["lower"] - lateral))

        dist = par["beta_opp"] - cp.sum(cp.multiply(par["beta"], p), axis=1)
        nc_constraints = [dist >= par["d_coll"]]
        nc_obj = decay @ cp.pos(par["d_safe"] - dist)
        nc_relax_obj = decay @ cp.pos(par["d_coll"] - dist)

        # (n_k^T (p[k] - p_opp[k]))^2 weighted by w_k, zero weights when not leading
        blocking_obj = cp.sum_squares(cp.sum(cp.multiply(par["blk_n"], p), axis=1) - par["blk_c"])

        # "Win the Race": progress along the tangent at the last point
        obj = -par["t_last"] @ p[-1, :]

        self._p = p
        self._par = par
        self._problems = [
            cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.blocking_weight * blocking_obj),
                       dyn_constraints + track_constraints + nc_constraints),
            # relax track constraints into an objective
            cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.track_relax_weight * track_obj),
                       dyn_constraints + nc_constraints),
            # relax non-collision constraints into an objective
            cp.Problem(cp.Minimize(obj + self.nc_weight * nc_obj + self.nc_relax_weight * nc_relax_obj),
                       dyn_constraints + track_constraints),
        ]

    def speed_limit(self, i_car, track_idx):
        """Raceline speed at track_idx, capped by the car's v_max."""
        v_max = self.car_params[i_car]["v_max"]
        v_ref = self.track.v_ref[track_idx]
        return min(v_max, v_ref) if v_ref > 0.0 else v_max

    def init_trajectory(self, i_car, p_0):
        """Initial guess: follow the track tangent at the speed limit."""
        trajectory = np.zeros(shape=(self.n_steps, 2))
        p = np.array(p_0[:2], dtype=float)
        for k in range(self.n_steps):
            idx, _, t, _, _, _ = self.track.frame_at(p)
            p = p + self.dt * self.speed_limit(i_car, idx) * t
            trajectory[k, :] = p
        return trajectory

    def _solve(self, prob):
        try:
            prob.solve(solver=self.solver, warm_start=True) if self.solver else prob.solve(warm_start=True)
        except cp.error.SolverError:
            return False
        return prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    def best_response(self, i_ego, state, trajectories):
        """Best response of car i_ego to the current trajectory of the other car."""
        i_opp = (i_ego + 1) % 2
        ego, opp = self.car_params[i_ego], self.car_params[i_opp]
        r_ego = ego["r_coll"]
        N = self.n_steps
        traj_ego, traj_opp = trajectories[i_ego], trajectories[i_opp]

        step = np.zeros(N)
        t = np.zeros((N, 2))
        n = np.zeros((N, 2))
        lat_c = np.zeros(N)
        upper = np.zeros(N)
        lower = np.zeros(N)
        for k in range(N):
            idx, c, t[k, :], n[k, :], d_left, d_right = self.track.frame_at(traj_ego[k, :])
            step[k] = self.speed_limit(i_ego, idx) * self.dt
            lat_c[k] = n[k, :].dot(c)
            upper[k] = max(d_left - r_ego, 0.0)
            lower[k] = -max(d_right - r_ego, 0.0)

        beta = traj_opp - traj_ego
        norms = np.linalg.norm(beta, axis=1)
        beta[norms >= 1e-6] /= norms[norms >= 1e-6, np.newaxis]

        # blocking heuristic, only active while ego is ahead
        blk_w = np.zeros(N)
        leader_term = np.dot(traj_ego[0, :] - traj_opp[0, :], t[0, :])
        if self.blocking and leader_term > 0.0:
            for k in range(N):
                leader_term = np.dot(traj_ego[k, :] - traj_opp[k, :], t[k, :])
                if leader_term > 0:
                    blk_w[k] = 1.0 / (leader_term * leader_term) / (k + 1) * 0.5 ** k
        blk_n = np.sqrt(blk_w)[:, np.newaxis] * n

        par = self._par
        par["p0"].value = np.asarray(state[i_ego, :2], dtype=float)
        par["step"].value = step
        par["normals"].value = n
        par["lat_c"].value = lat_c
        par["upper"].value = upper
        par["lower"].value = lower
        par["beta"].value = beta
        par["beta_opp"].value = np.sum(beta * traj_opp, axis=1)
        par["t_last"].value = t[-1, :]
        par["blk_n"].value = blk_n
        par["blk_c"].value = np.sum(blk_n * traj_opp, axis=1)
        par["d_coll"].value = ego["r_coll"] + opp["r_coll"]
        par["d_safe"].value = ego["r_safe"] + opp["r_safe"]

        for i, prob in enumerate(self._problems):
            if self._solve(prob):
                self.n_relaxations += min(i, 1)
                return self._p.value.copy()

        # nothing worked: keep the previous guess (has no collision guarantee)
        self.n_fallbacks += 1
        return traj_ego

    def iterative_br(self, i_ego, state):
        """Run the IBR game and return the trajectory of car i_ego, shape (n_steps, 2).
        :param state: positions of both cars, shape (2, >=2)
        """
        self.n_fallbacks = 0
        self.n_relaxations = 0
        t0 = time.time()
        trajectories = [self.init_trajectory(i, state[i, :]) for i in (0, 1)]
        for _ in range(self.n_game_iters - 1):
            for i in (i_ego, (i_ego + 1) % 2):
                for _ in range(self.n_sqp_iters - 1):
                    trajectories[i] = self.best_response(i, state, trajectories)
        # one last time for i_ego
        for _ in range(self.n_sqp_iters):
            trajectories[i_ego] = self.best_response(i_ego, state, trajectories)
        self.solve_time = time.time() - t0
        self.last_trajectories = trajectories
        return trajectories[i_ego]

    def truncate(self, p_i, trajectory):
        """Index of the first trajectory point that is ahead of p_i along the track tangent."""
        _, _, t, _, _, _ = self.track.frame_at(p_i)
        truncate_distance = 0.01
        for k in range(len(trajectory)):
            if t.dot(trajectory[k, :] - p_i[:2]) > truncate_distance:
                return k
        return len(trajectory)
