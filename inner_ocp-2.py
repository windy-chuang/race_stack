#!/usr/bin/env python3
"""
Standalone inner OCP of Spica et al. 2018 ("A Real-Time Game Theoretic Planner for
Autonomous Two-Player Drone Racing"), problem (16)/(9) -- single player, opponent
trajectory FIXED (no iterated best response).

    max   s(p^N)                                   progress along the centerline
    s.t.  p^k = p^{k-1} + u^k,   ||u^k|| <= vmax*dt
          -w_r(s_k) <= n(p^k)^T (p^k - tau(p^k)) <= w_l(s_k)      track
          ||p^k - q^k|| >= d_min                                   collision (q = opponent)

Solved by sequential QCLP (convexify around iterate theta^m):
    collision : beta^T (q^k - p^k) >= d_min,   beta = (q^k - p^k_m)/||q^k - p^k_m||
    track     : n_m^T (p^k - tau_m) in [-w_r, w_l]  (n, tau from projection of p^k_m)
    cost      : -sigma_m^T p^N,  sigma = t^T / (1 - kappa (p^N_m - tau)^T n)      (eq. 18)

Robustness additions (the usual reasons the linearised problem is infeasible):
    * slack variables with L1 penalty on track and collision constraints
    * optional trust region + step halving on a true (non-linearised) merit function
    * every iteration is logged (solver status, slack, true violations)

Usage:  python3 inner_ocp.py global_waypoints.json   (runs inner OCP + IBR demo)
"""
import json
import sys
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from scipy.interpolate import CubicSpline


# --------------------------------------------------------------------------- track
class Track:
    """Closed track: periodic spline of the centerline, arclength parameterised."""

    def __init__(self, x, y, d_left, d_right, s=None, margin=0.0, ds_grid=0.02):
        x, y = np.asarray(x, float), np.asarray(y, float)
        pts = np.c_[x, y]
        seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        s_all = np.r_[0.0, np.cumsum(seg)]            # closes the loop
        self.L = float(s_all[-1])
        P = np.vstack([pts, pts[:1]])
        self.spl = CubicSpline(s_all, P, bc_type="periodic")
        dl = np.r_[d_left, d_left[:1]]
        dr = np.r_[d_right, d_right[:1]]
        self.wl = CubicSpline(s_all, np.maximum(dl - margin, 0.0), bc_type="periodic")
        self.wr = CubicSpline(s_all, np.maximum(dr - margin, 0.0), bc_type="periodic")
        self.sg = np.arange(0.0, self.L, ds_grid)      # dense grid for projection
        self.pg = self.spl(self.sg)

    # geometry at arclength s (wraps)
    def pos(self, s):
        return self.spl(np.mod(s, self.L))

    def tangent(self, s):
        d = self.spl(np.mod(s, self.L), 1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def normal(self, s):                               # LEFT normal
        t = self.tangent(s)
        return np.stack([-t[..., 1], t[..., 0]], axis=-1)

    def curvature(self, s):                            # signed, + = left turn
        d1 = self.spl(np.mod(s, self.L), 1)
        d2 = self.spl(np.mod(s, self.L), 2)
        return (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / \
               np.linalg.norm(d1, axis=-1) ** 3

    def project(self, p, s_hint=None, window=None):
        """Closest centerline arclength to point p (refined by Newton)."""
        p = np.asarray(p, float)
        if s_hint is None or window is None:
            i = np.argmin(np.sum((self.pg - p) ** 2, axis=1))
            s = self.sg[i]
        else:
            ds = np.arange(-window, window, self.sg[1] - self.sg[0]) + s_hint
            q = self.spl(np.mod(ds, self.L))
            s = ds[np.argmin(np.sum((q - p) ** 2, axis=1))]
        for _ in range(5):                             # Newton on d/ds ||c(s)-p||^2 = 0
            c, c1, c2 = (self.spl(np.mod(s, self.L), k) for k in (0, 1, 2))
            g = c1 @ (c - p)
            h = c2 @ (c - p) + c1 @ c1
            if abs(h) < 1e-9:
                break
            s -= np.clip(g / h, -0.1, 0.1)
        return float(np.mod(s, self.L))

    def frenet(self, p, s_hint=None, window=None):
        s = self.project(p, s_hint, window)
        n = self.normal(s)
        tau = self.pos(s)
        return s, tau, self.tangent(s), n, float(self.curvature(s)), float((p - tau) @ n)

    @staticmethod
    def from_json(path, margin=0.0):
        d = json.load(open(path))
        w = d["centerline_waypoints"]["wpnts"]
        return Track(
            [q["x_m"] for q in w], [q["y_m"] for q in w],
            np.array([q["d_left"] for q in w]), np.array([q["d_right"] for q in w]),
            margin=margin)


# --------------------------------------------------------------------------- OCP
@dataclass
class Params:
    N: int = 20              # horizon steps
    dt: float = 0.1          # step [s]
    vmax: float = 4.0        # max speed [m/s]
    d_min: float = 0.5       # min centre-to-centre distance [m]
    track_margin: float = 0.15   # car half-width, removed from the track half-widths [m]
    max_iter: int = 40
    tol: float = 1e-4        # ||theta^{m+1}-theta^m||
    trust: float = 2.0       # trust-region radius on each p^k [m]  (None = off)
    slack_w: float = 1e3     # L1 penalty on constraint slacks
    solver: str = "CLARABEL"
    verbose: bool = False


@dataclass
class Result:
    p: np.ndarray                    # (N+1, 2) positions, p[0] = start
    u: np.ndarray                    # (N, 2)
    s_progress: float                # arclength gained (unwrapped)
    status: str
    iters: int
    max_track_viol: float            # true violations of the final solution [m] (<=0 OK)
    max_coll_viol: float
    mu: np.ndarray = None            # (N,) multipliers of the collision constraints (>=0)
    beta: np.ndarray = None          # (N,2) unit vector ego->opponent used in the constraint
    log: list = field(default_factory=list)


def predict_opponent(track, q0, N, dt, v_opp):
    """Constant speed along the centerline tangent (paper's MPC-baseline opponent)."""
    s0 = track.project(q0)
    q = [np.asarray(q0, float)]
    for _ in range(N):
        s = track.project(q[-1], s_hint=s0, window=1.0)
        q.append(q[-1] + v_opp * dt * track.tangent(s))
        s0 = s
    return np.array(q)


def true_violations(track, p, q, prm, s_hint):
    """Non-linearised constraint violations (positive = violated)."""
    tv, cv = [], []
    for k in range(1, len(p)):
        s, _, _, _, _, lat = track.frenet(p[k], s_hint, 2.0 + prm.vmax * prm.dt * prm.N)
        tv.append(max(lat - float(track.wl(s)), -lat - float(track.wr(s))))
        cv.append(prm.d_min - np.linalg.norm(p[k] - q[k]))
    return max(tv), max(cv)


def merit(track, p, q, prm, s0, bonus=None):
    s_end = track.project(p[-1], s0, 2.0 + prm.vmax * prm.dt * prm.N)
    ds = (s_end - s0 + track.L / 2) % track.L - track.L / 2
    tv, cv = true_violations(track, p, q, prm, s0)
    b = 0.0 if bonus is None else bonus(p)
    return -ds - b + prm.slack_w * (max(tv, 0) + max(cv, 0)), ds


def warm_start(track, p0, prm, frac=0.7):
    """Drive along the centerline from the ego's projection at frac*vmax. The QCLP has
    slacks, so this need not be feasible; pass theta0=np.tile(p0,(N+1,1)) for 'stand still'."""
    s = track.project(p0)
    p = [np.asarray(p0, float)]
    for k in range(1, prm.N + 1):
        c = track.pos(s + frac * prm.vmax * prm.dt * k)
        p.append(c + (p[0] - track.pos(track.project(p0))) * max(0.0, 1 - k / 5))  # blend lateral offset out
    return np.array(p)


def solve_inner_ocp(track, p_ego, q_traj, prm=Params(), theta0=None, alpha=0.0, mu=None):
    """
    p_ego   : (2,)       start position of the ego car
    q_traj  : (N+1, 2)   opponent positions over the horizon (q_traj[0] = current)
    theta0  : optional (N+1,2) initial guess of p (in IBR: the player's previous iterate).
    alpha   : aggressiveness gain of eq. (16). alpha=0 -> plain inner OCP.
    mu      : (N,) multipliers mu_j^{k,l} of the OPPONENT's collision constraints from its last
              solve (Result.mu of the opponent). None -> zeros.
    Objective (maximised):  s_i(p^N) + alpha * sum_k mu^k beta^{kT} p^k,
    beta^k = (q^k - theta0^k)/||q^k - theta0^k||  (fixed for the whole solve, as in the paper).
    """
    N = prm.N
    p0 = np.asarray(p_ego, float)
    q = np.asarray(q_traj, float)
    assert q.shape == (N + 1, 2), f"q_traj must be {(N + 1, 2)}, got {q.shape}"
    s0 = track.project(p0)
    win = 2.0 + prm.vmax * prm.dt * N
    umax = prm.vmax * prm.dt
    if theta0 is None:
        theta0 = warm_start(track, p0, prm)
    p_m = np.asarray(theta0, float).copy()
    r = prm.trust
    log, status = [], "not_run"
    Lmat = np.tril(np.ones((N, N)))                    # p^k = p0 + sum_{j<=k} u^j
    mu_in = np.zeros(N) if mu is None else np.asarray(mu, float)
    d0 = q[1:] - p_m[1:]
    beta_obj = d0 / np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-6)
    w_obj = alpha * mu_in[:, None] * beta_obj          # (N,2): objective gradient wrt p^k
    bonus = (lambda p: float(np.sum(w_obj * p[1:]))) if alpha != 0 else None
    cur, _ = merit(track, p_m, q, prm, s0, bonus)

    # def merit(track, p, q, prm, s0, bonus=None):
        # s_end = track.project(p[-1], s0, 2.0 + prm.vmax * prm.dt * prm.N)
        # ds = (s_end - s0 + track.L / 2) % track.L - track.L / 2
        # tv, cv = true_violations(track, p, q, prm, s0)
        # b = 0.0 if bonus is None else bonus(p)
        # return -ds - b + prm.slack_w * (max(tv, 0) + max(cv, 0)), ds

    mu_out, beta_out = np.zeros(N), None

    for m in range(prm.max_iter):
        # ---- linearise around p_m
        fr = [track.frenet(p_m[k], s0 if k == 0 else None, None) for k in range(N + 1)]
        # NOTE: global projection (no hint) is robust to leaving the window; cheap enough.
        tau = np.array([f[1] for f in fr]); n = np.array([f[3] for f in fr])
        wl = np.array([float(track.wl(f[0])) for f in fr])
        wr = np.array([float(track.wr(f[0])) for f in fr])
        sN, tauN, tN, nN, kN, latN = fr[N][0], fr[N][1], fr[N][2], fr[N][3], fr[N][4], fr[N][5]
        den = 1.0 - kN * latN
        if den < 0.1:                                  # past the centre of curvature
            den = 0.1
        sigma = tN / den

        diff = q[1:] - p_m[1:]
        dist = np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-6)
        beta = diff / dist                              # (N,2)

        # ---- QCLP
        u = cp.Variable((N, 2))
        st = cp.Variable(N, nonneg=True)               # track slack
        sc = cp.Variable(N, nonneg=True)               # collision slack
        P = p0 + Lmat @ u                              # (N,2) = p^1..p^N
        cons = [cp.norm(u, 2, axis=1) <= umax]
        lat = cp.sum(cp.multiply(n[1:], P - tau[1:]), axis=1)
        cons += [lat <= wl[1:] + st, -lat <= wr[1:] + st]
        coll = prm.d_min - sc - cp.sum(cp.multiply(beta, q[1:] - P), axis=1) <= 0
        cons += [coll]
        if r is not None:
            cons += [cp.norm(P - p_m[1:], 2, axis=1) <= r]
        obj = cp.Minimize(-sigma @ P[N - 1] - cp.sum(cp.multiply(w_obj, P)) + prm.slack_w * (cp.sum(st) + cp.sum(sc)))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=prm.solver, verbose=prm.verbose)
        except cp.SolverError as e:
            status = f"solver_error: {e}"
            log.append(dict(it=m, status=status))
            break
        if prob.status not in ("optimal", "optimal_inaccurate"):
            status = f"qclp_{prob.status}"
            log.append(dict(it=m, status=status, trust=r))
            if r is not None and r > 1e-3:             # shrink and retry
                r *= 0.5
                continue
            break

        p_new = np.vstack([p0, P.value])
        # ---- accept/reject with the true merit (trust-region step control)
        new, ds = merit(track, p_new, q, prm, s0, bonus)
        step = np.linalg.norm(p_new - p_m)
        if r is not None and new > cur + 1e-9 and step > prm.tol:
            r *= 0.5
            log.append(dict(it=m, status="rejected", merit=new, cur=cur, trust=r))
            if r < 1e-3:
                status = "converged_stationary"    # no descent on true merit at tiny radius
                break
            continue
        gain, p_m, cur = cur - new, p_new, new
        mu_out = np.maximum(np.asarray(coll.dual_value).ravel(), 0.0); beta_out = beta
        if r is not None:
            r = min(2 * r, prm.trust)                  # grow back after a good step
        log.append(dict(it=m, status=prob.status, ds=ds, slack=float(st.value.sum() + sc.value.sum()),
                        step=step, trust=r, merit=new))
        if step < prm.tol or (m > 0 and abs(gain) < 1e-4):
            status = "converged"
            break
    else:
        status = "max_iter"

    tv, cv = true_violations(track, p_m, q, prm, s0)
    s_end = track.project(p_m[-1], s0, win)
    ds = (s_end - s0 + track.L / 2) % track.L - track.L / 2
    return Result(p_m, np.diff(p_m, axis=0), float(ds), status, len(log), tv, cv,
                  mu_out, beta_out if beta_out is not None else beta_obj, log)


# --------------------------------------------------------------------------- IBR
@dataclass
class IBRResult:
    p1: np.ndarray
    p2: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    converged: bool
    history: list = field(default_factory=list)     # per IBR iteration diagnostics
    res1: Result = None
    res2: Result = None


def ibr(track, x1, x2, prm=Params(), alpha=(0.0, 0.0), n_ibr=8, tol=1e-2,
        theta1=None, theta2=None, prm2=None):
    """
    Iterated best response (Fig. 2 of the paper): alternately solve (16) for player 1 and 2.
        theta_1^l = argmax  s_1 + alpha_1 * sum_k mu_2^{k,l-1} beta_12^{kT} p_1^k   (theta_2 fixed)
        theta_2^l = argmax  s_2 + alpha_2 * sum_k mu_1^{k,l}   beta_21^{kT} p_2^k   (theta_1 fixed)
    mu_j = multipliers of player j's collision constraint from its own last solve.
    theta1/theta2 : initial guesses (default: centerline warm start; pass last plan shifted by one
                    step for receding-horizon use).
    Returns both trajectories, the multipliers and a per-iteration history.
    """
    prm2 = prm if prm2 is None else prm2
    N = prm.N
    th1 = warm_start(track, np.asarray(x1, float), prm) if theta1 is None else np.asarray(theta1, float)
    th2 = warm_start(track, np.asarray(x2, float), prm2) if theta2 is None else np.asarray(theta2, float)
    mu1, mu2 = np.zeros(N), np.zeros(N)
    hist, conv, r1, r2 = [], False, None, None
    for l in range(n_ibr):
        r1 = solve_inner_ocp(track, x1, th2, prm, theta0=th1, alpha=alpha[0], mu=mu2)
        d1 = np.linalg.norm(r1.p - th1)
        th1, mu1 = r1.p, r1.mu
        r2 = solve_inner_ocp(track, x2, th1, prm2, theta0=th2, alpha=alpha[1], mu=mu1)
        d2 = np.linalg.norm(r2.p - th2)
        th2, mu2 = r2.p, r2.mu
        hist.append(dict(l=l, d1=float(d1), d2=float(d2), s1=r1.s_progress, s2=r2.s_progress,
                         st1=r1.status, st2=r2.status, mu1=float(mu1.sum()), mu2=float(mu2.sum()),
                         coll=float(max(r1.max_coll_viol, r2.max_coll_viol))))
        if max(d1, d2) < tol:
            conv = True
            break
    return IBRResult(th1, th2, mu1, mu2, conv, hist, r1, r2)


# --------------------------------------------------------------------------- demo
def demo(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prm = Params()
    track = Track.from_json(path, margin=prm.track_margin)
    print(f"track length {track.L:.2f} m")
    sg = np.linspace(0, track.L, 800)
    c, nn = track.pos(sg), track.normal(sg)

    # player 1 (ego) starts 0.6 m behind player 2, both near the centerline
    se, so = 5.0, 5.2
    x1 = track.pos(se) + 0.0 * track.normal(se)
    x2 = track.pos(so) + 0.2 * track.normal(so)

    # (a) plain inner OCP: opponent = constant-velocity prediction (no IBR)
    q = predict_opponent(track, x2, prm.N, prm.dt, 3.0)
    ra = solve_inner_ocp(track, x1, q, prm)
    print(f"\n(a) inner OCP only : {ra.status}, progress {ra.s_progress:.2f} m")

    cases = {"(b) IBR alpha=(0,0)": (0.0, 0.0), "(c) IBR alpha=(1,1)": (1.0, 1.0),
             "(d) IBR alpha=(1,0)": (1.0, 0.0)}
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    out = {"(a) inner OCP only": (ra.p, q)}
    for name, al in cases.items():
        R = ibr(track, x1, x2, prm, alpha=al, n_ibr=8)
        out[name] = (R.p1, R.p2)
        print(f"\n{name}: converged={R.converged} after {len(R.history)} IBR iterations")
        for h in R.history:
            print("   l={l}  |d1|={d1:.3f} |d2|={d2:.3f}  s1={s1:.2f} s2={s2:.2f}  "
                  "sum mu1={mu1:.2f} sum mu2={mu2:.2f}  coll_viol={coll:+.3f}  [{st1}/{st2}]".format(**h))
    for ax, (name, (a, b)) in zip(axs, out.items()):
        ax.plot(*c.T, "k--", lw=.5)
        ax.plot(*(c + nn * np.array(track.wl(sg))[:, None]).T, "g", lw=1)
        ax.plot(*(c - nn * np.array(track.wr(sg))[:, None]).T, "g", lw=1)
        ax.plot(*a.T, "b.-", label="player 1")
        ax.plot(*b.T, "r.-", label="player 2" if "IBR" in name else "opp pred")
        ax.plot(*a[0], "bo"); ax.plot(*b[0], "ro")
        ax.set_aspect("equal")
        mid = (x1 + x2) / 2
        ax.set_xlim(mid[0] - 4, mid[0] + 4); ax.set_ylim(mid[1] - 4, mid[1] + 4)
        ax.set_title(name); ax.legend()
    plt.tight_layout(); plt.savefig("ibr_demo.png", dpi=100)
    print("\nsaved ibr_demo.png")


if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "global_waypoints.json")
