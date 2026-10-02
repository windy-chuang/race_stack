#!/usr/bin/env python3
"""
ROS 2 node that drives one car with the IBR planner (see ibr_core.py).

- plan loop  (plan_rate_hz):    runs the IBR game for both cars, keeps the ego trajectory
- control loop (control_rate_hz): tracks the latest trajectory with race_stack's
  PP_Controller and publishes an AckermannDriveStamped

Default topics drive the *second* car of the f1tenth gym bridge (num_agent: 2), so the
normal race_stack pipeline can keep driving the first car.
"""
import os

import numpy as np
import rclpy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from controller.pp import PP_Controller
from f110_msgs.msg import Wpnt, WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32

from ibr_planner.ibr_core import IBRPlanner, Track2D

WAYPOINT_SPACING = 0.1  # [m], PP_Controller assumes waypoints 0.1 m apart


def yaw_from_quat(q) -> float:
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class IBRNode(Node):
    def __init__(self):
        super().__init__('ibr_planner')

        # topics
        self.declare_parameter('ego_odom_topic', '/opp_racecar/odom')
        self.declare_parameter('opp_odom_topic', '/opp_racecar/opp_odom')
        self.declare_parameter('drive_topic', '/opp_drive')
        self.declare_parameter('global_waypoints_topic', '/global_waypoints')
        self.declare_parameter('racecar_version', 'SIM')
        # loop rates
        self.declare_parameter('plan_rate_hz', 10.0)
        self.declare_parameter('control_rate_hz', 40.0)
        self.declare_parameter('plan_timeout_s', 0.5)
        # IBR
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('n_steps', 20)
        self.declare_parameter('n_game_iters', 2)
        self.declare_parameter('n_sqp_iters', 3)
        self.declare_parameter('blocking', False)
        self.declare_parameter('solver', '')
        self.declare_parameter('ego_v_max', 6.0)
        self.declare_parameter('ego_r_coll', 0.3)
        self.declare_parameter('ego_r_safe', 0.5)
        self.declare_parameter('opp_v_max', 6.0)
        self.declare_parameter('opp_r_coll', 0.3)
        self.declare_parameter('opp_r_safe', 0.5)
        self.declare_parameter('opp_v_from_odom', True)
        self.declare_parameter('opp_v_margin', 1.1)
        self.declare_parameter('opp_v_min', 1.0)
        # tracking
        self.declare_parameter('path_extension_m', 6.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.plan_timeout = p('plan_timeout_s')
        self.path_extension = p('path_extension_m')
        self.opp_v_from_odom = p('opp_v_from_odom')
        self.opp_v_margin = p('opp_v_margin')
        self.opp_v_min = p('opp_v_min')
        self.ibr_settings = dict(
            dt=p('dt'), n_steps=p('n_steps'), blocking=p('blocking'),
            n_game_iters=p('n_game_iters'), n_sqp_iters=p('n_sqp_iters'),
            solver=p('solver') or None)
        self.car_params = [
            {"v_max": p('ego_v_max'), "r_coll": p('ego_r_coll'), "r_safe": p('ego_r_safe')},
            {"v_max": p('opp_v_max'), "r_coll": p('opp_r_coll'), "r_safe": p('opp_r_safe')},
        ]

        # data containers
        self.planner = None
        self.ego = None  # [x, y, yaw, speed]
        self.opp = None
        self.local_wpnts = None  # (N, 8) array in the PP_Controller format
        self.last_plan_time = None

        self.pp = self.init_pp_controller(p('racecar_version'), p('control_rate_hz'))

        # subscribers
        self.create_subscription(WpntArray, p('global_waypoints_topic'), self.gb_cb, 10)
        self.create_subscription(Odometry, p('ego_odom_topic'), self.ego_odom_cb, 10)
        self.create_subscription(Odometry, p('opp_odom_topic'), self.opp_odom_cb, 10)

        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p('drive_topic'), 10)
        self.path_pub = self.create_publisher(Path, '/ibr/path', 10)
        self.wpnts_pub = self.create_publisher(WpntArray, '/ibr/local_waypoints', 10)
        self.solve_time_pub = self.create_publisher(Float32, '/ibr/solve_time', 10)

        # separate callback groups so the controller keeps running while IBR is solving
        self.create_timer(1 / p('plan_rate_hz'), self.plan_loop,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1 / p('control_rate_hz'), self.control_loop,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info('[IBR] waiting for global waypoints and odometry')

    def init_pp_controller(self, racecar_version: str, rate: float) -> PP_Controller:
        """Same setup as controller_manager.init_pp_controller, without the ROS node."""
        config_dir = os.path.join(get_package_share_directory('stack_master'), 'config', racecar_version)
        with open(os.path.join(config_dir, 'l1_params.yaml'), 'r') as f:
            l1 = yaml.safe_load(f)['controller']['ros__parameters']
        with open(os.path.join(config_dir, 'sim_params.yaml'), 'r') as f:
            car = yaml.safe_load(f)
        return PP_Controller(
            l1["t_clip_min"], l1["t_clip_max"], l1["m_l1"], l1["q_l1"],
            l1["speed_lookahead"], l1["lat_err_coeff"],
            l1["acc_scaler_for_steer"], l1["dec_scaler_for_steer"],
            l1["start_scale_speed"], l1["end_scale_speed"], l1["downscale_factor"],
            l1["speed_lookahead_for_steer"],
            l1["prioritize_dyn"], l1["trailing_gap"],
            l1["trailing_p_gain"], l1["trailing_i_gain"], l1["trailing_d_gain"],
            l1["blind_trailing_speed"],
            rate, car['lr'] + car['lf'],
            logger_info=self.get_logger().info,
            logger_warn=self.get_logger().warn)

    #############
    # CALLBACKS #
    #############
    def gb_cb(self, msg: WpntArray):
        if self.planner is not None or len(msg.wpnts) == 0:
            return  # the global trajectory is republished periodically, build only once
        track = Track2D(
            x=[w.x_m for w in msg.wpnts], y=[w.y_m for w in msg.wpnts],
            d_left=[w.d_left for w in msg.wpnts], d_right=[w.d_right for w in msg.wpnts],
            v_ref=[w.vx_mps for w in msg.wpnts])
        self.planner = IBRPlanner(track, self.car_params, **self.ibr_settings)
        self.get_logger().info(f'[IBR] track built from {track.n_points} waypoints')

    @staticmethod
    def odom_to_state(msg: Odometry):
        pose = msg.pose.pose
        speed = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        return np.array([pose.position.x, pose.position.y, yaw_from_quat(pose.orientation), speed])

    def ego_odom_cb(self, msg: Odometry):
        self.ego = self.odom_to_state(msg)

    def opp_odom_cb(self, msg: Odometry):
        self.opp = self.odom_to_state(msg)

    ########
    # PLAN #
    ########
    def plan_loop(self):
        if self.planner is None or self.ego is None or self.opp is None:
            return
        ego, opp = self.ego.copy(), self.opp.copy()

        # the opponent is a spliner car, not an IBR player: bound its speed by what it actually does
        if self.opp_v_from_odom:
            self.planner.car_params[1]["v_max"] = max(opp[3] * self.opp_v_margin, self.opp_v_min)

        state = np.array([ego[:2], opp[:2]])
        trajectory = self.planner.iterative_br(0, state)
        trajectory = trajectory[self.planner.truncate(ego[:2], trajectory):]
        if self.planner.n_fallbacks > 0:
            self.get_logger().warn(f'[IBR] solver failed {self.planner.n_fallbacks}x, used previous guess')
        if len(trajectory) == 0:
            self.get_logger().warn('[IBR] whole trajectory truncated, keeping previous plan')
            return

        self.local_wpnts = self.to_pp_waypoints(ego[:2], trajectory)
        self.last_plan_time = self.get_clock().now()
        self.solve_time_pub.publish(Float32(data=float(self.planner.solve_time)))
        self.publish_plan(self.local_wpnts)

    def to_pp_waypoints(self, ego_xy, trajectory) -> np.ndarray:
        """Resample [car position, IBR trajectory, raceline continuation] to 0.1 m spacing.

        The raceline continuation keeps the PP lookahead point valid when the IBR horizon is short.
        :return: (N, 8) array [x, y, v, norm_bound, s, kappa, psi, ax] as used by PP_Controller
        """
        track = self.planner.track
        dt = self.planner.dt
        pts = np.vstack((ego_xy, trajectory))
        seg_v = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt
        v = np.concatenate(([seg_v[0]], seg_v))

        idx_end = track.frame_at(pts[-1])[0]
        n_ext = int(self.path_extension / WAYPOINT_SPACING)
        ext_idx = (idx_end + 1 + np.arange(n_ext)) % track.n_points
        ext_v = [self.planner.speed_limit(0, i) for i in ext_idx]
        pts = np.vstack((pts, track.centers[ext_idx]))
        v = np.concatenate((v, ext_v))

        # drop duplicate points so the arc length is strictly increasing
        keep = np.concatenate(([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6))
        pts, v = pts[keep], v[keep]
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        s_new = np.arange(0.0, s[-1], WAYPOINT_SPACING)
        x = np.interp(s_new, s, pts[:, 0])
        y = np.interp(s_new, s, pts[:, 1])
        v_new = np.interp(s_new, s, v)
        psi = np.arctan2(np.gradient(y), np.gradient(x))
        kappa = np.gradient(np.unwrap(psi)) / WAYPOINT_SPACING
        out = np.zeros((len(s_new), 8))
        out[:, 0], out[:, 1], out[:, 2] = x, y, v_new
        out[:, 3] = 0.5
        out[:, 4], out[:, 5], out[:, 6] = s_new, kappa, psi
        return out

    ###########
    # CONTROL #
    ###########
    def control_loop(self):
        wpnts, ego = self.local_wpnts, self.ego
        stale = self.last_plan_time is None or \
            (self.get_clock().now() - self.last_plan_time).nanoseconds * 1e-9 > self.plan_timeout
        if wpnts is None or ego is None or stale:
            if self.last_plan_time is not None:
                self.get_logger().warn('[IBR] no recent plan, stopping', throttle_duration_sec=1.0)
            self.publish_drive(0.0, 0.0)
            return

        # signed lateral error w.r.t. the planned path (PP uses it to stretch the lookahead)
        i = int(np.argmin(np.sum((wpnts[:, :2] - ego[:2]) ** 2, axis=1)))
        psi = wpnts[i, 6]
        d = -np.sin(psi) * (ego[0] - wpnts[i, 0]) + np.cos(psi) * (ego[1] - wpnts[i, 1])

        speed, _, _, steer, _, _, _ = self.pp.main_loop(
            "GB_TRACK",                          # no trailing logic, IBR handles the opponent
            np.array([ego[:3]]),                 # position_in_map [x, y, yaw]
            wpnts,
            ego[3],                              # speed_now
            None,                                # opponent
            np.array([wpnts[i, 4], d, ego[3], 0.0]),  # position_in_map_frenet [s, d, vs, vd]
            np.zeros(10),                        # acc_now, no IMU in sim
            None)                                # track_length, only used for trailing
        self.publish_drive(float(speed), float(steer))

    #############
    # PUBLISHERS #
    #############
    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = speed
        msg.drive.steering_angle = steer
        self.drive_pub.publish(msg)

    def publish_plan(self, wpnts: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = 'map'
        wpnt_array = WpntArray()
        wpnt_array.header = path.header
        for k, (x, y, v, _, s, kappa, psi, _) in enumerate(wpnts):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            path.poses.append(pose)
            wpnt_array.wpnts.append(Wpnt(id=k, s_m=float(s), x_m=float(x), y_m=float(y),
                                         psi_rad=float(psi), kappa_radpm=float(kappa), vx_mps=float(v)))
        self.path_pub.publish(path)
        self.wpnts_pub.publish(wpnt_array)


def main(args=None):
    rclpy.init(args=args)
    node = IBRNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
