import copy

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray


class GBFollower(Node):
    """
    Stand-in for the state machine: publishes the global raceline ahead of the car
    as local waypoints with scaled speed. Ignores opponents entirely.
    """

    def __init__(self):
        super().__init__('gb_follower')
        self.rate_hz = self.declare_parameter('rate_hz', 40).value
        self.speed_scale = self.declare_parameter('speed_scale', 0.3).value
        self.n_loc_wpnts = self.declare_parameter('n_loc_wpnts', 80).value
        
        
        self.wpnts = None
        self.wpnt_dist = None
        self.cur_s = None

        self.create_subscription(WpntArray, '/global_waypoints_scaled', self.gb_cb, 10)
        self.create_subscription(Odometry, 'car_state/frenet/odom', self.state_cb, 10)
        self.loc_pub = self.create_publisher(WpntArray, 'local_waypoints', 10)
        self.create_timer(1 / self.rate_hz, self.loop)

    def gb_cb(self, msg: WpntArray):
        # copy so scaling never compounds on repeated messages
        wpnts = copy.deepcopy(msg.wpnts)
        for wpnt in wpnts:
            wpnt.vx_mps *= self.speed_scale
        self.wpnts = wpnts
        self.wpnt_dist = wpnts[1].s_m - wpnts[0].s_m

    def state_cb(self, msg: Odometry):
        self.cur_s = msg.pose.pose.position.x

    def loop(self):
        if self.wpnts is None or self.cur_s is None:
            return
        n = len(self.wpnts)
        start = int(self.cur_s / self.wpnt_dist + 0.5)
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.wpnts = [self.wpnts[(start + i) % n] for i in range(self.n_loc_wpnts)]
        self.loc_pub.publish(out)


def main():
    rclpy.init()
    node = GBFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
