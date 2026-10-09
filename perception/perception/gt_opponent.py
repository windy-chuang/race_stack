from f110_msgs.msg import WpntArray, ObstacleArray, Obstacle
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

class GTOpponent(Node):
    def __init__(self):
        super().__init__('gt_opponent')
        self.opp_ns = self.declare_parameter('opp_ns', 'opp').value
        self.size = self.declare_parameter('size', 0.5).value
        self.track_length = None
        self.create_subscription(WpntArray, '/global_waypoints', self.wpnts_cb, 10)
        # absolute topic -> the OTHER car's state
        self.create_subscription(Odometry, f'/{self.opp_ns}/car_state/frenet/odom', self.opp_cb, 10)
        # relative topic -> lands in THIS car's namespace
        self.obs_pub = self.create_publisher(ObstacleArray, 'perception/obstacles', 5)
        self.opp_pub = self.create_publisher(Odometry, 'perception/opponent', 5)

    def wpnts_cb(self, msg):
        self.track_length = msg.wpnts[-1].s_m

    def opp_cb(self, msg):
        if self.track_length is None:
            return
        s = msg.pose.pose.position.x % self.track_length
        d = msg.pose.pose.position.y
        obs = Obstacle()
        obs.id = 0
        obs.size = self.size
        obs.s_center, obs.d_center = s, d
        obs.s_start = (s - self.size / 2) % self.track_length
        obs.s_end   = (s + self.size / 2) % self.track_length
        obs.d_right, obs.d_left = d - self.size / 2, d + self.size / 2
        obs.vs = msg.twist.twist.linear.x
        obs.vd = msg.twist.twist.linear.y
        obs.is_static, obs.is_visible, obs.is_actually_a_gap = False, True, False
        out = ObstacleArray()
        out.header = msg.header
        out.obstacles = [obs]
        self.obs_pub.publish(out)

        opp_odom = Odometry()
        opp_odom.header = msg.header
        opp_odom.pose.pose.orientation.w = 1.0
        opp_odom.pose.pose.position.x = s
        opp_odom.pose.pose.position.y = d
        opp_odom.twist.twist.linear.x = msg.twist.twist.linear.x
        opp_odom.twist.twist.linear.y = msg.twist.twist.linear.y
        self.opp_pub.publish(opp_odom)


def main():
    rclpy.init()
    node = GTOpponent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
