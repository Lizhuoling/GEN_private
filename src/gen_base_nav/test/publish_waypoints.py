from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import FollowWaypoints

class WaypointClient(Node):
    def __init__(self):
        super().__init__('waypoint_client')
        self.client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

    def send_goal(self, waypoints):
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints
        self.client.wait_for_server()
        self.client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    waypoint_client = WaypointClient()

    # 定义路径点
    waypoints = []
    wp1 = PoseStamped()
    wp1.header.frame_id = "map"
    wp1.pose.position.x = -3.0
    wp1.pose.position.y = -3.0
    wp1.pose.orientation.z = 1.0    # sin(theta/2)
    wp1.pose.orientation.w = 0.0    # cos(theta/2)
    waypoints.append(wp1)

    wp2 = PoseStamped()
    wp2.header.frame_id = "map"
    wp2.pose.position.x = 3.0
    wp2.pose.position.y = 3.0
    wp2.pose.orientation.z = 1.0
    wp2.pose.orientation.w = 0.0
    waypoints.append(wp2)

    waypoint_client.send_goal(waypoints)
    rclpy.spin(waypoint_client)

if __name__ == '__main__':
    main()