import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import FollowWaypoints
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from math import atan2, cos, sin, sqrt

class WaypointClient(Node):
    def __init__(self, total_loops):
        super().__init__('waypoint_client')
        self.client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.total_loops = total_loops
        self.current_loop = 0 
        self.waypoints = []  
        self.goal_handle = None  

    def send_goal(self, waypoints):
        self.waypoints = waypoints
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints
        
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowWaypoints action server not available')
            return False

        self.client.send_goal_async(goal_msg).add_done_callback(
            self.goal_response_callback
        )
        return True

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info(f'Goal accepted, starting loop {self.current_loop + 1}/{self.total_loops}')
        self.goal_handle.get_result_async().add_done_callback(
            self.result_callback
        )

    def result_callback(self, future):
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.current_loop += 1
            self.get_logger().info(f'Loop {self.current_loop} completed successfully')
            
            if self.current_loop < self.total_loops:
                self.get_logger().info('Starting next loop...')
                self.send_goal(self.waypoints)
            else:
                self.get_logger().info(f'All {self.total_loops} loops completed')
                rclpy.shutdown()
        else:
            self.get_logger().error(f'Loop {self.current_loop + 1} failed with status: {status}')
            rclpy.shutdown()

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
    qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
    qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
    qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
    
    return [qx, qy, qz, qw]

def main(xy_waypoints, total_loops=1, args=None):
    rclpy.init(args=args)
    
    waypoints = []
    for i in range(len(xy_waypoints)):
        wp = PoseStamped()
        wp.header.frame_id = "map"
        wp.header.stamp = rclpy.clock.Clock().now().to_msg()
        wp.pose.position.x = xy_waypoints[i][0]
        wp.pose.position.y = xy_waypoints[i][1]
        
        # Calculate the orientation to point towards the next waypoint
        if i < len(xy_waypoints) - 1:
            next_x, next_y = xy_waypoints[i + 1]
        else:
            next_x, next_y = xy_waypoints[0]  # Loop back to the first waypoint
        
        dx = next_x - xy_waypoints[i][0]
        dy = next_y - xy_waypoints[i][1]
        angle_to_next = atan2(dy, dx)
        
        q = quaternion_from_euler(0, 0, angle_to_next)
        wp.pose.orientation.x = q[0]
        wp.pose.orientation.y = q[1]
        wp.pose.orientation.z = q[2]
        wp.pose.orientation.w = q[3]
        
        waypoints.append(wp)
    
    waypoint_client = WaypointClient(total_loops)
    if not waypoint_client.send_goal(waypoints):
        rclpy.shutdown()
        return
    
    rclpy.spin(waypoint_client)

if __name__ == '__main__':
    xy_waypoints = [
        (-7.2, -1.0),
        (-7.0, 11.0),
        (-8.2, 16.2),
        (7.5, 16.4),
        (7.5, -10.1),
        (-8.3, -10.5),
        (-8.5, -5.5),
        (-5.4, -4.3),
        (-5.6, -1.3),
    ]
    main(xy_waypoints, total_loops=1)