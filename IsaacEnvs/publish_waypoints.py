import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from math import atan2, cos, sin, sqrt
import time
import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R

ODOM_TOPIC_NAME = '/chassis/odom'
GOAL_TOPIC_NAME = '/goal_pose'
ROBOT_FRAME = 'base_link'

class WaypointClient(Node):
    def __init__(self, total_loops, waypoints, tolerance = 0.3):
        super().__init__('waypoint_client')
        self.total_loops = total_loops
        self.waypoints = waypoints  
        self.tolerance = tolerance
        self.current_loop = 0 
        self.current_waypoint_id = 0
        self.ego_speed, self.ego_x, self.ego_y = None, None, None
        
        self.odom_subscriber = self.create_subscription(Odometry, ODOM_TOPIC_NAME, self.odom_callback, 10)
        self.goal_publisher = self.create_publisher(PoseStamped, GOAL_TOPIC_NAME, 10)  
        self.timer = self.create_timer(0.2, self.send_goal)
        
    def odom_callback(self, odom_msg):
        self.ego_speed = np.array(odom_msg.twist.twist.linear.x, dtype=np.float32)
        self.ego_x = np.array(odom_msg.pose.pose.position.x, dtype=np.float32)
        self.ego_y = np.array(odom_msg.pose.pose.position.y, dtype=np.float32)

    def send_goal(self,):
        if (self.ego_x is None or self.ego_y is None) and rclpy.ok():
            self.get_logger().info("Wait for ego...")
            time.sleep(1.0)
            return False

        if not rclpy.ok():
            return False
        
        if self.current_loop >= self.total_loops:
            self.get_logger().info("All loops have been completed.")
            rclpy.shutdown()
            return True
        
        if self.current_waypoint_id >= len(self.waypoints):
            self.get_logger().info(f"Completed loop {self.current_loop + 1}/{self.total_loops}.")
            self.current_waypoint_id = 0
            self.current_loop += 1
        
        goal_point = self.waypoints[self.current_waypoint_id]
        dx = self.ego_x - goal_point.pose.position.x
        dy = self.ego_y - goal_point.pose.position.y
        distance = sqrt(dx**2 + dy**2)
        if distance < self.tolerance: 
            self.current_waypoint_id += 1
            return True

        self.goal_publisher.publish(goal_point)
        self.get_logger().info(f"To waypoint {self.current_waypoint_id}, Distance: {distance}m, Current: {self.ego_x.item(), self.ego_y.item()}, Goal: {goal_point.pose.position.x, goal_point.pose.position.y}")
            

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

def world_to_odom(init_pos, init_quat_wxyz, world_points):
    '''
    Input:
        world_points: numpy array of shape (n, 3)
        init_pos: (x, y, z)
        init_quat_wxyz: (w, x, y, z)
    '''
    world_points = np.array(world_points, dtype=np.float64)
    init_pos = np.array(init_pos, dtype=np.float64).reshape(3, 1)
    quat_xyzw = np.array([init_quat_wxyz[1], init_quat_wxyz[2], init_quat_wxyz[3], init_quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    R_world_to_init = rot.as_matrix()
    R_odom_from_world = R_world_to_init.T
    t_odom_from_world = -R_odom_from_world @ init_pos
    T_odom_from_world = np.eye(4, dtype=np.float64)
    T_odom_from_world[:3, :3] = R_odom_from_world
    T_odom_from_world[:3, 3] = t_odom_from_world.flatten()
    num_points = world_points.shape[0]
    world_points_hom = np.hstack([world_points, np.ones((num_points, 1))])
    odom_points_hom = (T_odom_from_world @ world_points_hom.T).T
    return odom_points_hom[:, :3]

def main(xy_waypoints, total_loops=1, args=None):
    rclpy.init(args=args)
    
    waypoints = []
    for i in range(len(xy_waypoints)):
        wp = PoseStamped()
        wp.header.frame_id = "odom"
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
    
    waypoint_client = WaypointClient(total_loops, waypoints)
    
    rclpy.spin(waypoint_client)

if __name__ == '__main__':
    # warehouse
    '''robot_init_pos = [-6.0, -1.0, 0.4]
    robot_init_quat = [0.0, 0.0, 0.0, 1.0]  # wxyz
    world_waypoints = [
        (-6.0, 15.3, 0.4),
        (7.1, 15.3, 0.4),
        (7.1, -8.5, 0.4),
        (-8.4, -8.5, 0.4),
        (-6.0, -1.0, 0.4),
    ]'''

    # garden waypoints
    robot_init_pos = [60.1, -8.6, 4.0]
    robot_init_quat = [0.0, 0.0, 0.0, 1.0]  # wxyz
    world_waypoints = [
        (49.2, -4.9, 4.0),
        (56.5, 0.5, 4.0),
        (66.8, 0.5, 4.0),
        (60.1, -8.6, 4.0),
    ]
    
    odom_waypoints = world_to_odom(init_pos = robot_init_pos, init_quat_wxyz = robot_init_quat, world_points = world_waypoints)
    xy_waypoints = odom_waypoints[:, :2].tolist()
    main(xy_waypoints, total_loops = 2)