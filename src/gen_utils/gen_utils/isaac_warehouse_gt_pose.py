import pdb
import math
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import GetState
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster

class isaac_warehouse_gt_pose(Node):
    def __init__(self):
        super().__init__('isaac_warehouse_gt_pose')
        self.get_logger().info('isaac_warehouse_gt_pose Node is running...')
        
        # Initial pose in the map frame. 
        self.declare_parameter('initial_pose.x', -6.0)
        self.declare_parameter('initial_pose.y', -1.0)
        self.declare_parameter('initial_pose.z', 0.0)
        self.declare_parameter('initial_pose.yaw', 3.14159)
        init_x = self.get_parameter('initial_pose.x').get_parameter_value().double_value
        init_y = self.get_parameter('initial_pose.y').get_parameter_value().double_value
        init_z = self.get_parameter('initial_pose.z').get_parameter_value().double_value
        init_yaw = self.get_parameter('initial_pose.yaw').get_parameter_value().double_value
        init_roll, init_pitch = 0.0, 0.0
        init_qx = math.sin(init_roll / 2) * math.cos(init_pitch / 2) * math.cos(init_yaw / 2) - math.cos(init_roll / 2) * math.sin(init_pitch / 2) * math.sin(init_yaw / 2)
        init_qy = math.cos(init_roll / 2) * math.sin(init_pitch / 2) * math.cos(init_yaw / 2) + math.sin(init_roll / 2) * math.cos(init_pitch / 2) * math.sin(init_yaw / 2)
        init_qz = math.cos(init_roll / 2) * math.cos(init_pitch / 2) * math.sin(init_yaw / 2) - math.sin(init_roll / 2) * math.sin(init_pitch / 2) * math.cos(init_yaw / 2)
        init_qw = math.cos(init_roll / 2) * math.cos(init_pitch / 2) * math.cos(init_yaw / 2) + math.sin(init_roll / 2) * math.sin(init_pitch / 2) * math.sin(init_yaw / 2)
        self.init_pose = [init_x, init_y, init_z, init_qx, init_qy, init_qz, init_qw]

        self.gps_subscriber = self.create_subscription(
            TFMessage,
            '/chassis/gps',
            self.gps_callback,
            10
        )
        
        #self.gt_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)
        
        # publish map->odom TF.
        self.tf_broadcaster = TransformBroadcaster(self)

    def gps_callback(self, tf_msg):
        for transform in tf_msg.transforms:
            if transform.child_frame_id == 'base_link':
                map_to_odom = TransformStamped()
                map_to_odom.header.stamp = transform.header.stamp
                map_to_odom.header.frame_id = 'map'
                map_to_odom.child_frame_id = 'odom'
                map_to_odom.transform.translation.x = self.init_pose[0]
                map_to_odom.transform.translation.y = self.init_pose[1]
                map_to_odom.transform.translation.z = self.init_pose[2]
                map_to_odom.transform.rotation.x = self.init_pose[3]
                map_to_odom.transform.rotation.y = self.init_pose[4]
                map_to_odom.transform.rotation.z = self.init_pose[5]
                map_to_odom.transform.rotation.w = self.init_pose[6]
                self.tf_broadcaster.sendTransform(map_to_odom)
                
                
                '''gt_pose_msg = PoseWithCovarianceStamped()
                gt_pose_msg.header.stamp = transform.header.stamp
                gt_pose_msg.header.frame_id = 'map'
                gt_pose_msg.pose.pose.position.x = transform.transform.translation.x
                gt_pose_msg.pose.pose.position.y = transform.transform.translation.y
                gt_pose_msg.pose.pose.position.z = transform.transform.translation.z
                gt_pose_msg.pose.pose.orientation.x = transform.transform.rotation.x
                gt_pose_msg.pose.pose.orientation.y = transform.transform.rotation.y
                gt_pose_msg.pose.pose.orientation.z = transform.transform.rotation.z
                gt_pose_msg.pose.pose.orientation.w = transform.transform.rotation.w
                gt_pose_msg.pose.covariance = [0.0] * 36
                self.gt_pose_publisher.publish(gt_pose_msg)'''

def main(args=None):
    rclpy.init(args=args)
    node = isaac_warehouse_gt_pose()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()