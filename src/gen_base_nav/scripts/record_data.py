import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, PointCloud2, CameraInfo, Image
from nav_msgs.msg import Path, Odometry
from tf2_msgs.msg import TFMessage
import pdb
import time
import numpy as np
from utils import transform_angular_velocity, quaternion_to_rotation_matrix, traj_frame_reproject

class TF_Subscriber(Node):
    def __init__(self, TF_list):
        super().__init__('data_record_tf_subscriber')
        self.tf_list = TF_list
        self.found_transforms = {}
        self.tf_subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            qos_profile_sensor_data
        )
        self.ready_flag = False
        self.get_logger().info('Waiting for transforms...')

    def tf_callback(self, msg):
        for transform in msg.transforms:
            key = (transform.header.frame_id, transform.child_frame_id)
            if key in self.tf_list:
                self.found_transforms[key] = transform
                self.get_logger().info(f'Found transform from {transform.header.frame_id} to {transform.child_frame_id}')
                if len(self.found_transforms) == len(self.tf_list):
                    self.get_logger().info('All required transforms found. Proceeding with the program.')
                    self.tf_subscription.destroy()
                    self.ready_flag = True
                    
    def get_transform(self, frame_id, child_frame_id):
        key = (frame_id, child_frame_id)
        return self.found_transforms[key] if key in self.found_transforms else None

class TopicSubscriber(Node):
    def __init__(self, topic_list, callback_timeout=0.1):
        super().__init__('topic_subscriber')
        self.topic_list = topic_list
        self.callback_timeout = callback_timeout
        self.message_data = {}
        self.last_callback_time = {}
        self.timeout_flag = False

        for topic in topic_list:
            topic_name, topic_type = topic
            self.create_subscription(
                topic_type,
                topic_name,
                self.create_callback(topic_name),
                qos_profile_sensor_data
            )
            self.last_callback_time[topic_name] = time.time()
            self.get_logger().info(f'Subscribed to topic: {topic_name}')

    def create_callback(self, topic_name):
        def callback(msg):
            self.message_data[topic_name] = msg
            self.last_callback_time[topic_name] = time.time()
        return callback

    def get_topic_data(self, topic_name):
        return self.message_data.get(topic_name, None)

    def check_callbacks(self):
        current_time = time.time()
        for topic_name, last_time in self.last_callback_time.items():
            if current_time - last_time > self.callback_timeout:
                self.timeout_flag = True
                break
        else:
            self.timeout_flag = False

class DataRecorder():
    def __init__(self, topic_subscriber, tf_subscriber):
        self.topic_subscriber = topic_subscriber
        self.tf_subscriber = tf_subscriber
        self.running = True

    def run(self):
        while self.running:
            try:
                if not self.tf_subscriber.ready_flag:
                    rclpy.spin_once(self.tf_subscriber)
                    continue
                rclpy.spin_once(self.topic_subscriber)
                self.topic_subscriber.check_callbacks()
                self.process()
            except KeyboardInterrupt:
                self.running = False
                self.topic_subscriber.get_logger().info('KeyboardInterrupt caught. Shutting down.')
            except ExternalShutdownException:
                self.running = False
                self.topic_subscriber.get_logger().info('External shutdown request received. Shutting down.')
            except Exception as e:
                self.topic_subscriber.get_logger().error(f'Unexpected error: {e}')
                self.running = False

        self.topic_subscriber.destroy_node()
        rclpy.try_shutdown()

    def process(self):
        msg_dict = {}
        for topic in self.topic_subscriber.topic_list:
            topic_name, topic_type = topic
            msg_dict[topic_name] = self.topic_subscriber.get_topic_data(topic_name)

        for topic_name, msg in msg_dict.items():
            if msg is None:
                self.topic_subscriber.get_logger().warning(f'Topic {topic_name} has no data.')
                return None
        
        if self.topic_subscriber.timeout_flag: return None
        
        # Angular velocity and acceleration  
        chassis_imu_data = self.topic_subscriber.get_topic_data('/chassis/imu')
        imu_angular_velocity = chassis_imu_data.angular_velocity
        imu_linear_acceleration = chassis_imu_data.linear_acceleration
        tf_base2imu = self.tf_subscriber.get_transform('base_link', 'chassis_imu')
        assert tf_base2imu.transform.translation.x == 0 and tf_base2imu.transform.translation.y == 0 and tf_base2imu.transform.translation.z == 0, \
            'The base_link and chassis_imu must be aligned in the same position.'
        imu2base_R = quaternion_to_rotation_matrix(tf_base2imu.transform.rotation).T
        base_angular_velocity = transform_angular_velocity(imu_angular_velocity, imu2base_R)
        base_linear_acceleration = transform_angular_velocity(imu_linear_acceleration, imu2base_R)
        # Global planner planner plan
        odom_global_plan = self.topic_subscriber.get_topic_data('/transformed_global_plan').poses
        odom_global_plan_trans = np.array([(ele.pose.position.x, ele.pose.position.y, ele.pose.position.z) for ele in odom_global_plan])
        odom_global_plan_quat = np.array([(ele.pose.orientation.x, ele.pose.orientation.y, ele.pose.orientation.z, ele.pose.orientation.w) for ele in odom_global_plan])
        chassis_odom = self.topic_subscriber.get_topic_data('/chassis/odom')
        chassis_odom_pos, chassis_odom_quat = chassis_odom.pose.pose.position, chassis_odom.pose.pose.orientation
        base_global_plan_trans, base_global_plan_quat = traj_frame_reproject(odom_global_plan_trans, odom_global_plan_quat, chassis_odom_pos, chassis_odom_quat)
        # Local path planner plan
        odom_local_plan = self.topic_subscriber.get_topic_data('/local_plan').poses
        odom_local_plan_trans = np.array([(ele.pose.position.x, ele.pose.position.y, ele.pose.position.z) for ele in odom_local_plan])
        odom_local_plan_quat = np.array([(ele.pose.orientation.x, ele.pose.orientation.y, ele.pose.orientation.z, ele.pose.orientation.w) for ele in odom_local_plan])
        base_global_plan_trans, base_global_plan_quat = traj_frame_reproject(odom_local_plan_trans, odom_local_plan_quat, chassis_odom_pos, chassis_odom_quat)
        # Local planner velocity command
        cmd_vel_nav = self.topic_subscriber.get_topic_data('/cmd_vel_nav')
        cmd_linear = np.array([cmd_vel_nav.linear.x, cmd_vel_nav.linear.y, cmd_vel_nav.linear.z])
        cmd_angular = np.array([cmd_vel_nav.angular.x, cmd_vel_nav.angular.y, cmd_vel_nav.angular.z])
        # Image

def main(args=None):
    rclpy.init(args=args)

    topic_list = [
        ('/chassis/imu', Imu),
        ('/transformed_global_plan', Path),
        ('/chassis/odom', Odometry),
        ('/local_plan', Path),
        ('/cmd_vel_nav', Twist),
        ('/front_3d_lidar/lidar_points', PointCloud2),
        ('/front_stereo_camera/left/camera_info', CameraInfo),
        ('/front_stereo_camera/left/image_raw', Image),
        ('/front_stereo_camera/left/depth_raw', Image),
        ('/back_stereo_camera/left/camera_info', CameraInfo),
        ('/back_stereo_camera/left/image_raw', Image),
        ('/back_stereo_camera/left/depth_raw', Image),
        ('/left_stereo_camera/left/camera_info', CameraInfo),
        ('/left_stereo_camera/left/image_raw', Image),
        ('/left_stereo_camera/left/depth_raw', Image),
        ('/right_stereo_camera/left/camera_info', CameraInfo),
        ('/right_stereo_camera/left/image_raw', Image),
        ('/right_stereo_camera/left/depth_raw', Image),
    ]
    
    TF_list = [
        ('base_link', 'chassis_imu'),
    ]

    topic_subscriber = TopicSubscriber(topic_list)
    tf_subscriber = TF_Subscriber(TF_list)
    data_recorder = DataRecorder(topic_subscriber, tf_subscriber)
    data_recorder.run()

if __name__ == '__main__':
    main()