import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, PointCloud2, CameraInfo, Image
from nav_msgs.msg import Path
import pdb

class TopicSubscriber(Node):
    def __init__(self, topic_list):
        super().__init__('topic_subscriber')
        self.topic_list = topic_list
        self.message_data = {}

        # Dynamically create subscriptions for each topic in the list
        for topic in topic_list:
            topic_name, topic_type = topic
            self.create_subscription(
                topic_type,
                topic_name,
                self.create_callback(topic_name),
                qos_profile_sensor_data
            )
            self.get_logger().info(f'Subscribed to topic: {topic_name}')

    def create_callback(self, topic_name):
        # Create a callback function for each topic
        def callback(msg):
            self.message_data[topic_name] = msg
        return callback

    def get_topic_data(self, topic_name):
        return self.message_data.get(topic_name, None)

class DataRecorder():
    def __init__(self, topic_subscriber):
        self.topic_subscriber = topic_subscriber
        self.running = True

    def run(self):
        while self.running:
            try:
                rclpy.spin_once(self.topic_subscriber)
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
        cmd_vel_nav_msg = self.topic_subscriber.get_topic_data('/cmd_vel_nav')
        local_plan_msg = self.topic_subscriber.get_topic_data('/local_plan')
        if local_plan_msg is not None and cmd_vel_nav_msg is not None:
            print(len(local_plan_msg.poses))

def main(args=None):
    rclpy.init(args=args)

    # List of topics to subscribe to, in the format [(topic_name, topic_type), ...]
    topic_list = [
        ('/cmd_vel_nav', Twist),
        ('/local_plan', Path),
        ('/chassis_imu', Imu),
        ('/transformed_global_plan', Path),
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

    topic_subscriber = TopicSubscriber(topic_list)
    data_recorder = DataRecorder(topic_subscriber)
    data_recorder.run()

if __name__ == '__main__':
    main()