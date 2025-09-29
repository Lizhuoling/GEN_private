import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class CameraTeleop(Node):
    def __init__(self):
        super().__init__('camera_teleop')

        # 订阅图像
        self.sub = self.create_subscription(
            Image,
            '/front_stereo_camera/left/image_raw',
            self.image_cb,
            10
        )

        # 发布速度
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.br = CvBridge()
        self.twist = Twist()

        # 定时器：20 Hz 检查键盘 + 发布速度
        self.create_timer(0.05, self.timer_cb)

        # 建立一个 OpenCV 窗口用来捕获键盘
        cv2.namedWindow('camera_teleop', cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('Use arrow keys to move, ESC or q to quit.')

    def image_cb(self, msg):
        # 把 ROS Image → OpenCV Mat
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('camera_teleop', frame)
        cv2.waitKey(1)  # 必须调用一次才能刷新窗口并捕获键盘

    def timer_cb(self):
        key = cv2.waitKeyEx(1) & 0xFF
        linear_x  = 0.0
        angular_z = 0.0

        if key == 27 or key == ord('q'):          # ESC 或 q 退出
            rclpy.shutdown()
            return
        elif key == 82 or key == ord('w'):        # ↑ 或 w
            linear_x = 0.5
        elif key == 84 or key == ord('s'):        # ↓ 或 s
            linear_x = -0.5
        elif key == 81 or key == ord('a'):        # ← 或 a
            angular_z = 1.0
        elif key == 83 or key == ord('d'):        # → 或 d
            angular_z = -1.0

        self.twist.linear.x  = linear_x
        self.twist.angular.z = angular_z
        self.pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = CameraTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()