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
import cv2
import os
import random
import h5py

from utils import *

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
    def __init__(self, topic_subscriber, tf_subscriber, data_save_path, target_file_num = None, file_id = None, A_init_prob = 1/18, B_init_prob = 0.5, target_ratio = 4.0):
        self.topic_subscriber = topic_subscriber
        self.tf_subscriber = tf_subscriber
        self.data_save_path = data_save_path
        self.target_file_num = target_file_num
        if file_id is not None:
            self.file_id = file_id
        else:
            self.file_id = self.get_max_file_id()
            if self.file_id is None: self.file_id = 0
        
        self.running = True
        self.class_A_num = 0
        self.class_B_num = 0
        self.A_prob = A_init_prob
        self.B_prob = B_init_prob
        self.target_ratio = target_ratio
        self.alpha = 0.1

    def run(self):
        while self.running:
            try:
                if not self.tf_subscriber.ready_flag:
                    rclpy.spin_once(self.tf_subscriber)
                rclpy.spin_once(self.topic_subscriber)
                self.topic_subscriber.check_callbacks()
                self.process()
                if self.target_file_num is not None and self.file_id >= self.target_file_num:
                    self.running = False
                    self.tf_subscriber.get_logger().info('Target file number reached. Shutting down.')
                    break
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
        tf_base2imu = self.tf_subscriber.get_transform(frame_id = 'base_link', child_frame_id = 'chassis_imu')
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
        base_global_plan = np.concatenate((base_global_plan_trans, base_global_plan_quat), axis = 1)
        # Local path planner plan
        odom_local_plan = self.topic_subscriber.get_topic_data('/local_plan').poses
        odom_local_plan_trans = np.array([(ele.pose.position.x, ele.pose.position.y, ele.pose.position.z) for ele in odom_local_plan])
        odom_local_plan_quat = np.array([(ele.pose.orientation.x, ele.pose.orientation.y, ele.pose.orientation.z, ele.pose.orientation.w) for ele in odom_local_plan])
        base_local_plan_trans, base_local_plan_quat = traj_frame_reproject(odom_local_plan_trans, odom_local_plan_quat, chassis_odom_pos, chassis_odom_quat)
        base_local_plan = np.concatenate((base_local_plan_trans, base_local_plan_quat), axis = 1)
        # Local planner velocity command
        cmd_vel_nav = self.topic_subscriber.get_topic_data('/cmd_vel_nav')
        cmd_linear = np.array([cmd_vel_nav.linear.x, cmd_vel_nav.linear.y, cmd_vel_nav.linear.z])
        cmd_angular = np.array([cmd_vel_nav.angular.x, cmd_vel_nav.angular.y, cmd_vel_nav.angular.z])
        # Front camera
        front_cam_rgb = self.rgbmsg_to_cv2(self.topic_subscriber.get_topic_data('/front_stereo_camera/left/image_raw')) # Left shape: (h, w, 3)
        front_cam_depth = self.depthmsg_to_cv2(self.topic_subscriber.get_topic_data('/front_stereo_camera/left/depth_raw')) # Left shape: (h, w)
        front_cam_info = self.topic_subscriber.get_topic_data('/front_stereo_camera/left/camera_info')
        front_cam_intrinsics = front_cam_info.k.reshape(3, 3)
        # Right camera
        right_cam_rgb = self.rgbmsg_to_cv2(self.topic_subscriber.get_topic_data('/right_stereo_camera/left/image_raw')) # Left shape: (h, w, 3)
        right_cam_depth = self.depthmsg_to_cv2(self.topic_subscriber.get_topic_data('/right_stereo_camera/left/depth_raw')) # Left shape: (h, w)
        right_cam_info = self.topic_subscriber.get_topic_data('/right_stereo_camera/left/camera_info')
        right_cam_intrinsics = right_cam_info.k.reshape(3, 3)
        # Left camera
        left_cam_rgb = self.rgbmsg_to_cv2(self.topic_subscriber.get_topic_data('/left_stereo_camera/left/image_raw')) # Left shape: (h, w, 3)
        left_cam_depth = self.depthmsg_to_cv2(self.topic_subscriber.get_topic_data('/left_stereo_camera/left/depth_raw')) # Left shape: (h, w)
        left_cam_info = self.topic_subscriber.get_topic_data('/left_stereo_camera/left/camera_info')
        left_cam_intrinsics = left_cam_info.k.reshape(3, 3)
        # Back camera
        back_cam_rgb = self.rgbmsg_to_cv2(self.topic_subscriber.get_topic_data('/back_stereo_camera/left/image_raw')) # Left shape: (h, w, 3)
        back_cam_depth = self.depthmsg_to_cv2(self.topic_subscriber.get_topic_data('/back_stereo_camera/left/depth_raw')) # Left shape: (h, w)
        back_cam_info = self.topic_subscriber.get_topic_data('/back_stereo_camera/left/camera_info')
        back_cam_intrinsics = back_cam_info.k.reshape(3, 3)
        
        # image and depth visualization
        '''cv2.imwrite('front_cam_rgb.png', np.array(front_cam_rgb[:, :, ::-1]))
        import matplotlib.pyplot as plt
        vis_depth = front_cam_depth.copy()
        vis_depth[vis_depth > 30] = 30
        plt.imshow(vis_depth, cmap='Spectral_r')
        plt.axis('off')
        plt.savefig('front_cam_depth.png', bbox_inches='tight', pad_inches=0)'''

        # point cloud visualization
        '''front_cam_pc_xyzrgb = get_colored_point_cloud(front_cam_rgb, front_cam_depth, front_cam_intrinsics)
        fcam2base_tf = self.tf_subscriber.get_transform(frame_id = 'base_link', child_frame_id = 'front_stereo_camera_left_rgb')
        front_cam_pc_xyzrgb_base = transform_campc2basepc(front_cam_pc_xyzrgb, fcam2base_tf.transform.rotation, fcam2base_tf.transform.translation)
        right_cam_pc_xyzrgb = get_colored_point_cloud(right_cam_rgb, right_cam_depth, right_cam_intrinsics)
        rcam2base_tf = self.tf_subscriber.get_transform(frame_id = 'base_link', child_frame_id = 'right_stereo_camera_left_rgb')
        right_cam_pc_xyzrgb_base = transform_campc2basepc(right_cam_pc_xyzrgb, rcam2base_tf.transform.rotation, rcam2base_tf.transform.translation)
        left_cam_pc_xyzrgb = get_colored_point_cloud(left_cam_rgb, left_cam_depth, left_cam_intrinsics)
        lcam2base_tf = self.tf_subscriber.get_transform(frame_id = 'base_link', child_frame_id = 'left_stereo_camera_left_rgb')
        left_cam_pc_xyzrgb_base = transform_campc2basepc(left_cam_pc_xyzrgb, lcam2base_tf.transform.rotation, lcam2base_tf.transform.translation)
        back_cam_pc_xyzrgb = get_colored_point_cloud(back_cam_rgb, back_cam_depth, back_cam_intrinsics)
        bcam2base_tf = self.tf_subscriber.get_transform(frame_id = 'base_link', child_frame_id = 'rear_stereo_camera_left_rgb')
        back_cam_pc_xyzrgb_base = transform_campc2basepc(back_cam_pc_xyzrgb, bcam2base_tf.transform.rotation, bcam2base_tf.transform.translation)
        pc_xyzrgb = np.concatenate([front_cam_pc_xyzrgb_base, right_cam_pc_xyzrgb_base, left_cam_pc_xyzrgb_base, back_cam_pc_xyzrgb_base], axis=0)
        np.save('pc_xyzrgb.npy', pc_xyzrgb)
        pdb.set_trace()'''
        
        save_flag = False
        if  cmd_angular[-1] < 0.1:  # class A
            if random.random() < self.A_prob:
                save_flag = True
                self.class_A_num += 1
        elif cmd_angular[-1] >= 0.1:  # class B
            if random.random() < self.B_prob:
                save_flag = True
                self.class_B_num += 1
        error = (self.class_A_num / max(1, self.class_B_num) - self.target_ratio) / self.target_ratio
        error = min(error, 3.0)
        self.A_prob = self.A_prob * (1 - self.alpha * error)
        self.A_prob = max(min(self.A_prob, 0.2), 0.002)
        self.B_prob = self.B_prob * (1 + self.alpha * error)
        self.B_prob = max(min(self.B_prob, 0.2), 0.01)
    
        if save_flag:
            while True:
                # Every hdf5 file contains no more than 100 data samples.
                data_writer = H5Writer(file_path = os.path.join(self.data_save_path, "h5py", f'batch_{self.file_id:05d}.hdf5'))
                if data_writer.get_data_sample_num() >= 100:
                    self.file_id += 1
                    data_writer.close()
                else:
                    break
                
            if not data_writer.has_group('transforms'):
                transforms_dict = {}
                for (frame_id, child_frame_id) in self.tf_subscriber.tf_list: 
                    tansform_key = f'{frame_id}2{child_frame_id}'
                    transform = self.tf_subscriber.get_transform(frame_id, child_frame_id)
                    translation = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
                    rotation = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
                    transforms_dict[tansform_key] = {'translation': translation, 'rotation': rotation}
                data_writer.recursive_save_dict(dict(transforms = transforms_dict))
                
            if not data_writer.has_group('samples'):
                data_writer.create_data_group(group_name = 'samples', parent_path = '/')
            sample_id = data_writer.get_data_sample_num()
            data_dict = {f'sample_{sample_id}': {}}
            data_dict[f'sample_{sample_id}'] = dict(
                cur_angular_vel = np.array([base_angular_velocity.x, base_angular_velocity.y, base_angular_velocity.z]),
                cur_linear_acc = np.array([base_linear_acceleration.x, base_linear_acceleration.y, base_linear_acceleration.z]),
                global_plan = base_global_plan,
                local_plan = base_local_plan,
                target_linear = cmd_linear,
                target_angular = cmd_angular,
                front_cam_rgb = front_cam_rgb,
                front_cam_depth = front_cam_depth,
                front_cam_intrinsics = front_cam_intrinsics,
                right_cam_rgb = right_cam_rgb,
                right_cam_depth = right_cam_depth,
                right_cam_intrinsics = right_cam_intrinsics,
                left_cam_rgb = left_cam_rgb,
                left_cam_depth = left_cam_depth,
                left_cam_intrinsics = left_cam_intrinsics,
                back_cam_rgb = back_cam_rgb,
                back_cam_depth = back_cam_depth,
                back_cam_intrinsics = back_cam_intrinsics,
            )
            data_writer.recursive_save_dict(data_dict, parent_key = '/samples')
            print(f'Write sample {sample_id} to file {self.file_id}.')
            
    def get_max_file_id(self,):
        file_list = sorted(os.listdir(os.path.join(self.data_save_path, "h5py")))
        file_list = [ele for ele in file_list if ele.endswith('.hdf5')]
        max_id = None
        for file_name in file_list:
            id = int(file_name.rsplit('.')[0].split('_')[1])
            if max_id is None or id > max_id:
                max_id = id 
        return max_id
        
    def rgbmsg_to_cv2(self, img_msg):
        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)
        if img_msg.encoding in ['mono8', '8UC1']:
            img_buf = img_buf.reshape(img_msg.height, img_msg.width)
        elif img_msg.encoding in ['bgr8', 'rgb8', '8UC3']:
            img_buf = img_buf.reshape(img_msg.height, img_msg.width, 3)
        elif img_msg.encoding in ['bgra8', 'rgba8', '8UC4']:
            img_buf = img_buf.reshape(img_msg.height, img_msg.width, 4)
        else:
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
        return img_buf
    
    def depthmsg_to_cv2(self, img_msg):
        dtype = np.dtype("float32")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)
        if img_msg.encoding == '32FC1':
            img_buf = img_buf.reshape(img_msg.height, img_msg.width)
        else:
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
        return img_buf
    
class H5Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'a')
        
    def has_group(self, name, path = '/'):
        return name in self.file[path].keys()
    
    def recursive_save_dict(self, data_dict, parent_key = '/'):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                if key in self.file[parent_key].keys():
                    raise Exception(f'Key {key} already exists in group {parent_key}')
                else:
                    self.file[parent_key].create_group(key)
                    self.recursive_save_dict(value, parent_key = os.path.join(parent_key, key))
            elif isinstance(value, np.ndarray):
                if key in self.file[parent_key].keys():
                    raise Exception(f'Key {key} already exists in group {parent_key}')
                else:
                    self.file[parent_key].create_dataset(key, data=value, compression='gzip', compression_opts=4)
            else:
                raise ValueError(f'Unsupported type: {type(value)}')
            
    def get_data_sample_num(self,):
        if 'samples' not in self.file.keys():
            return 0
        else:
            return len(self.file['samples'].keys())

    def create_data_group(self, group_name, parent_path = '/'):
        self.file[parent_path].create_group(group_name)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
        ('base_link', 'front_stereo_camera_left_rgb'),
        ('base_link', 'left_stereo_camera_left_rgb'),
        ('base_link', 'right_stereo_camera_left_rgb'),
        ('base_link', 'rear_stereo_camera_left_rgb'),
    ]
    
    data_save_path = '/home/cvte/twilight/data/gen_nav/warehouse'
    topic_subscriber = TopicSubscriber(topic_list)
    tf_subscriber = TF_Subscriber(TF_list)
    data_recorder = DataRecorder(topic_subscriber, tf_subscriber, data_save_path, target_file_num = 10)
    data_recorder.run()
    
if __name__ == '__main__':
    main()