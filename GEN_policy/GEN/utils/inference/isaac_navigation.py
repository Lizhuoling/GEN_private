import pdb
import math
import numpy as np
import time
import cv2
import logging
from threading import Thread, Lock
import torch
from torchvision.transforms import functional as F
import roslibpy
import base64

class TopicSubscriber():
    def __init__(self, topic_list, callback_timeout=0.5, callback_time_min=0.02):
        self.topic_list = topic_list
        self.callback_timeout = callback_timeout
        self.callback_time_min = callback_time_min  # Minimum interval between callback executions
        self.logger = logging.getLogger("GEN")
        
        self.message_data = {}
        self.last_callback_time = {}
        self.last_execution_time = {}  # Track last time callback logic was executed
        self.timeout_flag = False
        self.data_lock = Lock()
        self.ros_connected = False
        
        self.ros_client = roslibpy.Ros(host='localhost', port=9090)
        self.ros_thread = Thread(target=self._ros_loop, daemon=True)
        self.ros_thread.start()
        self._wait_for_connection(10.0)
        
        self.action_publisher = roslibpy.Topic(self.ros_client, '/cmd_vel_policy', 'geometry_msgs/Twist')

    def _ros_loop(self):
        try:
            self.ros_client.run()
            self.ros_connected = False
            self.logger.warning("ROS connection terminated")
        except Exception as e:
            self.ros_connected = False
            self.logger.error(f"ROS loop error: {str(e)}")

    def _wait_for_connection(self, timeout):
        start_time = time.time()
        while not self.ros_connected:
            if time.time() - start_time > timeout:
                raise RuntimeError("ROS connection timeout")
            if hasattr(self.ros_client, 'is_connected') and self.ros_client.is_connected:
                self.ros_connected = True
                self._subscribe_topics()
                self.logger.info("ROS connected")
            time.sleep(0.1)

    def _subscribe_topics(self):
        for topic in self.topic_list:
            topic_name, topic_type = topic
            try:
                ros_topic = roslibpy.Topic(self.ros_client, topic_name, topic_type, queue_size=10)
            except TypeError:
                ros_topic = roslibpy.Topic(self.ros_client, topic_name, topic_type)
            ros_topic.subscribe(self._create_callback(topic_name))
            with self.data_lock:
                self.last_callback_time[topic_name] = time.time()
                self.last_execution_time[topic_name] = 0.0  # Initialize execution time tracker
            self.logger.info(f"Subscribed to {topic_name}")

    def _create_callback(self, topic_name):
        def callback(msg):
            current_time = time.time()
            
            with self.data_lock:
                # Check if minimum interval has passed since last execution
                if current_time - self.last_execution_time[topic_name] < self.callback_time_min:
                    return  # Skip execution if too soon
                
                # Update data only if interval condition is met
                self.message_data[topic_name] = msg
                self.last_callback_time[topic_name] = current_time
                self.last_execution_time[topic_name] = current_time  # Update execution timestamp
        return callback

    def get_topic_data(self, topic_name):
        with self.data_lock:
            return self.message_data.get(topic_name, None)

    def check_callbacks(self):
        current_time = time.time()
        timeout_detected = False
        
        with self.data_lock:
            for topic_name, last_time in self.last_callback_time.items():
                if current_time - last_time > self.callback_timeout:
                    #self.logger.warning(f"Topic {topic_name} timeout: {current_time - last_time:.2f}s")
                    timeout_detected = True
                    break
        
        self.timeout_flag = timeout_detected
        return not timeout_detected

    def close(self):
        if self.ros_connected and hasattr(self.ros_client, 'terminate'):
            self.ros_client.terminate()
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=2.0)

class IsaacNavEnviManager():
    def __init__(self, cfg, policy):
        self.cfg = cfg
        self.policy = policy.eval()
        self.logger = logging.getLogger("GEN")
        
        if cfg['DATA']['MAIN_MODALITY'] == 'image':
            self.topic_subscriber = self.setup_vision_callback()
        elif cfg['DATA']['MAIN_MODALITY'] == 'point':
            self.topic_subscriber = self.setup_point_callback()
        else:
            raise NotImplementedError
        
    def setup_vision_callback(self,):
        vision_subscribe_topic_list = [
            ('/chassis/imu', 'sensor_msgs/msg/Imu'),
            ('/transformed_global_plan', 'nav_msgs/Path'),
            ('/chassis/odom', 'nav_msgs/msg/Odometry'),
            ('/tf', 'tf2_msgs/TFMessage'),
            ('/front_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/left_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/back_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/right_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
        ]
        return TopicSubscriber(vision_subscribe_topic_list)
    
    def setup_point_callback(self,):
        point_subscribe_topic_list = [
            ('/front_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/front_stereo_camera/left/depth_raw', 'sensor_msgs/msg/Image'),
            ('/chassis/odom', 'nav_msgs/msg/Odometry'),
            ('/robot/vel', 'geometry_msgs/Twist'),
            ('/goal_pose', 'geometry_msgs/PoseStamped')
        ]
        return TopicSubscriber(point_subscribe_topic_list)
        
    def inference(self,):
        with torch.no_grad():
            while True:
                self.topic_subscriber.check_callbacks()
                msg_dict = self.get_data()
                
                if msg_dict is None or self.topic_subscriber.timeout_flag: 
                    time.sleep(0.005)
                    continue
                
                policy_input_array = self.get_policy_input(msg_dict)
                policy_input = self.transform_policy_input(policy_input_array)
                
                action = self.policy(None, policy_input['padded_global_plan'], policy_input['padded_global_plan_mask'], policy_input['envi_obs'])
                action = action[0, 0].cpu().numpy().tolist()  # Left shape: (2)
                vel_cmd = {
                    'linear':  {'x': action[0], 'y': 0.0, 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': action[1]}
                }
                self.topic_subscriber.action_publisher.publish(vel_cmd)
                self.logger.info(f'An action published: {vel_cmd}.')
                
    def get_policy_input(self, msg_dict):
        if self.cfg['DATA']['MAIN_MODALITY'] == 'image':
            odom_global_plan_trans, odom_global_plan_quat = self.convert_global_plan_to_array(msg_dict['/transformed_global_plan'])
            chassis_odom_pos = msg_dict['/chassis/odom']['pose']['pose']['position']
            chassis_odom_pos = np.array([chassis_odom_pos['x'], chassis_odom_pos['y'], chassis_odom_pos['z']])
            chassis_odom_quat = msg_dict['/chassis/odom']['pose']['pose']['orientation']
            chassis_odom_quat = np.array([chassis_odom_quat['x'], chassis_odom_quat['y'], chassis_odom_quat['z'], chassis_odom_quat['w']])
            base_global_plan_trans, base_global_plan_quat = self.traj_frame_reproject(odom_global_plan_trans, odom_global_plan_quat, chassis_odom_pos, chassis_odom_quat)
            global_plan = np.concatenate((base_global_plan_trans, base_global_plan_quat), axis=1)    # numpy array of shape (n, 7)
            
            front_img = self.image_to_numpy(msg_dict['/front_stereo_camera/left/image_raw'])
            left_img = self.image_to_numpy(msg_dict['/left_stereo_camera/left/image_raw'])
            back_img = self.image_to_numpy(msg_dict['/back_stereo_camera/left/image_raw'])
            right_img = self.image_to_numpy(msg_dict['/right_stereo_camera/left/image_raw'])
            
            return dict(
                global_plan=global_plan,
                front_cam_rgb=front_img,
                left_cam_rgb=left_img,
                back_cam_rgb=back_img,
                right_cam_rgb=right_img,
            )
        elif self.cfg['DATA']['MAIN_MODALITY'] == 'point':
            pdb.set_trace()
        else:
            raise NotImplementedError
        
    def transform_policy_input(self, policy_input):
        if self.cfg['DATA']['MAIN_MODALITY'] == 'image':
            global_plan = policy_input['global_plan']
            padded_global_plan_mask = np.zeros((self.cfg['DATA']['GLOBAL_PLAN_LENGTH'],), dtype=np.bool)    # Left shape: (GLOBAL_PLAN_LENGTH,)
            if global_plan.shape[0] > self.cfg['DATA']['GLOBAL_PLAN_LENGTH']:
                padded_global_plan = global_plan[:self.cfg['DATA']['GLOBAL_PLAN_LENGTH']]   # Left shape: (GLOBAL_PLAN_LENGTH, 7)
            else:
                padded_global_plan = np.concatenate((global_plan, np.zeros((self.cfg['DATA']['GLOBAL_PLAN_LENGTH'] - global_plan.shape[0], \
                    global_plan.shape[1]), dtype=np.float32)), axis=0)  # Left shape: (GLOBAL_PLAN_LENGTH, 7)
                padded_global_plan_mask[global_plan.shape[0]:] = True
            
            image_list = []
            for camera_name in self.cfg['DATA']['CAMERA_NAMES']:
                image_list.append(policy_input[f'{camera_name}_rgb'][:].astype(np.float32))  # Left shape: (H, W, 3)
            if len(image_list) > 0:
                image_array = np.stack(image_list, axis=0)  # Left shape: (N, H, W, 3)
            else:
                image_array = np.zeros((0, 0, 0, 3), dtype=np.float32)  # Left shape: (1, 1, 1, 3)

            padded_global_plan = torch.from_numpy(padded_global_plan).float()[None]   # left shape: (GLOBAL_PLAN_LENGTH, 7)
            padded_global_plan_mask = torch.from_numpy(padded_global_plan_mask).bool()[None]    # left shape: (GLOBAL_PLAN_LENGTH,)
            image_array = torch.from_numpy(image_array).float()[None]    # Left shape: (1, N, H, W, 3)
            
            padded_global_plan, padded_global_plan_mask, envi_obs = padded_global_plan.cuda(), padded_global_plan_mask.cuda(), image_array.cuda()
            envi_obs = envi_obs.permute(0, 1, 4, 2, 3)
            
            return dict(
                padded_global_plan=padded_global_plan,
                padded_global_plan_mask=padded_global_plan_mask,
                envi_obs=envi_obs,
            )
        elif self.cfg['DATA']['MAIN_MODALITY'] == 'point':
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def get_data(self,):
        msg_dict = {}
        for topic in self.topic_subscriber.topic_list:
            topic_name, topic_type = topic
            msg_dict[topic_name] = self.topic_subscriber.get_topic_data(topic_name)

        for topic_name, msg in msg_dict.items():
            if msg is None:
                self.logger.warning(f'Topic {topic_name} has no data.')
                return None
        return msg_dict
    
    def find_tf_transform(self, tf_transforms, frame_id, child_frame_id):
        for transform in tf_transforms:
            if transform['header']['frame_id'] == frame_id and transform['child_frame_id'] == child_frame_id:
                return transform
        return None
    
    def convert_global_plan_to_array(self, global_plan):
        pos_x = np.array([pose['pose']['position']['x'] for pose in global_plan['poses']])
        pos_y = np.array([pose['pose']['position']['y'] for pose in global_plan['poses']])
        pos_z = np.array([pose['pose']['position']['z'] for pose in global_plan['poses']])
        ori_x = np.array([pose['pose']['orientation']['x'] for pose in global_plan['poses']])
        ori_y = np.array([pose['pose']['orientation']['y'] for pose in global_plan['poses']])
        ori_z = np.array([pose['pose']['orientation']['z'] for pose in global_plan['poses']])
        ori_w = np.array([pose['pose']['orientation']['w'] for pose in global_plan['poses']])
        odom_global_plan_trans = np.array([pos_x, pos_y, pos_z]).T
        odom_global_plan_quat = np.array([ori_x, ori_y, ori_z, ori_w]).T
        return odom_global_plan_trans, odom_global_plan_quat
    
    def traj_frame_reproject(self, P_odom, Q_odom, odom_pos, odom_quat):
        '''
        Description:
            T_odom2base vector (3,) and Q_odom2base quat (4,) are the translation and orientation from the odom frame to the base_link frame.
            P_odom (n, 3) and Q_odom (n, 4) are the position and orientation of the robot in the odom frame. 
            Now we need to calculate the position and orientation of the robot in the base_link frame, returned as P_base and Q_base.
        '''
        P_odom_minus_T = P_odom - odom_pos
        R_odom_base = quaternion_to_rotation_matrix(odom_quat)
        P_base = P_odom_minus_T @ R_odom_base.T
        # Calculate Q_base
        Q_inv = quaternion_inverse(odom_quat)
        Q_base = quaternion_multiply(Q_inv, Q_odom)
        Q_base = Q_base / np.linalg.norm(Q_base, axis=1, keepdims=True)
        return P_base, Q_base
    
    def image_to_numpy(self, message):
        width = message['width']
        height = message['height']
        encoding = message['encoding']
        data = message['data']
        
        if encoding in ['rgb8', 'bgr8']:
            dtype = np.uint8
            channels = 3
        elif encoding in ['mono8', '8UC1']:
            dtype = np.uint8
            channels = 1
        elif encoding in ['rgb16', 'bgr16']:
            dtype = np.uint16
            channels = 3
        elif encoding in ['mono16', '16UC1']:
            dtype = np.uint16
            channels = 1
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        image_array = np.frombuffer(base64.b64decode(data), dtype=dtype)
        image_array = image_array.reshape((height, width, channels))
        
        if encoding == 'bgr8':
            image_array = image_array[..., ::-1]
        
        return image_array
    
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
def quaternion_inverse(q):
    # q must be (x, y, z, w)
    q_inv = np.empty_like(q)
    q_inv[..., 0] = -q[..., 0]
    q_inv[..., 1] = -q[..., 1]
    q_inv[..., 2] = -q[..., 2]
    q_inv[..., 3] = q[..., 3]
    return q_inv

def quaternion_multiply(q1, q2):
    q1_x, q1_y, q1_z, q1_w = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    q2_x, q2_y, q2_z, q2_w = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    x = q1_x * q2_w + q1_y * q2_z - q1_z * q2_y + q1_w * q2_x
    y = -q1_x * q2_z + q1_y * q2_w + q1_z * q2_x + q1_w * q2_y
    z = q1_x * q2_y - q1_y * q2_x + q1_z * q2_w + q1_w * q2_z
    w = -q1_x * q2_x - q1_y * q2_y - q1_z * q2_z + q1_w * q2_w
    
    return np.stack([x, y, z, w], axis=-1)