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
from scipy.spatial.transform import Rotation as R

from rsl_rl.modules import ActorCritic

class TopicProcessor():
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
        
        self.action_publisher = roslibpy.Topic(self.ros_client, '/cmd_vel', 'geometry_msgs/Twist')

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
                self.last_callback_time[topic_name] = 0.0
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
    
    def publish_action(self, action):
        action = action.tolist()
        cmd_vel_x, cmd_vel_y, cmd_angular_z = action[0][0], action[0][1], action[0][2]
        twist_msg = roslibpy.Message({
            'linear': {
                'x': cmd_vel_x,
                'y': cmd_vel_y,
                'z': 0.0
            },
            'angular': {
                'x': 0.0,
                'y': 0.0,
                'z': cmd_angular_z,
            }
        })
        self.action_publisher.publish(twist_msg)

    def close(self):
        if self.ros_connected and hasattr(self.ros_client, 'terminate'):
            self.ros_client.terminate()
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=2.0)
            
class PoseTransformer2D:
    @staticmethod
    def yaw_from_quat(quat):
        """
        Extract yaw angle (rad) from quaternion (ignores pitch/roll for 2D navigation)
        :param quat: np.array [x,y,z,w] - Input quaternion (odom frame)
        :return: Yaw angle (rad) around Z-axis (counterclockwise positive)
        """
        # Convert quaternion to Euler angles (roll, pitch, yaw) in ZYX order (ROS standard)
        rot = R.from_quat(quat)
        euler = rot.as_euler('zyx', degrees=False)  # Z(yaw), Y(pitch), X(roll)
        return euler[0]  # Only return yaw (Z-axis rotation)
    
    @staticmethod
    def yaw_to_rot_matrix_2d(yaw):
        """
        Convert 2D yaw angle to 2x2 rotation matrix (for X/Y plane)
        :param yaw: Yaw angle (rad)
        :return: 2x2 rotation matrix
        """
        return np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
    
    @staticmethod
    def transform_2d_pose(current_pos_2d, current_yaw, target_pos_2d, target_yaw):
        """
        Transform 2D pose from odom frame to base frame (horizontal plane only)
        :param current_pos_2d: np.array [x,y] - Current base position in odom frame
        :param current_yaw: Yaw angle (rad) - Current base orientation in odom frame
        :param target_pos_2d: np.array [x,y] - Goal position in odom frame
        :param target_yaw: Yaw angle (rad) - Goal orientation in odom frame
        :return: (base_goal_pos_2d [x,y], base_goal_yaw) - Goal in base frame
        """
        # Step 1: Calculate position offset in odom frame (target - current)
        pos_offset = target_pos_2d - current_pos_2d
        
        # Step 2: Rotate offset by inverse of current yaw (convert to base frame)
        # Inverse rotation = transpose of rotation matrix (orthogonal property)
        rot_mat = PoseTransformer2D.yaw_to_rot_matrix_2d(current_yaw)
        rot_mat_inv = rot_mat.T  # Inverse for 2D rotation (equivalent to -yaw)
        base_goal_pos_2d = rot_mat_inv @ pos_offset
        
        # Step 3: Calculate relative yaw (goal yaw - current yaw)
        base_goal_yaw = target_yaw - current_yaw
        
        # Normalize yaw to [-π, π] (optional but recommended for navigation)
        base_goal_yaw = np.arctan2(np.sin(base_goal_yaw), np.cos(base_goal_yaw))
        
        return base_goal_pos_2d, base_goal_yaw
            
class point_nav_agent():
    def __init__(self, ckpt_path, topic_processor):
        self.ckpt_path = ckpt_path
        self.topic_processor = topic_processor
        
        self.policy = ActorCritic(num_actor_obs = 54, num_critic_obs = 54, num_actions = 3, inperception_dim = None, perception_in_dim = None, perception_out_dim = None,
                                  actor_hidden_dims = [512, 256, 128], critic_hidden_dims = [512, 256, 128], activation="elu",)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        weight = checkpoint['model_state_dict']
        self.policy.load_state_dict(weight)
        self.policy = self.policy.cuda()
        
        self.last_action = np.zeros((1, 3), dtype=np.float32)
        
    def run(self,):
        while True:
            if not self.topic_processor.check_callbacks():
                print("topic timeout")
                time.sleep(0.05)
                continue
            robot_vel = self.topic_processor.get_topic_data('/robot/vel')
            if robot_vel is None: 
                print("robot_vel has not been received")
            linear_vel = np.array((robot_vel['linear']['x'], robot_vel['linear']['y'], robot_vel['linear']['z']))[None]   # Left shape: (1, 3)
            angular_vel = np.array((robot_vel['angular']['x'], robot_vel['angular']['y'], robot_vel['angular']['z']))[None]   # Left shape: (1, 3)
            
            odom_pose = self.topic_processor.get_topic_data('/chassis/odom')['pose']['pose']
            odom_pos_2d = np.array([odom_pose['position']['x'], odom_pose['position']['y']])  # Ignore Z component
            odom_quat = np.array([odom_pose['orientation']['x'], odom_pose['orientation']['y'], odom_pose['orientation']['z'], odom_pose['orientation']['w']])
            current_yaw = PoseTransformer2D.yaw_from_quat(odom_quat)  # Extract only yaw
            odom_goal_pose = self.topic_processor.get_topic_data('/goal_pose')['pose']
            odom_goal_pos_2d = np.array([odom_goal_pose['position']['x'], odom_goal_pose['position']['y']])  # Ignore Z component
            odom_goal_quat = np.array([odom_goal_pose['orientation']['x'], odom_goal_pose['orientation']['y'], odom_goal_pose['orientation']['z'], odom_goal_pose['orientation']['w']])
            target_yaw = PoseTransformer2D.yaw_from_quat(odom_goal_quat)
            base_goal_pos_2d, base_goal_yaw = PoseTransformer2D.transform_2d_pose(
                current_pos_2d=odom_pos_2d,
                current_yaw=current_yaw,
                target_pos_2d=odom_goal_pos_2d,
                target_yaw=target_yaw
            )
            command = np.array([base_goal_pos_2d[0], base_goal_pos_2d[1], base_goal_yaw])
            command = np.clip(command, a_min = [-3, -5, -np.inf], a_max=[8, 5, np.inf])[None]   # Left shape: (1, 3)
            
            time_left = np.array([0.1,], dtype=np.float32)[None] # Left shape: (1, 1). Progress range: 0~1
            last_action = self.last_action   # Left shape: (1, 3)
            
            scan_data = np.array(self.topic_processor.get_topic_data('/lidar/scan_2d')['data'])[None]   # Left shape: (1, 41)
            
            actor_obs = np.concatenate([linear_vel, angular_vel, command, time_left, last_action, scan_data], axis=-1)     # current_actor_obs shape: (1, 54)
            actor_obs_tensor = torch.Tensor(actor_obs).cuda()
            cmd_vel = self.policy.act_inference(actor_obs_tensor).detach().cpu().numpy()   # Left shape: (1, 3)
            self.last_action = cmd_vel
            
            self.topic_processor.publish_action(cmd_vel)
            
if __name__ == "__main__":
    publish_topic_list = [
        ('/cmd_vel', 'geometry_msgs/msg/Twist'),
    ]
    subscribe_topic_list = [
        ('/chassis/odom', 'nav_msgs/msg/Odometry'),
        ('/robot/vel', 'geometry_msgs/Twist'),
        ('/lidar/scan_2d', 'std_msgs/Float32MultiArray'),
        ('/goal_pose', 'geometry_msgs/msg/PoseStamped')
    ]
    topic_processor = TopicProcessor(subscribe_topic_list)
    
    ckpt_path = '/home/cvte/twilight/code/gymlab/logs/elephant2_flat/2025-09-19_11-35-56_baseline/model_1999.pt'
    policy_manager = point_nav_agent(ckpt_path = ckpt_path, topic_processor = topic_processor)
    policy_manager.run()