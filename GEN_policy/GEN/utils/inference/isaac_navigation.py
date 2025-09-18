import pdb
import math
import numpy as np
import time
import cv2
import logging
import torch
from torchvision.transforms import functional as F
import roslibpy

class TopicSubscriber():
    def __init__(self, topic_list, callback_timeout=0.1):
        self.topic_list = topic_list
        self.callback_timeout = callback_timeout
        self.logger = logging.getLogger("GEN")
        
        self.message_data = {}
        self.last_callback_time = {}
        self.timeout_flag = False
        
        self.ros_client = roslibpy.Ros(host='localhost', port=9090)

        for topic in topic_list:
            topic_name, topic_type = topic
            image_topic = roslibpy.Topic(
                self.ros_client,
                topic_name,
                topic_type,
            )
            image_topic.subscribe(self.create_callback(topic_name))
            
            self.last_callback_time[topic_name] = time.time()
            self.logger.info(f'Subscribed to topic: {topic_name}')

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

class IsaacNavEnviManager():
    def __init__(self, cfg, policy):
        self.cfg = cfg
        self.policy = policy.eval()
        self.logger = logging.getLogger("GEN")
        
        if cfg['DATA']['MAIN_MODALITY'] == 'image':
            self.topic_subscriber = self.setup_vision_callback()
            
        self.topic_subscriber.ros_client.run()
        
    def setup_vision_callback(self,):
        vision_subscribe_topic_list = [
            ('/chassis/imu', 'sensor_msgs/msg/Imu'),
            ('/transformed_global_plan', 'nav_msgs/msg/Path'),
            ('/chassis/odom', 'nav_msgs/msg/Odometry'),
            ('/front_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/left_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/back_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
            ('/right_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
        ]
        return TopicSubscriber(vision_subscribe_topic_list)
        
    def inference(self,):
        with torch.no_grad():
            while True:
                data_update_flag =  self.get_data()
                if not data_update_flag or self.topic_subscriber.timeout_flag:
                    continue
                pdb.set_trace()
        
    def get_data(self,):
        msg_dict = {}
        for topic in self.topic_subscriber.topic_list:
            topic_name, topic_type = topic
            msg_dict[topic_name] = self.topic_subscriber.get_topic_data(topic_name)

        for topic_name, msg in msg_dict.items():
            if msg is None:
                self.logger.warning(f'Topic {topic_name} has no data.')
                return False