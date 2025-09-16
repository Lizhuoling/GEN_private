import roslibpy
import pdb
import numpy as np
import cv2
import base64

def image_to_numpy(message):
    """将sensor_msgs/msg/Image消息转换为NumPy数组"""
    # 提取图像基本信息
    width = message['width']
    height = message['height']
    encoding = message['encoding']
    data = message['data']  # 原始二进制数据
    
    # 根据编码格式处理数据
    if encoding in ['rgb8', 'bgr8']:
        # 8位RGB/BGR图像（3通道）
        dtype = np.uint8
        channels = 3
    elif encoding in ['mono8', '8UC1']:
        # 8位单通道灰度图
        dtype = np.uint8
        channels = 1
    elif encoding in ['rgb16', 'bgr16']:
        # 16位RGB/BGR图像
        dtype = np.uint16
        channels = 3
    elif encoding in ['mono16', '16UC1']:
        # 16位单通道图像
        dtype = np.uint16
        channels = 1
    else:
        raise ValueError(f"不支持的图像编码格式: {encoding}")
    
    image_array = np.frombuffer(base64.b64decode(data), dtype=dtype)
    image_array = image_array.reshape((height, width, channels))
    
    # 如果是BGR格式，可转为RGB（根据需要选择）
    if encoding == 'bgr8':
        image_array = image_array[..., ::-1]  # BGR转RGB
    
    return image_array

def handle_cmd_message(message):
    print("接收到命令消息...")
    print(message)
    pdb.set_trace()

def handle_image_message(message):
    print("接收到图像消息，转换为NumPy数组...")
    
    try:
        img_np = image_to_numpy(message)
        print(f"转换成功！数组形状: {img_np.shape}, 数据类型: {img_np.dtype}")
        pdb.set_trace()
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    ros_client = roslibpy.Ros(host='localhost', port=9090)
    
    '''cmd_vel_topic = roslibpy.Topic(
        ros_client,
        '/cmd_vel',
        'geometry_msgs/msg/Twist'
    )
    cmd_vel_topic.subscribe(handle_cmd_message)'''
    
    image_topic = roslibpy.Topic(
        ros_client,
        '/front_stereo_camera/left/image_raw',
        'sensor_msgs/msg/Image'
    )
    image_topic.subscribe(handle_image_message)
    
    try:
        ros_client.run()
        while True:
            pass
    except KeyboardInterrupt:
        print("程序中断，关闭连接...")
        image_topic.unsubscribe()
        ros_client.terminate()
    