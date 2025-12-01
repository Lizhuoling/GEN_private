# Run "source /home/cvte/twilight/environment/Isaac_Sim_5.0/setup_conda_env.sh" before running this script.
# Run ``python main.py --enable_cameras" to start this simulation environment.
# To import 3D GS based scene, must use IsaacLab v2.2.0.

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab Manipulation Environment.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import random
import pdb
import numpy as np
import logging
import math
import torch
import base64
import roslibpy
import time
import open3d as o3d
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R
from twisted.internet import reactor

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import CameraCfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG

def _in_reactor(fn, *a, **kw):
    reactor.callFromThread(fn, *a, **kw)

class TopicProcessor():
    def __init__(self, subscribe_topic_list, publish_topic_list, callback_timeout=0.1, callback_time_min=0.02, max_workers=None):
        self.subscribe_topic_list = subscribe_topic_list
        self.publish_topic_list = publish_topic_list
        self.callback_timeout = callback_timeout
        self.callback_time_min = callback_time_min
        self.logger = logging.getLogger("Isaac")
        
        self.message_data = {}
        self.subsrciber_dict = {}
        self.publisher_dict = {}
        self.data_lock = Lock()
        self.ros_connected = False
        
        # Use MessagePack for more efficient serialization
        self.ros_client = roslibpy.Ros(host='localhost', port=9090,)
        
        # Thread pool for parallel callback processing
        self.max_workers = max_workers or len(self.subscribe_topic_list)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.ros_thread = Thread(target=self._ros_loop, daemon=True)
        self.ros_thread.start()
        self._wait_for_connection(10.0)

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
                raise RuntimeError("ROS connection timeout, remember to first run 'ros2 launch rosbridge_server rosbridge_websocket_launch.xml' to start ROS.")
            if hasattr(self.ros_client, 'is_connected') and self.ros_client.is_connected:
                self.ros_connected = True
                self._subscribe_topics()
                self._publish_topics()
                self._publish_tf()
                self.logger.info("ROS connected")
            time.sleep(0.1)

    def _subscribe_topics(self):
        for topic in self.subscribe_topic_list:
            topic_name, topic_type = topic
            try:
                # Add queue_size and throttle_rate parameters
                ros_topic = roslibpy.Topic(
                    self.ros_client, 
                    topic_name, 
                    topic_type, 
                    queue_size=10,
                    throttle_rate=int(self.callback_time_min * 1000)  # Convert to ms
                )
            except TypeError:
                raise Exception(f"Fail to subscribe to {topic_name}")
            
            ros_topic.subscribe(self._create_callback(topic_name))
            self.subsrciber_dict[topic_name] = ros_topic
            
            self.logger.info(f"Subscribed to {topic_name}")
            
    def _publish_topics(self):
        for topic in self.publish_topic_list:
            topic_name, topic_type = topic
            try:
                # Add queue_size and throttle_rate parameters
                ros_topic = roslibpy.Topic(
                    self.ros_client, 
                    topic_name, 
                    topic_type, 
                    queue_size=10,
                )
            except TypeError:
                raise Exception(f"Fail to publish to {topic_name}")
            self.publisher_dict[topic_name] = ros_topic

            self.logger.info(f"Published to {topic_name}")
            
    def _publish_tf(self,):
        ros_topic = roslibpy.Topic(
            self.ros_client, 
            '/tf',
            'tf2_msgs/msg/TFMessage'
        )
        self.publisher_dict['/tf'] = ros_topic
        self.logger.info(f"Published to /tf")

    def _create_callback(self, topic_name):
        def callback(msg):
            current_time = time.time()
            
            with self.data_lock:
                self.message_data[topic_name] = msg
        
        return callback

    def shutdown(self):
        """Properly clean up resources"""
        self.ros_client.terminate()
        self.executor.shutdown(wait=False)
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        self.logger.info("Subscriber shutdown complete")
        
    def get_message(self, topic_name):
        with self.data_lock:
            if topic_name in self.message_data:
                return self.message_data[topic_name]
            else:
                return None
            
    def publish_rgb_image(self, rgb_image, topic_name):
        if rgb_image.dtype != np.uint8:
            raise ValueError(f"The RGB image must be of type uint8, but got {rgb_image.dtype}")
        height, width, channels = rgb_image.shape
        if channels != 3:
            raise ValueError(f"The RGB image must have 3 channels, but got {channels}")
        image_data = base64.b64encode(bytes(rgb_image.tobytes())).decode('utf-8')
        msg = roslibpy.Message({
            'header': {
                'stamp': roslibpy.Time.now(),
                'frame_id': 'hand_cam_link'
            },
            'height': height,
            'width': width,
            'encoding': 'rgb8',
            'is_bigendian': 0,
            'step': width * 3,
            'data': image_data
        })
        _in_reactor(self.publisher_dict[topic_name].publish, msg)
        
    def publish_depth_image(self, depth_image, topic_name):
        if depth_image.dtype != np.float32:
            raise ValueError(f"The depth image must be of type float32, but got {depth_image.dtype}")
        height, width = depth_image.shape
        image_data = base64.b64encode(bytes(depth_image.astype('<f4').tobytes())).decode('utf-8')
        msg = roslibpy.Message({
            'header': {
                'stamp': roslibpy.Time.now(),
                'frame_id': 'hand_cam_link'
            },
            'height': height,
            'width': width,
            'encoding': '32FC1',
            'is_bigendian': 0,
            'step': width * 4,
            'data': image_data
        })
        _in_reactor(self.publisher_dict[topic_name].publish, msg)
        
    def publish_pose(self, pos, quat, topic_name):
        msg = roslibpy.Message({
            'header': { 
                'stamp': roslibpy.Time.now(),
                'frame_id': 'world',
            },
            'pose': {
                'position': {
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                },
                'orientation': {
                    'x': quat[1],
                    'y': quat[2],
                    'z': quat[3],
                    'w': quat[0],
                }
            }
        })
        _in_reactor(self.publisher_dict[topic_name].publish, msg)
        
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


@configclass
class ManipulationSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/mount",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", 
            scale=(1.5, 1.5, 1.4),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.12, -0.32),
            rot=(1.0, 0.0, 0.0, 0.0),   # wxyz
        ),
    )
    
    converter = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/converter",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.8, -0.7, -1.0], rot=[0, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', f"ConveyorBelt_A06.usd"),
            scale=(2.0, 0.6, 0.56)
        ),
    )
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    
    hand_cam =  CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    
    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    
def rgbd_to_world_pcd(rgb, depth, K, T_cam2world):
    """
    rgb: (H, W, 3) uint8
    depth: (H, W) float32 [meter]
    K: (3, 3) intrinsic
    T_cam2world: (4, 4) cam -> world
    return: o3d.geometry.PointCloud in the world frame
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.reshape(-1)
    valid = z > 0
    u, v, z = u.reshape(-1)[valid], v.reshape(-1)[valid], z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)
    pts_world = (T_cam2world[:3, :3] @ pts_cam.T).T + T_cam2world[:3, 3]
    colors = rgb.reshape(-1, 3)[valid] / 255.0
    
    # Add noise
    noise_ratio = 1.0
    random_colors = np.random.rand(*colors.shape)
    noisy_colors = (1 - noise_ratio) * colors + noise_ratio * random_colors
    colors = np.clip(noisy_colors, 0, 1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def run_simulator(sim, scene, envi_params, topic_processor):
    robot = scene["robot"] 
    sim_dt = sim.get_physics_dt()
    
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    gripper_dof_idx = [robot.joint_names.index(n) for n in ["panda_finger_joint1", "panda_finger_joint2"]]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", 
        ik_params={
            'lambda_val': 0.1, 'joint_velocity_limit': 0.05,
            'position_tolerance': 1e-4,
            'orientation_tolerance': 1e-3,
        }
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    # Initialize the robot joint
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos[:, :7], joint_ids=robot_entity_cfg.joint_ids)
    robot.set_joint_position_target(joint_pos[:, 7:], joint_ids=gripper_dof_idx)
    scene.update(sim_dt)
    last_publish_time = time.time()
    while simulation_app.is_running():
        target_pos = torch.tensor([0.3222, 0.0017, 0.5964], device=sim.device)
        target_quat = torch.tensor([-0.0013,  0.9732, -0.0029,  0.2301], device=sim.device)
        ik_cmd = torch.cat([target_pos, target_quat]).expand(scene.num_envs, 7)
        gripper_cmd = torch.full((scene.num_envs, 2), 0.04, device=sim.device) 
        # Compute IK
        diff_ik_controller.set_command(ik_cmd)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Update robot hand control target
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_cmd, joint_ids=gripper_dof_idx)
        # Get environment information
        hand_cam_rgb = scene['hand_cam'].data.output["rgb"]   # Left shape: (num_envs, 480, 640, 3)
        hand_cam_depth = scene['hand_cam'].data.output["distance_to_image_plane"] # Left shape: (num_envs, 480, 640, 1)
        external_cam_rgb = scene['external_cam'].data.output["rgb"]   # Left shape: (num_envs, 480, 640, 3)
        external_cam_depth = scene['external_cam'].data.output["distance_to_image_plane"] # Left shape: (num_envs, 480, 640, 1)
        hand_cam_rgb = hand_cam_rgb[0].cpu().numpy()    # Left shape: (480, 640, 3)
        hand_cam_depth = hand_cam_depth[0, :, :, 0].cpu().numpy()   # Left shape: (480, 640)
        external_cam_rgb = external_cam_rgb[0].cpu().numpy()    # Left shape: (480, 640, 3)
        external_cam_depth = external_cam_depth[0, :, :, 0].cpu().numpy()   # Left shape: (480, 640)
        # External camera pose
        external_cam_world_pos = envi_params['external_cam_world_pos']
        external_cam_world_rot = envi_params['external_cam_world_rot']  # wxyz
        T_externalcam2world = np.eye(4)
        T_externalcam2world[:3, 3] = external_cam_world_pos
        T_externalcam2world[:3, :3] = R.from_quat(np.roll(external_cam_world_rot, -1)).as_matrix()   # The input of R.from_quat must be xyzw
        external_cam_instrinsic = scene['external_cam'].data.intrinsic_matrices[0].cpu().numpy()
        # Hand camera pose
        hand_pos, hand_quat = scene['robot'].data.body_pos_w[0, ee_jacobi_idx].cpu().numpy(), robot.data.body_quat_w[0, ee_jacobi_idx].cpu().numpy()    # quat: wxyz
        T_hand2world = np.eye(4)
        T_hand2world[:3, 3] = hand_pos
        T_hand2world[:3, :3] = R.from_quat(np.roll(hand_quat, -1)).as_matrix()   # The input of R.from_quat must be xyzw
        T_cam2hand = np.eye(4)
        T_cam2hand[:3, 3] = envi_params['hand_cam_relative_pos']
        hand_cam_relative_rot_xyzw = np.roll(envi_params['hand_cam_relative_rot'], -1)  # wxyz -> xyzw
        T_cam2hand[:3, :3] = R.from_quat(hand_cam_relative_rot_xyzw).as_matrix()
        T_handcam2world = T_hand2world @ T_cam2hand
        hand_cam_world_pos = T_handcam2world[:3, 3]
        hand_cam_world_quat = np.roll(R.from_matrix(T_handcam2world[:3, :3]).as_quat(), 1)  # wxyz
        hand_cam_instrinsic = scene['hand_cam'].data.intrinsic_matrices[0].cpu().numpy()

        # Visualize point cloud.
        if time.time() - last_publish_time > 2.0:
            pcd_external2world  = rgbd_to_world_pcd(external_cam_rgb, external_cam_depth, external_cam_instrinsic, T_externalcam2world)
            pcd_hand2world = rgbd_to_world_pcd(hand_cam_rgb, hand_cam_depth, hand_cam_instrinsic, T_handcam2world)
            #pcd_merged = merge_pc(pcd_external2world, pcd_hand2world)       
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_external2world)
            opt = vis.get_render_option()
            opt.point_size = 3
            params = o3d.io.read_pinhole_camera_parameters("o3d_viewpoint.json")
            view_control = vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(params)
            vis.run()
            #view_control = vis.get_view_control()
            #params = view_control.convert_to_pinhole_camera_parameters()
            #o3d.io.write_pinhole_camera_parameters("o3d_viewpoint.json", params)
            vis.destroy_window() 
        # Publish ros2 topic
        '''if time.time() - last_publish_time > envi_params['topic_publish_min_time']:
            topic_processor.publish_rgb_image(external_cam_rgb, '/external_camera/image_raw')
            topic_processor.publish_depth_image(external_cam_depth, '/external_camera/depth_raw')
            topic_processor.publish_pose(external_cam_world_pos, external_cam_world_rot, '/external_camera/pose')
            topic_processor.publish_rgb_image(hand_cam_rgb, '/hand_camera/image_raw')
            topic_processor.publish_depth_image(hand_cam_depth, '/hand_camera/depth_raw')
            last_publish_time = time.time()'''
        
        # Update simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
def make_envi(envi_params):
    scene_cfg = ManipulationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg.robot.init_state.pos = envi_params["robot_default_pos"]
    scene_cfg.robot.init_state.rot = envi_params["robot_default_rot"]
    scene_cfg.robot.init_state.joint_pos = envi_params["robot_default_joint_dict"]
    scene_cfg.hand_cam.offset = CameraCfg.OffsetCfg(pos=envi_params["hand_cam_relative_pos"], rot=envi_params["hand_cam_relative_rot"], convention="isaac") # rot: wxyz
    scene_cfg.external_cam.offset = CameraCfg.OffsetCfg(pos=envi_params["external_cam_world_pos"], rot=envi_params["external_cam_world_rot"], convention="isaac")
    
    asset_folder_list = sorted(os.listdir(envi_params['obj_asset_path']))
    asset_folder_list = [folder for folder in asset_folder_list if not folder.startswith('.')]
    asset_folder_list = [asset_folder_list[i] for i in [1, 0, 2, 3, 4]]
    for i in range(envi_params['obj_num']):
        obj_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obj_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(envi_params['obj_asset_path'], asset_folder_list[i], 'model.usd'), 
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                #pos=(0.6 * (i) + 0.1, -0.7 + random.uniform(-0.1, 0.2), 0.25),
                pos=(0.6 * (i) + 0.1, -0.57, 0.2),
                rot=(1.0, 0.0, 0.0, 0.0),   # wxyz
            ),
        )
        setattr(scene_cfg, f"obj_{i}", obj_cfg)
    return scene_cfg

def main(envi_params, topic_processor):
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = make_envi(envi_params)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, envi_params, topic_processor)


if __name__ == "__main__":
    set_seed()
    publish_topic_list = [
        ('/external_camera/image_raw', 'sensor_msgs/msg/Image'),
        ('/external_camera/depth_raw', 'sensor_msgs/msg/Image'),
        ('/external_camera/pose', 'geometry_msgs/PoseStamped'),
        ('/hand_camera/image_raw', 'sensor_msgs/msg/Image'),
        ('/hand_camera/depth_raw', 'sensor_msgs/msg/Image'),
        ('/hand_camera/pose', 'geometry_msgs/PoseStamped'),
    ]
    subscribe_topic_list = [
        ('/cmd_vel', 'geometry_msgs/msg/Twist'),
    ]
    topic_processor = TopicProcessor(subscribe_topic_list, publish_topic_list)
    envi_params = dict(
        robot_default_pos = (0.0, -0.155, -0.31),
        robot_default_rot = (0.7071, 0.0, 0.0, -0.7071),
        robot_default_joint_dict = {'panda_joint1': -0.0756, 'panda_joint2': -0.8898, 'panda_joint3': 0.0561, 'panda_joint4': -2.5263, 'panda_joint5': 0.0675, 'panda_joint6': 2.1010, \
            'panda_joint7': 0.741, 'panda_finger_joint.*': 0.04},
        external_cam_world_pos = (0.0, -1.2, 0.48), 
        external_cam_world_rot=(0.9239, 0.3827, 0.0, 0.0),  # wxyz
        hand_cam_relative_pos = (0.12, -0.02, 0.0), 
        hand_cam_relative_rot=(0.252, 0.697, 0.647, 0.177), # wxyz
        obj_asset_path = '/home/cvte/twilight/Isaac_Sim/train_assets',
        obj_num = 5,
        topic_publish_min_time = 0.5,
    )
    
    main(envi_params, topic_processor)
    simulation_app.close()
