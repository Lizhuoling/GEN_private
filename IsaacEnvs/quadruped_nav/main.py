# Run "source /home/cvte/twilight/environment/Isaac_Sim_5.0/setup_conda_env.sh" before running this script.
# Run ``python main.py --enable_cameras" to start this simulation environment.
# To import 3D GS based scene, must use IsaacLab v2.2.0.

# Initialize as a Isaac Sim launch file.
import argparse      
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Quadrupedal robot navigation environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

import cli_args
cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import os
import pdb
import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
import time
import roslibpy
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R
import torch
import time
from tensordict import TensorDict
sys.path.append('/home/cvte/twilight/code/IsaacLab/source/isaaclab')

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
import gymnasium as gym

from aliengo_asset import ALIENGO_CFG

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
                'frame_id': 'front_cam_link'
            },
            'height': height,
            'width': width,
            'encoding': 'rgb8',
            'is_bigendian': 0,
            'step': width * 3,
            'data': image_data
        })
        self.publisher_dict[topic_name].publish(msg)
        
    def publish_depth_image(self, depth_image, topic_name):
        if depth_image.dtype != np.float32:
            raise ValueError(f"The depth image must be of type float32, but got {depth_image.dtype}")
        height, width = depth_image.shape
        image_data = base64.b64encode(bytes(depth_image.astype('<f4').tobytes())).decode('utf-8')
        msg = roslibpy.Message({
            'header': {
                'stamp': roslibpy.Time.now(),
                'frame_id': 'front_cam_link'
            },
            'height': height,
            'width': width,
            'encoding': '32FC1',
            'is_bigendian': 0,
            'step': width * 4,
            'data': image_data
        })
        self.publisher_dict[topic_name].publish(msg)
        
    def publish_odom(self, robot_data, topic_name):
        initial_root_state  = robot_data.default_root_state.clone()
        current_root_state = robot_data.root_state_w.clone()
        pos_init = initial_root_state[0, :3].cpu().numpy()
        quat_init = initial_root_state[0, 3:7].cpu().numpy()  # [w, x, y, z]

        pos_curr = current_root_state[0, :3].cpu().numpy()  # [w, x, y, z]
        quat_curr = current_root_state[0, 3:7].cpu().numpy()    # [w, x, y, z]

        q_init_inv = R.from_quat(quat_init[[1, 2, 3, 0]]).inv()  # scipy uses [x,y,z,w]
        q_curr = R.from_quat(quat_curr[[1, 2, 3, 0]])
        q_rel = q_curr * q_init_inv
        delta_quat = q_rel.as_quat()
        
        delta_pos_world = pos_curr - pos_init
        delta_odom = q_init_inv.apply(delta_pos_world)
        
        lin_vel_world = current_root_state[0, 7:10].cpu().numpy()
        ang_vel_world = current_root_state[0, 10:13].cpu().numpy()
        R_w2b = q_curr.inv() # R_w2b: Rotation from world to robot base_link
        lin_vel_base = R_w2b.apply(lin_vel_world)
        ang_vel_base = R_w2b.apply(ang_vel_world)
        
        msg = {
            'header': {
                'stamp': roslibpy.Time.now(),
                'frame_id': 'odom'
            },
            'child_frame_id': 'base_link',
            'pose': {
                'pose': {
                    'position': {
                        'x': float(delta_odom[0]),
                        'y': float(delta_odom[1]),
                        'z': float(delta_odom[2])
                    },
                    'orientation': {
                        'x': float(delta_quat[0]),
                        'y': float(delta_quat[1]),
                        'z': float(delta_quat[2]),
                        'w': float(delta_quat[3])
                    }
                }
            },
            'twist': {
                'twist': {
                    'linear': {
                        'x': float(lin_vel_base[0]),
                        'y': float(lin_vel_base[1]),
                        'z': float(lin_vel_base[2])
                    },
                    'angular': {
                        'x': float(ang_vel_base[0]),
                        'y': float(ang_vel_base[1]),
                        'z': float(ang_vel_base[2])
                    }
                }
            }
        }
        self.publisher_dict[topic_name].publish(msg)
        
    def publish_tf(self, robot_data):
        initial_root_state  = robot_data.default_root_state.clone()
        current_root_state = robot_data.root_state_w.clone()
        pos_init = initial_root_state[0, :3].cpu().numpy()
        quat_init = initial_root_state[0, 3:7].cpu().numpy()  # [w, x, y, z]

        pos_curr = current_root_state[0, :3].cpu().numpy()
        quat_curr = current_root_state[0, 3:7].cpu().numpy()
        
        now = roslibpy.Time.now()
        transforms = []

        world2odom_tf = {
            'header': {
                'stamp': now,
                'frame_id': 'world'
            },
            'child_frame_id': 'odom',
            'transform': {
                'translation': {'x': float(pos_init[0]),
                                'y': float(pos_init[1]),
                                'z': float(pos_init[2])},
                'rotation':    {'x': float(quat_init[1]),
                                'y': float(quat_init[2]),
                                'z': float(quat_init[3]),
                                'w': float(quat_init[0])}
            }
        }
        transforms.append(world2odom_tf)
        
        world2base_tf = {
            'header': {
                'stamp': now,
                'frame_id': 'world'
            },
            'child_frame_id': 'base_link',
            'transform': {
                'translation': {'x': float(pos_curr[0]),
                                'y': float(pos_curr[1]),
                                'z': float(pos_curr[2])},
                'rotation':    {'x': float(quat_curr[1]),
                                'y': float(quat_curr[2]),
                                'z': float(quat_curr[3]),
                                'w': float(quat_curr[0])}
            }
        }
        transforms.append(world2base_tf)
        
        tf_message = roslibpy.Message({'transforms': transforms})
        self.publisher_dict['/tf'].publish(tf_message)

OBS_HISTORY_LENGTH = 5
JOINTS_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[0.5, 0.0, 0.2]], device=env.device).repeat(env.num_envs, 1)

def ros2_commands(env: ManagerBasedEnv) -> torch.Tensor:
    global topic_processor
    cmd_vel = topic_processor.get_message('/cmd_vel')
    if cmd_vel is None:
        cmd_vel_tensor = torch.tensor([0.0, 0.0, 0.0], device=env.device)[None,].repeat(env.num_envs, 1)
    else:
        cmd_vel_x, cmd_vel_y, cmd_angular_z = cmd_vel['linear']['x'], cmd_vel['linear']['y'], cmd_vel['angular']['z']
        cmd_vel_tensor = torch.tensor([cmd_vel_x, cmd_vel_y, cmd_angular_z], device=env.device)[None,].repeat(env.num_envs, 1)
    print(f'cmd velocity: {cmd_vel_tensor}')
    return cmd_vel_tensor


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    ground_cfg = AssetBaseCfg(
            prim_path="/World/ground", 
            spawn=sim_utils.UsdFileCfg(usd_path=f"/home/cvte/twilight/data/IsaacSim/CVTE2_scene/carter_warehouse.usd"))
    
    #ground_cfg = AssetBaseCfg(
    #    prim_path="/World/ground", 
    #    spawn=sim_utils.UsdFileCfg(usd_path=f"/home/cvte/twilight/data/IsaacSim/CVTE2_scene/cvte2_mesh_3dgs.usd"))

    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    robot_front_cam =  CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=JOINTS_ORDER, 
                                           scale=0.5, 
                                           use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        base_vel = ObsTerm(func=mdp.base_lin_vel, # 3
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, # 3
            history_length=OBS_HISTORY_LENGTH, 
            flatten_history_dim=False)
        
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity,    # 3
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False)
        
        velocity_commands = ObsTerm(func=ros2_commands, # 3, xy linear velocity command and yaw angular velocity command
            history_length=OBS_HISTORY_LENGTH, 
            flatten_history_dim=False)
        
        joints_pos_delta = ObsTerm(func=mdp.joint_pos_rel,  # 12
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False,
            params={
            "asset_cfg": SceneEntityCfg("robot", joint_ids=JOINT_IDS),
            })
        
        joints_vel = ObsTerm(   # 12
            func=mdp.joint_vel,
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False,
            params={
            "asset_cfg": SceneEntityCfg("robot", joint_ids=JOINT_IDS),
            }
        )
        
        actions = ObsTerm(func=mdp.last_action, # 12
                          history_length=OBS_HISTORY_LENGTH, 
                          flatten_history_dim=False)
        
        # We do not collect clock_data signal here
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0
    use_clock_signal = True
    if(use_clock_signal):
        observation_space += 4
    # observation history
    use_observation_history = True
    history_length = 5
    if(use_observation_history):
        single_observation_space = observation_space # Placeholder. Later we may add map, but only from the latest obs
        observation_space *= history_length

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    events: EventCfg = EventCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
        
class PhaseGenerator():
    def __init__(self,):
        desired_gait = "trot"  # trot, crawl, pace
        if(desired_gait == "trot"):
            self.step_freq = 1.4
            self.duty_factor = 0.65
            self.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])
            self._velocity_gait_multiplier = 1.0
        elif(desired_gait == "crawl"):
            self.step_freq = 0.5
            self.duty_factor = 0.8
            self.phase_offset = np.array([0.0, 0.5, 0.75, 0.25])
            self.velocity_gait_multiplier = 0.5
        elif(desired_gait == "pace"):
            self.step_freq = 1.4
            self.duty_factor = 0.7
            self.phase_offset = np.array([0.8, 0.3, 0.8, 0.3])
            self.velocity_gait_multiplier = 1.0
        self.phase_signal = self.phase_offset
            
        self.RL_FREQ = 50
            
    def add_phase(self, command, obs):
        self.phase_signal += self.step_freq * (1 / (self.RL_FREQ))
        self.phase_signal = self.phase_signal % 1.0
        phase_signal = self.phase_signal.reshape(1, 1, 4).repeat(obs.shape[1], axis = 1)
        phase_signal = torch.Tensor(phase_signal).to(obs.device)
        obs = torch.cat((obs, phase_signal), axis = -1)
        zero_command_mask = torch.norm(command, dim = -1) < 0.01
        obs[zero_command_mask][:, -4:] = -1.0
        obs[48:52] = -1.0
        return obs
        
def construct_policy():
    policy_path = "aliengo_asset/aliengo_policy.pt"
    actor_critic = ActorCritic(num_actor_obs = 260, num_critic_obs = 260, num_actions = 12, actor_hidden_dims = [128, 128, 128], critic_hidden_dims = [128, 128, 128]).cuda()
    loaded_dict = torch.load(policy_path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict = True)
    return actor_critic

def main(topic_processor, topic_publish_min_time = 0.1):
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    
    policy = construct_policy()
    phase_generator = PhaseGenerator()

    # Reset environment
    print("[INFO]: Resetting environment...")
    reset_start = time.time()
    obs, _ = env.reset()
    reset_time = time.time() - reset_start
    print(f"[INFO]: Environment reset took {reset_time:.3f} seconds")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    last_publish_time = time.time()
    
    try:
        while True:
            with torch.inference_mode():
                policy_obs = obs["policy"]  # Left shape: (num_envs, time_len, attribute_len - 4), attribute_len should be 52
                vel_command = policy_obs[:, :, 9:12]
                policy_obs = phase_generator.add_phase(vel_command, policy_obs) # Left shape: (num_envs, time_len, attribute_len)
                policy_obs = policy_obs.reshape(policy_obs.shape[0], -1).contiguous()
                # infer action
                action = policy.actor(policy_obs)[:, :12]
                # step env
                obs, _ = env.step(action)
                
            img_rgb = env.scene['robot_front_cam'].data.output["rgb"]   # Left shape: (num_envs, 480, 640, 3)
            img_depth = env.scene['robot_front_cam'].data.output["distance_to_image_plane"] # Left shape: (num_envs, 480, 640, 1)
            rgb = img_rgb[0].cpu().numpy()
            depth = img_depth[0, :, :, 0].cpu().numpy()
            
            if time.time() - last_publish_time > topic_publish_min_time:
                topic_processor.publish_rgb_image(rgb, '/front_stereo_camera/left/image_raw')
                topic_processor.publish_depth_image(depth, '/front_stereo_camera/left/depth_raw')
                topic_processor.publish_odom(env.scene.articulations['robot'].data, '/chassis/odom')
                topic_processor.publish_tf(env.scene.articulations['robot'].data)
                last_publish_time = time.time()
            
            '''cv2.imwrite('vis.png', rgb[:, :, ::-1])
            plt.imshow(img_depth[0, :, :, 0].cpu().numpy(), cmap='Spectral_r')
            plt.axis('off')
            plt.savefig('front_cam_depth.png', bbox_inches='tight', pad_inches=0)'''
                
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt, exiting")
    finally:
        env.close()

if __name__ == "__main__":
    publish_topic_list = [
        ('/chassis/odom', 'nav_msgs/msg/Odometry'),
        ('/front_stereo_camera/left/image_raw', 'sensor_msgs/msg/Image'),
        ('/front_stereo_camera/left/depth_raw', 'sensor_msgs/msg/Image'),
    ]
    subscribe_topic_list = [
        ('/cmd_vel', 'geometry_msgs/msg/Twist'),
    ]
    topic_processor = TopicProcessor(subscribe_topic_list, publish_topic_list)
    
    # run the main function
    main(topic_processor = topic_processor)
    # close sim app
    simulation_app.close()
        

        
