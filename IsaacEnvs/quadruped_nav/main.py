# Run "source /home/cvte/twilight/environment/Isaac_Sim_5.0/setup_conda_env.sh" before running this script.

# Initialize as a Isaac Sim launch file.
import argparse      
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment with LiDAR.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--benchmark_steps", type=int, default=1000, help="Number of steps to run for benchmarking.")

import cli_args
cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import sys
import os
import pdb
import warnings
import logging
import time
import roslibpy
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor

import torch
import time
import sys
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
import gymnasium as gym

from aliengo_asset import ALIENGO_CFG

class TopicSubscriber():
    def __init__(self, topic_list, callback_timeout=0.1, callback_time_min=0.02, max_workers=None):
        self.topic_list = topic_list
        self.callback_timeout = callback_timeout
        self.callback_time_min = callback_time_min
        self.logger = logging.getLogger("Isaac")
        
        self.message_data = {}
        self.last_callback_time = {}
        self.last_execution_time = {}
        self.timeout_flag = False
        self.data_lock = Lock()
        self.ros_connected = False
        
        # Use MessagePack for more efficient serialization
        self.ros_client = roslibpy.Ros(host='localhost', port=9090,)
        
        # Thread pool for parallel callback processing
        self.max_workers = max_workers or len(topic_list)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.ros_thread = Thread(target=self._ros_loop, daemon=True)
        self.ros_thread.start()
        self._wait_for_connection(10.0)
        
        self.action_publisher = roslibpy.Topic(
            self.ros_client, 
            '/cmd_vel_policy', 
            'geometry_msgs/Twist',
            queue_size=10  # Add queue size for publisher
        )

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
                # Add queue_size and throttle_rate parameters
                ros_topic = roslibpy.Topic(
                    self.ros_client, 
                    topic_name, 
                    topic_type, 
                    queue_size=10,
                    throttle_rate=int(self.callback_time_min * 1000)  # Convert to ms
                )
            except TypeError:
                ros_topic = roslibpy.Topic(self.ros_client, topic_name, topic_type)
            
            ros_topic.subscribe(self._create_callback(topic_name))
            
            with self.data_lock:
                self.last_callback_time[topic_name] = time.time()
                self.last_execution_time[topic_name] = 0.0
            
            self.logger.info(f"Subscribed to {topic_name}")

    def _create_callback(self, topic_name):
        def callback(msg):
            current_time = time.time()
            
            with self.data_lock:
                self.message_data[topic_name] = msg
                self.last_callback_time[topic_name] = current_time
                last_exec = self.last_execution_time[topic_name]
                self.last_execution_time[topic_name] = current_time
            
            # Offload processing to thread pool
            self.executor.submit(
                self._process_message,
                topic_name,
                msg,
                current_time,
                last_exec
            )
        
        return callback

    def _process_message(self, topic_name, msg, current_time, last_exec):
        """
        Override this method with your actual message processing logic
        This runs in a separate thread from the thread pool
        """
        # Example processing placeholder
        try:
            # Add your message processing logic here
            processing_time = time.time() - current_time
            if processing_time > self.callback_time_min * 0.8:
                self.logger.warning(
                    f"Slow processing for {topic_name}: {processing_time:.4f}s"
                )
        except Exception as e:
            self.logger.error(f"Error processing {topic_name}: {str(e)}")

    def shutdown(self):
        """Properly clean up resources"""
        self.ros_client.terminate()
        self.executor.shutdown(wait=False)
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        self.logger.info("Subscriber shutdown complete")


OBS_HISTORY_LENGTH = 5
JOINTS_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    ground_cfg = AssetBaseCfg(
            prim_path="/World/ground", 
            spawn=sim_utils.UsdFileCfg(usd_path=f"/home/cvte/twilight/data/IsaacSim/CVTE2_scene/carter_warehouse.usd"))

    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
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
        
        base_vel = ObsTerm(func=mdp.base_lin_vel, 
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
            history_length=OBS_HISTORY_LENGTH, 
            flatten_history_dim=False)
        
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity,
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False)
        
        velocity_commands = ObsTerm(func=constant_commands, # xy linear velocity command and yaw angular velocity command
            history_length=OBS_HISTORY_LENGTH, 
            flatten_history_dim=False)
        
        joints_pos_delta = ObsTerm(func=mdp.joint_pos_rel,
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False,
            params={
            "asset_cfg": SceneEntityCfg("robot", joint_ids=JOINT_IDS),
            })
        
        joints_vel = ObsTerm(
            func=mdp.joint_vel,
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=False,
            params={
            "asset_cfg": SceneEntityCfg("robot", joint_ids=JOINT_IDS),
            }
        )
        
        actions = ObsTerm(func=mdp.last_action, 
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
        
def construct_policy():
    policy_path = "aliengo_asset/aliengo_policy.pt"
    
    # For obtaining the policy network from rsl_rl.
    policy_task_name = 'Locomotion-Aliengo-Rough-Blind' # Only for loading the policy network
    env_cfg = parse_env_cfg(policy_task_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(policy_task_name, args_cli)
    random_tensor = torch.randn(1, 260).cuda()
    random_policy_obs = TensorDict(
        {
            "policy": random_tensor,
        },
        batch_size=torch.Size([1]),
        device=None
    )
    cfg = agent_cfg.to_dict()
    cfg['policy'] = {'class_name': 'ActorCritic', 'init_noise_std': 1.0, 'noise_std_type': 'scalar', 'actor_obs_normalization': {}, 'critic_obs_normalization': {}, \
        'actor_hidden_dims': [128, 128, 128], 'critic_hidden_dims': [128, 128, 128], 'activation': 'elu'}
    cfg["obs_groups"] = {'policy': ['policy'], 'critic': ['policy']}
    policy_cfg = cfg["policy"]

    # resolve deprecated normalization config
    if cfg.get("empirical_normalization") is not None:
        warnings.warn(
            "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
            "`critic_obs_normalization` as part of the `policy` configuration instead.",
            DeprecationWarning,
        )
        if policy_cfg.get("actor_obs_normalization") is None:
            policy_cfg["actor_obs_normalization"] = cfg["empirical_normalization"]
        if policy_cfg.get("critic_obs_normalization") is None:
            policy_cfg["critic_obs_normalization"] = cfg["empirical_normalization"]

    # initialize the actor-critic
    actor_critic_class = eval(policy_cfg.pop("class_name"))
    actor_critic: ActorCritic = actor_critic_class(
        random_policy_obs, cfg["obs_groups"], 12, **policy_cfg
    ).cuda()
    
    loaded_dict = torch.load(policy_path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict = True)
    
    return actor_critic

def main():
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    
    policy = construct_policy()

    # Reset environment
    print("[INFO]: Resetting environment...")
    reset_start = time.time()
    obs, _ = env.reset()
    reset_time = time.time() - reset_start
    print(f"[INFO]: Environment reset took {reset_time:.3f} seconds")

    print(f"[INFO]: Starting benchmark for {args_cli.benchmark_steps} steps...")
    step_times = []
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    benchmark_start = time.time()

    for step in range(args_cli.benchmark_steps):
        step_start = time.time()
        
        with torch.inference_mode():
            # infer action
            action = policy(obs["policy"])[:, :12]
            # step env
            obs, _ = env.step(action)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_end = time.time()
        step_times.append(step_end - step_start)
        
        if (step + 1) % 100 == 0:
            avg_time = sum(step_times[-100:]) / 100
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"[INFO]: Step {step + 1}/{args_cli.benchmark_steps}, "
                  f"Avg time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
    
    benchmark_end = time.time()
    total_time = benchmark_end - benchmark_start
    
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
        

        
