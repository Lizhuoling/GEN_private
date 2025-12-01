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
        
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


@configclass
class AssetSceneCfg(InteractiveSceneCfg):
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
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

def run_simulator(sim, scene,):
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
def make_envi(asset_root):
    scene_cfg = AssetSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    
    asset_folder_list = sorted(os.listdir(asset_root))
    asset_folder_list = [folder for folder in asset_folder_list if not folder.startswith('.')]
    row_num = 5

    for i, asset_folder_name in enumerate(asset_folder_list):
        row_id = i // row_num
        col_id = i % row_num
        obj_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obj_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(asset_root, asset_folder_name, 'model.usd'), 
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(col_id * 0.4, row_id * 0.4, 0.1),
                rot=(1.0, 0.0, 0.0, 0.0),   # wxyz
            ),
        )
        setattr(scene_cfg, f"obj_{i}", obj_cfg)
    return scene_cfg

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = make_envi(asset_root = '/home/cvte/twilight/Isaac_Sim/val_assets')
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    set_seed()
    main()
    simulation_app.close()
