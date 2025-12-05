# Override the built-in print function with a timestamp version
import builtins
import json
import os
import pdb
from datetime import datetime

import os
import time
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from tqdm import tqdm

class NavDP_Base_Datset(Dataset):
    def __init__(
        self,
        cfg,
        root_dirs,
        preload_path=False,
        preload=False,
        memory_size=8,
        predict_size=24,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
    ):
        self.cfg = cfg
        self.dataset_dirs = np.array([p for p in os.listdir(root_dirs) if os.path.isdir(os.path.join(root_dirs, p))])
        self.memory_size = memory_size
        self.scene_scale_size = scene_data_scale
        self.trajectory_data_scale = trajectory_data_scale
        self.predict_size = predict_size
        self.trajectory_dirs = []
        self.trajectory_data_dir = []
        self.trajectory_rgb_path = []
        self.trajectory_depth_path = []
        self.trajectory_afford_path = []
        self.item_cnt = 0
        self.batch_time_sum = 0.0
        self._last_time = None

        if preload is False:
            for group_dir in self.dataset_dirs:  # gibson_zed, 3dfront ...
                all_scene_dirs = np.array([p for p in os.listdir(os.path.join(root_dirs, group_dir))])
                select_scene_dirs = all_scene_dirs[
                    np.arange(0, all_scene_dirs.shape[0], 1 / self.scene_scale_size).astype(np.int32)
                ]
                for scene_dir in select_scene_dirs:
                    all_traj_dirs = np.array([p for p in os.listdir(os.path.join(root_dirs, group_dir, scene_dir))])
                    select_traj_dirs = all_traj_dirs[
                        np.arange(0, all_traj_dirs.shape[0], 1 / self.trajectory_data_scale).astype(np.int32)
                    ]
                    for traj_dir in tqdm(select_traj_dirs):
                        entire_task_dir = os.path.join(root_dirs, group_dir, scene_dir, traj_dir)
                        rgb_dir = os.path.join(entire_task_dir, "videos/chunk-000/observation.images.rgb/")
                        depth_dir = os.path.join(entire_task_dir, "videos/chunk-000/observation.images.depth/")
                        data_path = os.path.join(
                            entire_task_dir, 'data/chunk-000/episode_000000.parquet'
                        )  # intrinsic, extrinsic, cam_traj, path
                        afford_path = os.path.join(entire_task_dir, 'data/chunk-000/path.ply')

                        if (not os.path.exists(rgb_dir)) or (not os.path.exists(depth_dir)) or (not os.path.exists(data_path)) or (not os.path.exists(afford_path)):
                            continue

                        rgbs_length = len([p for p in os.listdir(rgb_dir)])
                        depths_length = len([p for p in os.listdir(depth_dir)])

                        rgbs_path = []
                        depths_path = []
                        if depths_length != rgbs_length:
                            continue
                        for i in range(rgbs_length):
                            rgbs_path.append(os.path.join(rgb_dir, "%d.jpg" % i))
                            depths_path.append(os.path.join(depth_dir, "%d.png" % i))
                        if os.path.exists(data_path) is False:
                            continue
                        self.trajectory_dirs.append(entire_task_dir)
                        self.trajectory_data_dir.append(data_path)
                        self.trajectory_rgb_path.append(rgbs_path)
                        self.trajectory_depth_path.append(depths_path)
                        self.trajectory_afford_path.append(afford_path)

            save_dict = {
                'trajectory_dirs': self.trajectory_dirs,
                'trajectory_data_dir': self.trajectory_data_dir,
                'trajectory_rgb_path': self.trajectory_rgb_path,
                'trajectory_depth_path': self.trajectory_depth_path,
                'trajectory_afford_path': self.trajectory_afford_path,
            }
            with open(preload_path, 'w') as f:
                json.dump(save_dict, f, indent=4)
        else:
            load_dict = json.load(open(preload_path, 'r'))
            self.trajectory_dirs = load_dict['trajectory_dirs']
            self.trajectory_data_dir = load_dict['trajectory_data_dir']
            self.trajectory_rgb_path = load_dict['trajectory_rgb_path']
            self.trajectory_depth_path = load_dict['trajectory_depth_path']
            self.trajectory_afford_path = load_dict['trajectory_afford_path']

    def __len__(self):
        return len(self.trajectory_dirs)

    def load_image(self, image_url):
        image = Image.open(image_url)
        image = np.array(image, np.uint8)
        return image

    def load_depth(self, depth_url):
        depth = Image.open(depth_url)
        depth = np.array(depth, np.uint16)
        return depth

    def load_pointcloud(self, pcd_url):
        pcd = o3d.io.read_point_cloud(pcd_url)
        return pcd

    def process_image(self, image_path):
        image = self.load_image(image_path)
        ori_height, ori_width, _ = image.shape
        resize_img = cv2.resize(image, self.cfg['DATA']['IMG_RESIZE_SHAPE'])
        prop = (self.cfg['DATA']['IMG_RESIZE_SHAPE'][0] / ori_width, self.cfg['DATA']['IMG_RESIZE_SHAPE'][1] / ori_height)
        return resize_img, prop

    def process_depth(self, depth_path):
        depth = self.load_depth(depth_path) / 10000.0
        ori_height, ori_width = depth.shape
        resize_depth = cv2.resize(depth, self.cfg['DATA']['IMG_RESIZE_SHAPE']).astype(np.float32)
        return resize_depth[:, :, np.newaxis]

    def process_data_parquet(self, index):
        if not os.path.isfile(self.trajectory_data_dir[index]):
            raise FileNotFoundError(self.trajectory_data_dir[index])
        df = pd.read_parquet(self.trajectory_data_dir[index])
        camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
        camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
        trajectory_length = len(df['action'].tolist())
        camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64)
        return camera_intrinsic, camera_extrinsic, camera_trajectory, trajectory_length
    
    def load_obs(self, rgb_paths, depth_paths, start_step):
        context_image, resize_ratio = self.process_image(rgb_paths[start_step])        # (H, W, 3)
        context_depth = self.process_depth(depth_paths[start_step])      # (H, W, 1)
        return context_image, context_depth, resize_ratio

    def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic):
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_world.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(np.linalg.inv(homo_RT), np.array([*T_world, 1]).T)[0:3]
            T_frame = np.array([T_frame[1], -T_frame[0], T_frame[2]])  # [:T[1],-T[0],T[2]
            return R_frame, T_frame
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(
                np.linalg.inv(homo_RT), np.concatenate((T_world, np.ones((T_world.shape[0], 1))), axis=-1).T
            ).T[:, 0:3]
            T_frame = T_frame[:, [1, 0, 2]]
            T_frame[:, 1] = -T_frame[:, 1]
            return R_frame, T_frame

    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic):
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_frame.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(homo_RT, np.array([-T_frame[1], T_frame[0], T_frame[2], 1]).T)[0:3]
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(
                homo_RT,
                np.concatenate(
                    (np.stack((-T_frame[:, 1], T_frame[:, 0], T_frame[:, 2]), axis=-1), np.ones((T_frame.shape[0], 1))),
                    axis=-1,
                ).T,
            ).T[:, 0:3]
        return R_world, T_world

    def xyz_to_xyt(self, xyz_actions, init_vector):
        xyt_actions = []
        for i in range(0, xyz_actions.shape[0] - 1):
            current_vector = xyz_actions[i + 1] - xyz_actions[i]
            dot_product = np.dot(init_vector[0:2], current_vector[0:2])
            cross_product = np.cross(init_vector[0:2], current_vector[0:2])
            theta = np.arctan2(cross_product, dot_product)
            xyt_actions.append([xyz_actions[i][0], xyz_actions[i][1], theta])
        return np.array(xyt_actions)

    def process_actions(self, extrinsics, base_extrinsic, start_step, end_step, pred_digit=1):
        label_linear_pos = []
        for f_ext in extrinsics[start_step : end_step + 1]:
            R, T = self.relative_pose(
                extrinsics[start_step][0:3, 0:3],
                extrinsics[start_step][0:3, 3],
                f_ext[0:3, 0:3],
                f_ext[0:3, 3],
                base_extrinsic,
            )
            label_linear_pos.append(T)
        label_actions = np.array(label_linear_pos)

        origin_world_points = extrinsics[start_step : end_step + 1, 0:3, 3]

        local_label_points = []
        for f_ext in origin_world_points:
            Rf, Tf = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3], np.eye(3), f_ext, base_extrinsic
            )
            local_label_points.append(Tf)
        local_label_points = np.array(local_label_points)
        action_indexes = np.clip(np.arange(self.predict_size + 1) * pred_digit, 0, label_actions.shape[0] - 2)
        action_is_invalid = np.arange(self.predict_size + 1) * pred_digit > label_actions.shape[0] - 2
        return local_label_points, origin_world_points, action_indexes, action_is_invalid

    def __getitem__(self, index):
        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
        ) = self.process_data_parquet(index)

        # Get the actual RGB path length to ensure indices don't exceed it
        rgb_path_length = len(self.trajectory_rgb_path[index])
        # Use the minimum of trajectory_length and rgb_path_length to ensure indices are valid
        effective_trajectory_length = min(trajectory_length, rgb_path_length)
        
        # Ensure effective_trajectory_length is at least 2 for valid sampling
        if effective_trajectory_length < 2:
            raise ValueError(f"Trajectory too short: effective_length={effective_trajectory_length}, rgb_path_length={rgb_path_length}, trajectory_length={trajectory_length}")
        
        pred_digit = 4
        start_frame_idx = np.random.randint(0, effective_trajectory_length - 2 - pred_digit)
        end_frame_idx = np.random.randint(start_frame_idx + 1 + pred_digit, effective_trajectory_length - 1)

        rgb_image, depth_image, resize_ratio = self.load_obs(self.trajectory_rgb_path[index], self.trajectory_depth_path[index], start_frame_idx,)   # memory_images shape: (h, w, 3), depth_image shape: (h, w, 1)
        camera_intrinsic[0] = camera_intrinsic[0] * resize_ratio[0]
        camera_intrinsic[1] = camera_intrinsic[1] * resize_ratio[1]
        
        target_local_points, target_world_points, action_indexes, action_is_invalid = self.process_actions(trajectory_extrinsics, trajectory_base_extrinsic, start_frame_idx, \
            end_frame_idx, pred_digit=pred_digit)
        # convert the xyz points into xy-theta points
        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        # based on the prediction length to decide the final prediction trajectories
        pred_actions = target_xyt_actions[action_indexes]
        point_goal = target_xyt_actions[-1]

        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        action_is_invalid = action_is_invalid[1:]

        # Summarize avg time of batch
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        
        point_goal = torch.tensor(point_goal, dtype=torch.float32)
        rgb_image = torch.tensor(rgb_image, dtype=torch.float32)
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        camera_intrinsic = torch.tensor(camera_intrinsic, dtype=torch.float32)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        action_is_invalid = torch.tensor(action_is_invalid, dtype=torch.bool)

        return point_goal, rgb_image, depth_image, camera_intrinsic, pred_actions, action_is_invalid


def navdp_collate_fn(batch):
    collated = {
        "batch_pg": torch.stack([item[0] for item in batch]),
        "batch_rgb": torch.stack([item[1] for item in batch]),
        "batch_depth": torch.stack([item[2] for item in batch]),
        "batch_intrinsic": torch.stack([item[3] for item in batch]),
        "batch_labels": torch.stack([item[4] for item in batch]),
        "batch_labels_invalid": torch.stack([item[5] for item in batch]),
    }
    return collated


if __name__ == "__main__":
    os.makedirs("./navdp_dataset_test/", exist_ok=True)
    dataset = NavDP_Base_Datset(
        "/shared/smartbot_new/liuyu/vln-n1-minival/",
        "./navdp_dataset_test/dataset_lerobot.json",
        8,
        24,
        224,
        trajectory_data_scale=0.1,
        scene_data_scale=0.1,
        preload=False,
    )

    for i in range(10):
        (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_image,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            pixel_flag,
        ) = dataset.__getitem__(i)
        if pixel_flag == 1.0:
            pixel_obs = pixel_goal.numpy()[:, :, 0:3] * 255
            pixel_obs[pixel_goal[:, :, 3] == 1] = np.array([0, 0, 255])

            draw_current_image = cv2.cvtColor(image_goal[:, :, 3:6].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_current_image = cv2.putText(
                draw_current_image, "Current-Image", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_goal_image = cv2.cvtColor(image_goal[:, :, 0:3].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_goal_image = cv2.putText(
                draw_goal_image, "Image-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_pixel_image = cv2.cvtColor(pixel_obs.copy(), cv2.COLOR_BGR2RGB)
            draw_pixel_image = cv2.putText(
                draw_pixel_image, "Pixel-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            goal_info_image = np.concatenate((draw_current_image, draw_goal_image, draw_pixel_image), axis=1)
            goal_info_image = cv2.putText(
                goal_info_image,
                "PointGoal=[{:.3f}, {:.3f}, {:.3f}]".format(*point_goal),
                (190, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
            )
            cv2.imwrite("./navdp_dataset_test/goal_information_%d.png" % i, goal_info_image)
