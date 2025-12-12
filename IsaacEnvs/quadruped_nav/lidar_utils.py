import os
import pdb
import time
import cv2
import glob
import math
import json
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import isaaclab.utils.math as math_utils

"""
lidar process
"""
def project_and_sample_2d(point_array, perception_dit):
    b, n, _ = point_array.shape
    device = point_array.device
    num_bins = perception_dit['num_bins']
    angle_step = perception_dit['angle_step']
    min_angle = perception_dit['min_angle']
    max_angle = perception_dit['max_angle']

    # 1. 提取坐标并预计算距离
    x, y, z = point_array[..., 0], point_array[..., 1], point_array[..., 2]
    distance = torch.sqrt(x.square() + y.square())  # [b, n]
    
    # 2. 计算有效点掩码并获取全局索引
    valid_mask = (z > 0.1) & (z <= 0.5) & (distance >= 0.2) & (distance <= 6.0)
    batch_idx, point_idx = torch.where(valid_mask)  # [total_valid]
    total_valid = batch_idx.numel()

    # 3. 预生成结果张量
    sampled_points = torch.full((b, num_bins, 3), 0.0, dtype=torch.float32, device=device)
    distance_points = torch.full((b, num_bins), 6.0, dtype=torch.float32, device=device)

    # 4. 预计算所有角度桶的基础数据
    bin_centers = min_angle + torch.arange(num_bins, device=device) * angle_step
    cos_vals = torch.cos(bin_centers)
    sin_vals = torch.sin(bin_centers)
    sampled_points[..., 0] = 6.0 * cos_vals.unsqueeze(0)  # [b, num_bins]
    sampled_points[..., 1] = 6.0 * sin_vals.unsqueeze(0)

    # 5. 处理无有效点情况
    if total_valid == 0:
        return sampled_points, distance_points

    # 6. 提取有效点数据
    valid_x = x[batch_idx, point_idx]
    valid_y = y[batch_idx, point_idx]
    valid_dist = distance[batch_idx, point_idx]
    valid_points = point_array[batch_idx, point_idx]
    
    # 7. 角度计算与过滤
    angles = torch.atan2(valid_y, valid_x)
    angle_mask = (angles >= min_angle) & (angles <= max_angle)
    
    final_batch_idx = batch_idx[angle_mask]
    final_angles = angles[angle_mask]
    final_dist = valid_dist[angle_mask]
    final_points = valid_points[angle_mask]
    filtered_valid = final_batch_idx.numel()

    if filtered_valid == 0:
        return sampled_points, distance_points

    # 8. 计算角度桶索引
    bin_indices = torch.floor((final_angles - min_angle) / angle_step).to(torch.int64)
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # 9. 创建全局唯一标识并排序（核心优化）
    global_bin_id = final_batch_idx * num_bins + bin_indices  # [filtered_valid]
    
    # 按全局ID+距离排序（确保同组内最近点在最前）
    # 排序键：全局ID * 1e6 + 距离（保证同组内距离小的在前）
    sort_keys = global_bin_id * 1000000 + final_dist
    sorted_indices = torch.argsort(sort_keys)
    
    # 排序后的数据
    sorted_gid = global_bin_id[sorted_indices]
    sorted_dist = final_dist[sorted_indices]
    sorted_points = final_points[sorted_indices]
    sorted_batch = final_batch_idx[sorted_indices]
    sorted_bin = bin_indices[sorted_indices]

    # 10. 替代torch.unique(return_index=True)的核心逻辑
    # 找到分组边界（相同全局ID的连续区域）
    # 差异不为0的位置即为新分组的开始
    group_mask = sorted_gid[1:] != sorted_gid[:-1]  # [filtered_valid-1]
    # 生成分组边界索引：[0] + 边界位置 + [结束位置]
    group_boundaries = torch.cat([
        torch.tensor([0], device=device),
        torch.where(group_mask)[0] + 1,
        torch.tensor([filtered_valid], device=device)
    ])  # [num_groups + 1]
    
    # 每个分组的第一个元素索引就是我们需要的"唯一索引"
    unique_indices = group_boundaries[:-1]  # [num_groups]

    # 11. 提取唯一桶的信息
    unique_batch = sorted_batch[unique_indices]
    unique_bin = sorted_bin[unique_indices]
    unique_dist = sorted_dist[unique_indices]
    unique_points = sorted_points[unique_indices]

    # 12. 批量更新结果
    sampled_points[unique_batch, unique_bin] = unique_points
    distance_points[unique_batch, unique_bin] = unique_dist

    return sampled_points, distance_points

def project_and_sample_3d(point_array, perception_dit):
    b, n, _ = point_array.shape
    device = point_array.device
    num_bins = perception_dit['num_bins']
    angle_step = perception_dit['angle_step']
    min_angle = perception_dit['min_angle']
    max_angle = perception_dit['max_angle']
    
    # 定义三个z轴高度范围 [min, max)
    z_ranges = perception_dit['z_ranges']
    num_ranges = len(z_ranges)
    N = num_ranges * num_bins  # 总采样点数量

    # 1. 提取坐标并预计算距离
    x, y, z = point_array[..., 0], point_array[..., 1], point_array[..., 2]
    distance = torch.sqrt(x.square() + y.square())  # [b, n]
    
    # 2. 预生成结果张量，形状为[b, N, 3]
    sampled_points = torch.full((b, N, 3), 0.0, dtype=torch.float32, device=device)

    # 3. 预计算所有角度桶的基础数据
    bin_centers = min_angle + torch.arange(num_bins, device=device) * angle_step
    cos_vals = torch.cos(bin_centers)
    sin_vals = torch.sin(bin_centers)

    # 4. 对每个z轴高度范围分别处理
    for range_idx, (z_min, z_max) in enumerate(z_ranges):
        # 计算当前高度范围在最终结果中的起始索引
        start_idx = range_idx * num_bins
        
        # 初始化当前高度范围的xy坐标
        sampled_points[:, start_idx:start_idx+num_bins, 0] = 6.0 * cos_vals.unsqueeze(0)
        sampled_points[:, start_idx:start_idx+num_bins, 1] = 6.0 * sin_vals.unsqueeze(0)
        sampled_points[:, start_idx:start_idx+num_bins, 2] = 0.2 * range_idx - 0.2

        # 计算当前高度范围的有效点掩码
        valid_mask = (z > z_min) & (z <= z_max) & (distance >= 0.2) & (distance <= 6.0)
        batch_idx, point_idx = torch.where(valid_mask)  # [total_valid]
        total_valid = batch_idx.numel()

        # 处理无有效点情况
        if total_valid == 0:
            continue

        # 提取有效点数据
        valid_x = x[batch_idx, point_idx]
        valid_y = y[batch_idx, point_idx]
        valid_dist = distance[batch_idx, point_idx]
        valid_points = point_array[batch_idx, point_idx]
        
        # 角度计算与过滤
        angles = torch.atan2(valid_y, valid_x)
        angle_mask = (angles >= min_angle) & (angles <= max_angle)
        
        final_batch_idx = batch_idx[angle_mask]
        final_angles = angles[angle_mask]
        final_dist = valid_dist[angle_mask]
        final_points = valid_points[angle_mask]
        filtered_valid = final_batch_idx.numel()

        if filtered_valid == 0:
            continue

        # 计算角度桶索引
        bin_indices = torch.floor((final_angles - min_angle) / angle_step).to(torch.int64)
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        # 创建全局唯一标识并排序
        global_bin_id = final_batch_idx * num_bins + bin_indices  # [filtered_valid]
        
        # 按全局ID+距离排序（确保同组内最近点在最前）
        sort_keys = global_bin_id * 1000000 + final_dist
        sorted_indices = torch.argsort(sort_keys)
        
        # 排序后的数据
        sorted_gid = global_bin_id[sorted_indices]
        sorted_dist = final_dist[sorted_indices]
        sorted_points = final_points[sorted_indices]
        sorted_batch = final_batch_idx[sorted_indices]
        sorted_bin = bin_indices[sorted_indices]

        # 替代torch.unique(return_index=True)的核心逻辑
        group_mask = sorted_gid[1:] != sorted_gid[:-1]  # [filtered_valid-1]
        group_boundaries = torch.cat([
            torch.tensor([0], device=device),
            torch.where(group_mask)[0] + 1,
            torch.tensor([filtered_valid], device=device)
        ])  # [num_groups + 1]
        
        # 每个分组的第一个元素索引
        unique_indices = group_boundaries[:-1]  # [num_groups]

        # 提取唯一桶的信息
        unique_batch = sorted_batch[unique_indices]
        unique_bin = sorted_bin[unique_indices]
        unique_points = sorted_points[unique_indices]

        # 计算在最终结果中的索引位置
        result_indices = start_idx + unique_bin
        
        # 批量更新当前高度范围的结果
        sampled_points[unique_batch, result_indices] = unique_points

    return sampled_points

def get_lidar_obs(sensor, robot, perception_dit, min_target):
    
    robot_base_quat_w = robot.data.root_quat_w
    hit_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    hit_vec[torch.isinf(hit_vec)] = 0.0
    hit_vec[torch.isnan(hit_vec)] = 0.0
    
    hit_vec_shape = hit_vec.shape
    hit_vec = hit_vec.view(-1, hit_vec.shape[-1])
    sensor_quat_default = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot_base_quat_w.device).unsqueeze(0).repeat(hit_vec_shape[0], 1)
    sensor_quat_w = math_utils.quat_mul(robot_base_quat_w, sensor_quat_default)
    quat_w_dup = (sensor_quat_w.unsqueeze(1).repeat(1, hit_vec_shape[1], 1)).view(-1, sensor_quat_w.shape[-1])
    hit_vec_lidar_frame = math_utils.quat_apply_inverse(quat_w_dup, hit_vec)
    hit_vec_lidar_frame = hit_vec_lidar_frame.view(hit_vec_shape[0], hit_vec_shape[1], hit_vec_lidar_frame.shape[-1])
    num_envs = hit_vec_lidar_frame.shape[0]
    
    if perception_dit['type'] == '3d':
        res = project_and_sample_3d(hit_vec_lidar_frame, perception_dit)
        differences = res[..., :2] - min_target
    else:
        pc, res = project_and_sample_2d(hit_vec_lidar_frame, perception_dit)
        res = torch.log2(res)
        differences = pc[..., :2] - min_target

    squared_distances = torch.sum(differences **2, dim=2)
    distances = torch.sqrt(squared_distances)  
    min_distance = distances.min(dim=1, keepdim=True)[0]
    
    return res.view(num_envs, -1), min_distance