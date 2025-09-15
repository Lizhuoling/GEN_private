from geometry_msgs.msg import Vector3, Quaternion

import pdb
import numpy as np
import cv2

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def transform_angular_velocity(angular_vel, R):
    omega = np.array([angular_vel.x, angular_vel.y, angular_vel.z])
    omega_parent = R @ omega
    result = Vector3()
    result.x = omega_parent[0]
    result.y = omega_parent[1]
    result.z = omega_parent[2]
    return result

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

def traj_frame_reproject(P_odom, Q_odom, T_odom2base, Q_odom2base):
    '''
    Description:
        T_odom2base vector (3,) and Q_odom2base quat (4,) are the translation and orientation from the odom frame to the base_link frame.
        P_odom (n, 3) and Q_odom (n, 4) are the position and orientation of the robot in the odom frame. 
        Now we need to calculate the position and orientation of the robot in the base_link frame, returned as P_base and Q_base.
    '''
    T_odom2base_array = np.array([T_odom2base.x, T_odom2base.y, T_odom2base.z])   # From Vector to numpy as array
    Q_odom2base_array = np.array([Q_odom2base.x, Q_odom2base.y, Q_odom2base.z, Q_odom2base.w])
    # Calculate P_base
    P_odom_minus_T = P_odom - T_odom2base_array
    R_odom_base = quaternion_to_rotation_matrix(Q_odom2base)
    P_base = P_odom_minus_T @ R_odom_base.T
    # Calculate Q_base
    Q_inv = quaternion_inverse(Q_odom2base_array)
    Q_base = quaternion_multiply(Q_inv, Q_odom)
    Q_base = Q_base / np.linalg.norm(Q_base, axis=1, keepdims=True)
    return P_base, Q_base

def get_colored_point_cloud(rgb_image, depth_image, camera_intrinsics, dist_threshold = 30):
    height, width = depth_image.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    z = depth_image.flatten()
    valid_mask = np.isfinite(z) & (z > 0) & (z < dist_threshold)
    z = z[valid_mask]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    x = x[valid_mask]
    y = y[valid_mask]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    pc_xyz = np.stack((X, Y, Z), axis=1).astype(np.float32)
    pc_rgb = rgb_image.reshape(-1, 3)[valid_mask].astype(np.float32) / 255.0
    pc_xyzrgb = np.concatenate((pc_xyz, pc_rgb), axis=1)
    return pc_xyzrgb

def transform_campc2basepc(pc_xyzrgb, R_base2cam, T_base2cam):
    '''
    Description:
        pc_xyzrgb (n, 6) is the point cloud in camera frame, where the first 3 columns are the xyz coordinates and the last 3 columns are the rgb values.
        R_base2cam and T_base2cam are the transformation from the base_link frame to the camera frame.
    '''
    pc_xyz_cam = pc_xyzrgb[:, :3]   # Left shape: (n, 3)  
    pc_rgb = pc_xyzrgb[:, 3:]   # Left shape: (n, 3)
    R_base2cam = quaternion_to_rotation_matrix(R_base2cam)  # Left shape: (3, 3)
    T_base2cam = np.array([T_base2cam.x, T_base2cam.y, T_base2cam.z])   # Left shape: (3,)
    pc_xyz_base = pc_xyz_cam @ R_base2cam.T + T_base2cam   # Left shape: (n, 3)
    pc_xyzrgb_base = np.concatenate((pc_xyz_base, pc_rgb), axis=1)   # Left shape: (n, 6)
    return pc_xyzrgb_base

def frechet_distance(traj1, traj2):
    """
    Description:
        Calculate the Frechet distance between two trajectories.
    """
    n, m = len(traj1), len(traj2)
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = np.linalg.norm(traj1[i] - traj2[j])
    dp = np.full((n, m), -1.0)
    dp[0, 0] = dist_matrix[0, 0]
    for j in range(1, m):
        dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
    for i in range(1, n):
        dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i-1, j], dp[i-1, j-1], dp[i, j-1]),
                dist_matrix[i, j]
            )
    return dp[n-1, m-1]
    