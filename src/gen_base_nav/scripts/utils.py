import numpy as np
from geometry_msgs.msg import Vector3, Quaternion

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