import pdb
import math
import torch

from GEN.utils.utils import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, quaternion_angle_diff, get_quaternion_A2B, quaternion_rotate, normal_from_cross_product

envi_cfg = dict(
    envi_name = 'MoveGrasp',
    robot_name = 'franka_panda',
    belt_speed = 0,
    obj_init_range = ((-0.0, 0.0), (0.12, 0.12)),
    num_obj_per_envi = 1,
    move_mode = 'linear',
    obj_mode = 'daily_objs',
    cam_visualization = True,
    robot_operation_range = {'x': [0.3, 0.7], 'y': [-0.3, 0.4], 'z': [0.4, 0.6]},
)

observation_cfg = dict(
    perception_model_name = 'HSVColorInstSegment',
    obj_track_queue_max_len = 5,
    instance_seg_per_steps = 1,
)

execution_cfg = dict(
    InitSkill = 'TargetTrackSkill',
    # Parameters for the TargetTrackSkill.
    target_track_handcam_dist = 0.2,    # The target distance between the hand camera and the target object point.
    target_track_handcam_ori = [1, 0, -2],  # The target oientation of the hand camera vector starting from the vector [1, 0, 0].
    target_track_handcam_offset = [0.08, 0, 0], # For more flexibility, we can add an offset to the hand camera position.
    target_track_gripper_ctrl = 1.0,
    continuous_track_target_flag = False,
    # Parameters for the MoveToPosSkill.
    move_to_target_pos = [0.15, -0.35, 0.63],
    trajectory_planner = 'CubicTrajectoryPlanner',
    plan_step_time_interval = 0.02,
    plan_max_velocity = 0.25,
    exit_pos_tolerance = 0.01,
    # Parameters for the RotateGripperSkill.
    rotate_euler = [math.pi / 4, 0, 0,], # zyx (yaw, pitch, roll)
    rotate_error_tolerance = 0.05,
    # Parameters for the OpenGripperSkill.
    OpenGripper_T = 10,
)

control_cfg = dict(
    # Parameters for Position PID
    Position_PID_kp = 10.0,
    Position_PID_ki = 0.0,
    Position_PID_kd = 0.005,
    Position_PID_output_range = [-0.2, 0.2],
    # Parameters for Velocity PID
    Velocity_PID_kp = 0.4,
    Velocity_PID_ki = 10.0,
    Velocity_PID_kd = 0.0005,
    Velocity_PID_output_range = [-1.0, 1.0],
    # Paramemters for Angular PID
    Angular_PID_kp = 5.0,
    Angular_PID_ki = 0.0,
    Angular_PID_kd = 0.005,
    Angular_PID_output_range = [-1.0, 1.0],
)

def get_track_target_cls_func(cur_skill):
    return None

def topcam_schedule_on_objects_func(mean_obj_coor3ds, obj_cls, operation_range, cur_skill):
    '''
    Description:
        Given the 3D centers of objects in the world coordinate system, decide which object as the target object.
    Input:
        mean_obj_coor3ds shape: (num_objs, 3)
    Output:
        env_target3d shape: (3,)
    '''
    objs_in_range_flag = (mean_obj_coor3ds[:, 0] > operation_range['x'][0]) & (mean_obj_coor3ds[:, 0] < operation_range['x'][1]) \
        & (mean_obj_coor3ds[:, 1] > operation_range['y'][0]) & (mean_obj_coor3ds[:, 1] < operation_range['y'][1])
        #& (mean_obj_coor3ds[:, 2] > operation_range['z'][0]) & (mean_obj_coor3ds[:, 2] < operation_range['z'][1]) # objs_in_range_flag shape: (num_obj,)
    in_range_mean_obj_coor3ds = mean_obj_coor3ds[objs_in_range_flag]
    if in_range_mean_obj_coor3ds.shape[0] > 0:
        in_range_select_obj_idx = in_range_mean_obj_coor3ds[:, 1].argmin()
        env_target3d = in_range_mean_obj_coor3ds[in_range_select_obj_idx]
        select_obj_idx = objs_in_range_flag.nonzero()[in_range_select_obj_idx, 0]
    else:
        select_obj_idx = None
        env_target3d = torch.zeros((3,), dtype = torch.float32).cuda() * float('nan')  # If no object is in the operation range, return nan
    return env_target3d, select_obj_idx

def TargetTrackSkill_checkupdate(self, obs_dict, ctrl_signal):
    # If no target is found, there is no need to check the skill update.
    if obs_dict['track_obj_pos3d'].isnan().any(): return None, {}
    # The current robot hand end status
    hand_pos = self.envi.get_robotend_pos() # Left shape: (1, 3)
    hand_rot = self.envi.get_robotend_ori()    # Left shape: (1, 4)
    hand_linear_speed = self.envi.get_robotend_pos_vel()    # Left shape: (1, 3)
    # The target robot hand end status
    target_pos = ctrl_signal['end_pos'].clone()    # Left shape: (1, 3)
    target_rot = ctrl_signal['end_ori'].clone()    # Left shape: (1, 4)
    target_linear_speed = obs_dict['track_obj_vel3d'].clone()    # Left shape: (1, 3)
    # Check whether the skill should be updated.
    pos_dist = torch.norm(hand_pos - target_pos, dim = -1)   # Left shape: (1,)
    ori_dist = quaternion_angle_diff(hand_rot, target_rot)  # Left shape: (1,)
    linear_speed_dist = torch.norm(hand_linear_speed - target_linear_speed, dim = -1)   # Left shape: (1,)
    update_flag = (pos_dist < 0.02) & (ori_dist < 0.15) & (linear_speed_dist < 0.1)   # Left shape: (1,)
    if update_flag.item():
        return 'TargetTrackTeleSkill', {}
    else:
        return None, {}
    
def TargetTrackTeleSkill_checkupdate(self, obs_dict, ctrl_signal):
    if ctrl_signal['exit_flag']:
        return 'TargetTrackSkill', {}
    else:
        return None, {}
