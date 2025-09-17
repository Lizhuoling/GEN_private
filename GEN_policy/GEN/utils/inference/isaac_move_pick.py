import isaacgym
from isaacgym import gymtorch
import pdb
import sys
import math
import scipy
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from GEN.utils.utils import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, quaternion_angle_diff, get_quaternion_A2B, quaternion_rotate, normal_from_cross_product
from GEN.data_envi.move_grasp import MoveGrasp

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MovePickTestEnviManager():
    def __init__(self, cfg, policy, stats):
        self.cfg = cfg
        self.policy = policy
        self.stats = stats
        self.belt_speed = -0.08
        self.obj_init_range = ((-0.15, 0.15), (0.1, 0.12))

        basic_num_envi_per_batch = self.cfg['EVAL']['TEST_ENVI_NUM'] // self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']
        self.num_envi_per_batch_list = [basic_num_envi_per_batch for i in range(self.cfg['EVAL']['TEST_ENVI_BATCH_NUM'])]
        self.num_envi_per_batch_list[-1] += self.cfg['EVAL']['TEST_ENVI_NUM'] % self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']
        
    def inference(self,):
        rewards = np.zeros((self.cfg['EVAL']['TEST_ENVI_NUM'],), dtype = np.float32)
        isaac_manager = IsaacMovePickManager(cfg = self.cfg, mode = 'inference', num_envs = self.num_envi_per_batch_list[0], belt_speed = self.belt_speed, \
            obj_init_range = self.obj_init_range, num_obj_per_envi = 5, move_mode = self.cfg['EVAL']['MOVE_MODE'], obj_mode = self.cfg['EVAL']['OBJ_MODE'], ctrl_per_steps = 1, stats = self.stats, cam_visualization = False)

        with torch.no_grad():
            envi_start_idx = 0
            for batch_idx in range(self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']):
                print("Start inference batch {}...".format(batch_idx))
                while isaac_manager.step <= self.cfg['EVAL']['INFERENCE_MAX_STEPS']:
                    isaac_manager.run_one_step(policy = self.policy)
                    cv2.waitKey(1)
                reward = self.get_reward(isaac_manager) # reward is a numpy array with the shape of (num_envi_per_batch_list[batch_idx])
                rewards[envi_start_idx :  envi_start_idx + self.num_envi_per_batch_list[batch_idx]] = reward
                isaac_manager.reset(num_envs = self.num_envi_per_batch_list[batch_idx])
                envi_start_idx += self.num_envi_per_batch_list[batch_idx]
        
        average_reward = np.mean(rewards)
        success_rate = np.sum(rewards) / (self.cfg['EVAL']['TEST_ENVI_NUM'] * isaac_manager.num_obj_per_envi)
        reward_info = dict(
            success_rate = success_rate,
            average_reward = average_reward
        )
        print(f'\average_reward: {average_reward} success_rate: {success_rate}\n')
        return reward_info
                
    def get_reward(self, isaac_manager):
        container_dims = isaac_manager.envi.container_dims
        container_pose = isaac_manager.envi.container_pose.p
        x_range = (container_pose.x - container_dims.x / 2, container_pose.x + container_dims.x / 2)
        y_range = (container_pose.y - container_dims.y / 2, container_pose.y + container_dims.y / 2)
        z_range = (container_pose.z - container_dims.z / 2, container_pose.z + container_dims.z / 2)
        rewards = [] 
        for env_id, env_box_ids in enumerate(isaac_manager.envi.obj_idxs):
            pos_3d = isaac_manager.envi.rb_states[env_box_ids, :3]
            in_container_flags = (pos_3d[:, 0] > x_range[0]) & (pos_3d[:, 0] < x_range[1]) & (pos_3d[:, 1] > y_range[0]) & (pos_3d[:, 1] < y_range[1]) & (pos_3d[:, 2] > z_range[0]) & (pos_3d[:, 2] < z_range[1])
            rewards.append(in_container_flags.sum().item())
        rewards = np.array(rewards)
        return rewards
                
class IsaacMovePickManager():
    def __init__(self, cfg, mode, num_envs, belt_speed, obj_init_range, num_obj_per_envi, move_mode, obj_mode, ctrl_per_steps, stats, cam_visualization):
        self.cfg = cfg
        self.mode = mode
        self.num_envs = num_envs
        self.num_obj_per_envi = num_obj_per_envi
        self.cam_target3d_deque_len = 20
        self.belt_speed = belt_speed
        self.obj_init_range = obj_init_range
        self.move_mode = move_mode
        self.obj_mode = obj_mode
        self.step = 0
        self.update_perception_step_num = ctrl_per_steps
        self.stats = stats
        self.cam_visualization = cam_visualization
        self.cam_target3d_estimator = position_estimator(num_envs = self.num_envs, deque_len = self.cam_target3d_deque_len, move_speed = (0, self.belt_speed, 0), move_acc = (0.0, 0.0, 0.0))
        if self.mode == 'inference':
            self.robot_name = self.cfg['DATA']['ROBOT_NAME']
        elif self.mode == 'teleoperation':
            self.robot_name = 'franka_panda'
        self.envi = MoveGrasp(num_envs = self.num_envs, num_obj_per_envi = self.num_obj_per_envi, belt_speed = self.belt_speed, obj_init_range = self.obj_init_range, \
            move_mode = move_mode, obj_mode = self.obj_mode, robot_name = self.robot_name, seed = None, vis_marker = False)
        self.empty_run(iters = 1)
        self.top_cam_inst_seg = AllColorInstSegment()
        self.hand_cam_point_seg = PointBasedInstSegment()
        self.init_robot_end_xyz = self.envi.rb_states[self.envi.hand_idxs, :3]
        self.handtrack_ori = self.dire_vector_to_quat(v_target = [1, 0, -2], v_initial = [1, 0, 0])    # The goal direction that the hand camera looks at the target.
        self.handtrack_dist = 0.2
        self.handcam2handend_offset, self.handcam2handend_ori = self.envi.compute_handcam2handend_transform()
        self.operation_range = {'x': [0.3, 0.7], 'y': [-0.3, 0.4], 'z': [0.4, 0.6]}
        self.status_dict = dict(
            top_schedule = 0,
            hand_manipulate = 1,
            hand_postprocess = 2,
        )
        self.envi_status = torch.zeros((self.num_envs,), dtype = torch.long).cuda()
        self.hand_cam_proper_range = ((0.1 * self.envi.img_width, 0.1 * self.envi.img_height), (0.9 * self.envi.img_width, 0.9 * self.envi.img_height))
        
        self.PositionPIDs = []  
        for env_id in range(self.num_envs):
            pos_pid = PositionPID(kp = 10.0, ki = 0, kd = 0.005, output_range = [-0.2, 0.2])
            self.PositionPIDs.append(pos_pid)
        self.VelocityPIDs = []
        for env_id in range(self.num_envs):
            velo_pid = VelocityPID(kp = 0.4, ki = 10, kd = 0.0005, output_range = [-1.0, 1.0])
            self.VelocityPIDs.append(velo_pid)
        self.AngularPIDs = []
        for env_id in range(self.num_envs):
            ang_pid = AngularPID(kp = 5.0, ki = 0, kd = 0.005, output_range = [-1.0, 1.0])
            self.AngularPIDs.append(ang_pid)
        self.traj_planner = CubicTrajectoryPlanner(time_interval = 0.02)
            
        self.visualizer = VisualizeManager(data_item_num = 3, visualize_len = 150)
        self.translation_offset = None
        self.rotation_offset = None
        self.gripper_ctrl = None
        self.cur_sample_buf = None
        self.policy_prediction_ready = torch.tensor([True] * self.num_envs, dtype = torch.bool).cuda()
        self.default_gripper_ctrl = 1.0
        self.stable_speeds = torch.zeros((self.num_envs, 3), dtype =  torch.float32).cuda()
        # Used for inference   
        if self.mode == 'inference':    
            self.past_action_deques = []
            self.action_preds = []
            self.postprocess_actions = []
            self.action_exec_cnts = []
            
            init_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, self.default_gripper_ctrl], dtype = np.float32)
            action_mean, action_std = self.stats['action_mean'], self.stats['action_std']
            if type(self.stats['action_mean']) != np.ndarray:
                action_mean = action_mean.numpy()
                action_std = action_std.numpy()
            self.norm_init_action = (init_action - action_mean) / action_std
            for env_id in range(self.num_envs):
                self.past_action_deques.append(deque(maxlen = self.cfg['DATA']['PAST_ACTION_LEN']))
                self.past_action_deques[env_id].append(self.norm_init_action)
                self.action_preds.append(None)
                self.postprocess_actions.append(None)
                self.action_exec_cnts.append(0)
        
    def reset(self, num_envs = None):
        self.clean_up()
        if num_envs is None:
            new_num_envs = self.num_envs
        else:
            new_num_envs = num_envs
        self.__init__(cfg = self.cfg, mode = self.mode, num_envs = new_num_envs, belt_speed = self.belt_speed, obj_init_range = self.obj_init_range, move_mode = self.move_mode, num_obj_per_envi = self.num_obj_per_envi, \
            obj_mode = self.obj_mode, ctrl_per_steps = self.update_perception_step_num, stats = self.stats, cam_visualization = self.cam_visualization)

    def get_envs_seed(self,):
        return self.envi.random_seed
        
    def empty_run(self, iters = 1):
        '''
        Description:
            Run the simulation without control. This is used to initialize the environment.
        '''
        for _ in range(iters):
            self.envi.update_simulator_before_ctrl()
            self.envi.update_simulator_after_ctrl()
            
    def dire_vector_to_quat(self, v_target, v_initial):
        '''
        Description:
            Compute the quaternion that transforms the direction represented by default_dire to dire_vector.
        Input:
            v_target: (3,) np.ndarray, the target direction vector.
            v_initial: (3,) np.ndarray, the initial direction vector.
        '''
        
        v_target = v_target / np.linalg.norm(v_target)
        axis = np.cross(v_initial, v_target)
        axis = axis / np.linalg.norm(axis)
        theta = math.acos(np.dot(v_initial, v_target))
        q_w = math.cos(theta / 2)
        q_x = axis[0] * math.sin(theta / 2)
        q_y = axis[1] * math.sin(theta / 2)
        q_z = axis[2] * math.sin(theta / 2)
        q = np.array([q_x, q_y, q_z, q_w])
        return q
        
    def run_one_step(self, policy = None):
        '''
        Description:
            Start the main process.
        '''
        self.envi.update_simulator_before_ctrl()
        cur_time = self.envi.get_time()
        target_world3d, handcam_estimate_uv = self.motion_estimate(cur_time)    # target_world3d shape: (num_envs, 3),  handcam_estimate_uv shape: (num_envs, 2)
        
        if self.is_update_step():
            # Get observation.
            vision_obs_list = self.envi.get_vision_observations()
            vision_obs_dict = self.dict_to_cudatensor(vision_obs_list)
            
            '''
            if (self.envi_status == self.status_dict['top_schedule']).sum() > 0:   # Conduct instance segmentation and scheduling on the top camera, and use the scheduling information to guide the movement of the hand camera. 
                inst_seg_result = self.top_cam_inst_seg(vision_obs_dict['top_rgb']) # The instance segmentation result on the top camera. list dim: num_env, num_obj, 2 (color name, mask)
                top_target_world3d, top_seg_masks = self.get_top_target_3d(inst_seg_result, vision_obs_dict['top_depth'])   # top_target_world3d shape: (num_envs, 3). If no valid target in an envi, the element is (NaN, NaN, NaN)
            else:
                top_target_world3d = torch.zeros((self.num_envs, 3), dtype = torch.float32).cuda() * float('nan')
                top_seg_masks = torch.zeros((self.num_envs, self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
            valid_top_target_flag = (~torch.isnan(top_target_world3d).any(dim = 1)) & (self.envi_status == self.status_dict['top_schedule'])  # Left shape: (num_envs,)
            target_world3d[valid_top_target_flag] = top_target_world3d[valid_top_target_flag]'''
            # Update target object observation.
            inst_seg_result = self.top_cam_inst_seg(vision_obs_dict['top_rgb']) # The instance segmentation result on the top camera. list dim: num_env, num_obj, 2 (color name, mask)
            top_target_world3d, top_target_center3d, top_seg_masks = self.get_top_target_3d(inst_seg_result, vision_obs_dict['top_depth'])   # top_target_world3d shape: (num_envs, 3). If no valid target in an envi, the element is (NaN, NaN, NaN)
            valid_top_target_flag = (~torch.isnan(top_target_world3d).any(dim = 1))  # Left shape: (num_envs,)
            target_world3d[valid_top_target_flag] = top_target_world3d[valid_top_target_flag]
            
            # Add the top camera estimated 3D positions of targets into the queue.
            for envi_id, flag in enumerate(valid_top_target_flag):
                if flag: 
                    self.cam_target3d_estimator.append_value(envi_id = envi_id, pos3d = top_target_world3d[envi_id], time = cur_time)
                    self.stable_speeds[envi_id] = torch.Tensor([0, self.belt_speed, 0]).cuda()
                else:   # If no target is found, clear the queue.
                    self.cam_target3d_estimator.reset_deque(envi_id = envi_id)
                    self.stable_speeds[envi_id] = torch.Tensor([0, 0, 0]).cuda()
        
        # Compute the hand camera moving target position.
        cams_ori = torch.Tensor(self.handtrack_ori)[None].cuda().expand(self.num_envs, -1) # Left shape: (num_envs, 4)
        handcam_track_dist = torch.Tensor([self.handtrack_dist,]).cuda().expand(self.num_envs,) # Left shape: (num_envs,)
        handcams_target_pos = self.compute_handcam_target_pos3d(target_pos = target_world3d, cam_ori = cams_ori, cam_target_dist = handcam_track_dist)  # Left shape: (num_envs, 3)
        handends_target_pos = handcams_target_pos + self.handcam2handend_offset # Left shape: (num_envs, 3)
        handends_target_ori = quaternion_rotate(cams_ori, self.handcam2handend_ori[None].expand(self.num_envs, -1)) # Left shape: (num_envs, 4)
        handends_target_ori = quaternion_multiply(torch.Tensor([-0.7071, 0, -0.7071, 0]).cuda()[None].expand(self.num_envs, -1), handends_target_ori)
        norm_gripper_ctrl = torch.Tensor([self.default_gripper_ctrl, self.default_gripper_ctrl]).cuda()[None, :].expand(self.num_envs, -1)
        
        # Check whether need to update the stage.
        if self.is_update_step():
            # Get manipulation offset from teleoperation or model inference.
            if self.is_ready_for_action():
                target_handcam_uv = self.get_handcam_guidance_2d_points(top_target_center3d) # Left shape: (num_env, 2)
                envs_target_repr, hand_seg_masks = self.get_hand_target_3d(target_handcam_uv, vision_obs_dict['hand_rgb'],  vision_obs_dict['hand_depth'])
                status_preds = None
                if self.mode == 'teleoperation' and self.translation_offset is not None:
                    handends_target_pos = handends_target_pos + torch.Tensor(self.translation_offset)[None].to(handends_target_pos.device)
                    handends_target_ori = quaternion_multiply(torch.Tensor(self.rotation_offset)[None].to(handends_target_ori.device), handends_target_ori)
                    norm_gripper_ctrl = torch.Tensor([self.gripper_ctrl, self.gripper_ctrl]).cuda()[None, :].expand(self.num_envs, -1)
                    self.cur_sample_buf = self.prepare_data_batch(vision_obs_list, envs_target_repr)
                elif self.mode == 'inference':
                    actions_to_execute = [None for _ in range(self.num_envs)]
                    policy_pred_flags = (self.envi_status == self.status_dict['hand_manipulate']) & self.policy_prediction_ready
                    # Environment that needs new policy prediction
                    if policy_pred_flags.any():
                        batch_reprs, batch_past_action, batch_past_action_is_pad = self.prepare_policy_input(envs_target_repr)
                        norm_action_preds, status_preds = policy(repr = batch_reprs, past_action = batch_past_action, action = None, past_action_is_pad = batch_past_action_is_pad, \
                            action_is_pad = None, status = None, task_instruction_list = None, dataset_type = ['pick',])  # action_preds shape: (num_envs, chunk_size, state_dim), status_preds shape: (num_envs,)
                        action_mean, action_std = self.stats['action_mean'][None, None].to(norm_action_preds.device), self.stats['action_std'][None, None].to(norm_action_preds.device)
                        action_preds = norm_action_preds * action_std + action_mean
                        action_preds = action_preds[:, :self.cfg['EVAL']['EXEC_CHUNK']] # Left shape: (num_envs, exec_chunk, state_dim)            
                        action_preds = self.close_gripper_expand(actions = action_preds, k = self.cfg['EVAL']['CLOSE_GRIPPER_EXPAND'], threshold = 0.1)    
                        for env_id, flag in enumerate(policy_pred_flags):
                            if not flag: continue
                            self.action_preds[env_id] = action_preds[env_id].clone()
                            self.policy_prediction_ready[env_id] = False
                        if self.cfg['POLICY']['STATUS_PREDICT']:
                            status_preds[~policy_pred_flags] = -1   # -1 means invalid status prediction
                    else:
                        if self.cfg['POLICY']['STATUS_PREDICT']: status_preds = (-1) * torch.ones(self.num_envs, dtype = torch.long).cuda()
                    # Environment that executes a predicted action
                    policy_exec_flags = (self.envi_status == self.status_dict['hand_manipulate'])
                    if policy_exec_flags.any():
                        for env_id, flag in enumerate(policy_exec_flags):
                            if not flag: continue
                            actions_to_execute[env_id] = self.action_preds[env_id][self.action_exec_cnts[env_id]].clone()
                            action_mean, action_std = self.stats['action_mean'].to(actions_to_execute[env_id].device), self.stats['action_std'].to(actions_to_execute[env_id].device)
                            norm_actions_to_execute = (actions_to_execute[env_id] - action_mean) / action_std
                            self.past_action_deques[env_id].append(norm_actions_to_execute.cpu().numpy()) # Only add one gripper value to the deque
                            self.action_exec_cnts[env_id] += self.cfg['EVAL']['EXEC_INTERVAL']
                            if self.action_exec_cnts[env_id] >= self.action_preds[env_id].shape[0]:
                                self.policy_prediction_ready[env_id] = True
                                self.action_exec_cnts[env_id] = 0
                    for env_id, action_to_execute in enumerate(actions_to_execute):
                        if action_to_execute == None: continue
                        handends_target_pos[env_id] = handends_target_pos[env_id] + action_to_execute[:3]   # translation offset
                        handends_target_ori[env_id] = quaternion_multiply(action_to_execute[None, 3:7], handends_target_ori[env_id : env_id+1])[0]
                        norm_gripper_ctrl[env_id] = torch.cat((action_to_execute[7:], action_to_execute[7:]), dim = 0)  # gripper control
            else:
                hand_seg_masks = torch.zeros((self.num_envs, self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
                status_preds = None
                
            if self.cam_visualization:
                self.visualize_obs(vision_obs_list, target_world3d, top_seg_masks, hand_seg_masks)
                
        if self.is_ready_for_postprocess():
            gripper_release_steps = 10
            for env_id in range(self.num_envs):
                if self.envi_status[env_id] != self.status_dict['hand_postprocess']: continue
                if self.postprocess_actions[env_id] == None:    # Schedule new actions
                    p0 = self.envi.rb_states[self.envi.hand_idxs[env_id], 0:3].cpu().numpy()
                    v0 = self.envi.rb_states[self.envi.hand_idxs[env_id], 7:10].cpu().numpy()
                    pf = np.array([0.15, -0.35, 0.63], dtype = np.float32)    # The position above the center of the container.
                    vf = np.array([0, 0, 0,], dtype = np.float32)   # The robot hand is static when opening the gripper.
                    plan_positions, plan_velocities = self.traj_planner.plan_trajectory(p0 = p0, v0 = v0, pf = pf, vf = vf, v_max = 0.25)
                    plan_grippers = np.zeros((plan_positions.shape[0],), dtype = np.float32)
                    plan_positions = np.concatenate((plan_positions, np.tile(plan_positions[-1, None], (gripper_release_steps, 1))), axis = 0)
                    plan_velocities = np.concatenate((plan_velocities, np.tile(plan_velocities[-1, None], (gripper_release_steps, 1))), axis = 0)
                    plan_grippers = np.concatenate((plan_grippers, np.ones((gripper_release_steps,), dtype = np.float32)), axis = 0)
                    self.postprocess_actions[env_id] = [plan_positions, plan_velocities, plan_grippers]
                handend_pos = self.postprocess_actions[env_id][0][self.action_exec_cnts[env_id]]
                handend_vel = self.postprocess_actions[env_id][1][self.action_exec_cnts[env_id]]
                norm_gripper_ctrl_value = self.postprocess_actions[env_id][2][self.action_exec_cnts[env_id]]
                self.action_exec_cnts[env_id] += 1
                handends_target_pos[env_id] = torch.Tensor(handend_pos).cuda()
                self.stable_speeds[envi_id] = torch.Tensor(handend_vel).cuda()
                norm_gripper_ctrl[env_id] = torch.Tensor([norm_gripper_ctrl_value, norm_gripper_ctrl_value]).cuda()
            
        if self.is_update_step():
            self.update_stage(handends_target_pos = handends_target_pos, handends_target_ori = handends_target_ori, hand_seg_masks = hand_seg_masks, valid_top_target_flag = valid_top_target_flag, \
                policy_status_predict = status_preds)
            
        # Denorm the gripper ctrl.
        gripper_upper_limit, gripper_lower_limit = torch.Tensor(self.envi.robot_upper_limits[7:]).cuda()[None], torch.Tensor(self.envi.robot_lower_limits[7:]).cuda()[None]
        norm_gripper_ctrl[norm_gripper_ctrl < 0.5] = 0
        gripper_ctrl = norm_gripper_ctrl * (gripper_upper_limit - gripper_lower_limit) + gripper_lower_limit
        
        # Compute the PID control signal.
        no_target_flags = handends_target_pos.isnan().any(dim = 1)    # Left shape: (num_envs,)
        pos_ctrls = []
        ori_ctrls = []
        for env_id, no_target_flag in enumerate(no_target_flags):
            if no_target_flag: handends_target_pos[env_id] = self.init_robot_end_xyz[env_id, :3]
            velo_tgt = self.PositionPIDs[env_id](tgt_pos = handends_target_pos[env_id], cur_pos = self.envi.rb_states[self.envi.hand_idxs[env_id], :3], time = cur_time, stable_speed = self.stable_speeds[env_id])    # velo_tgt shape: (3,)
            ctrl_pos = self.VelocityPIDs[env_id](tgt_vel = velo_tgt, cur_vel = self.envi.rb_states[self.envi.hand_idxs[env_id], 7:10], \
                cur_pos = self.envi.rb_states[self.envi.hand_idxs[env_id], :3], time = cur_time)    # ctrl_pos shape: (3,)
            ctrl_ori = self.AngularPIDs[env_id](tgt_ori = handends_target_ori[env_id], cur_ori = self.envi.rb_states[self.envi.hand_idxs[env_id], 3:7], time = cur_time)
            pos_ctrls.append(ctrl_pos)
            ori_ctrls.append(ctrl_ori)
        pos_ctrls = torch.stack(pos_ctrls, dim = 0) # Left shape: (num_envs, 3)
        ori_ctrls = torch.stack(ori_ctrls, dim = 0) # Left shape: (num_envs, 4)
        action = torch.cat((pos_ctrls, ori_ctrls, gripper_ctrl), dim = 1)   # Left shape: (num_envs, 9)
        self.envi.execute_action_ik(action)
        
        self.envi.update_simulator_after_ctrl()
        self.step += 1   
            
    def clean_up(self,):
        self.envi.clean_up()
        
    def is_update_step(self,):
        return self.step % self.update_perception_step_num == 0
    
    def is_ready_for_action(self,):
        envs_in_manipulate = (self.envi_status == self.status_dict['hand_manipulate'])
        return envs_in_manipulate.any()
    
    def is_ready_for_postprocess(self,):
        return (self.envi_status == self.status_dict['hand_postprocess']).any()
        
    def update_stage(self, handends_target_pos, handends_target_ori, hand_seg_masks, valid_top_target_flag, policy_status_predict):
        '''
        Description:
            Update the environment status.
        Input:
            handends_target_pos: shape: (num_envs, 3). The target position of robot hand ends.
            handends_target_ori: shape: (num_envs, 4). The target orientation of robot hand ends.
            hand_seg_masks: shape: (num_envs, h, w). The segmentation result of the hand camera.
            valid_top_target_flag: shape: (num_envs,). Whether the target pos is obtained from the top camera segmentation.
            policy_status_predict: None or shape (num_envs,). The status predicted by the policy. 0 is in manipulation, 1 is complete successfully, 2 is failed, and -1 is invalid.
        '''
        # Update from hand_manipulate to top_schedule
        if self.mode != 'teleoperation':
            update_flag = (hand_seg_masks.sum(dim = (1, 2)) == 0) & (self.envi_status == self.status_dict['hand_manipulate'])
            if update_flag.any():
                self.envi_status[update_flag] = self.status_dict['top_schedule']
                self.cam_target3d_estimator.reset_all_deque()
        
        # Update from top_schedule to hand_manipulate
        top_schedule_flag = (self.envi_status == self.status_dict['top_schedule'])  # Left shape: (num_envs,)
        if top_schedule_flag.any():
            hand_pos = self.envi.rb_states[self.envi.hand_idxs, :3] # Left shape: (num_envs, 3)
            hand_rot = self.envi.rb_states[self.envi.hand_idxs, 3:7]    # Left shape: (num_envs, 4)
            hand_linear_speed = self.envi.rb_states[self.envi.hand_idxs, 7:10]    # Left shape: (num_envs, 3)
            target_pos = handends_target_pos.clone()    # Left shape: (num_envs, 3)
            target_rot = handends_target_ori.clone()    # Left shape: (num_envs, 4)
            target_linear_speed = torch.Tensor([0, self.belt_speed, 0]).cuda()[None].expand(self.envi.num_envs, -1)    # Left shape: (num_envs, 3)
            pos_dist = torch.norm(hand_pos - target_pos, dim = -1)   # Left shape: (num_envs,)
            ori_dist = quaternion_angle_diff(hand_rot, target_rot)  # Left shape: (num_envs,)
            linear_speed_dist = torch.norm(hand_linear_speed - target_linear_speed, dim = -1)   # Left shape: (num_envs,)
            correct_track_flag = (pos_dist < 0.02) & (ori_dist < 0.15) & (linear_speed_dist < 0.1)   # Left shape: (num_envs,)
            status_update_flag = (top_schedule_flag & correct_track_flag & valid_top_target_flag)   # Left shape: (num_envs,)
            self.envi_status[status_update_flag] = self.status_dict['hand_manipulate']
            for env_id, env_update_flag in enumerate(status_update_flag):
                if self.mode == 'inference' and env_update_flag: 
                    self.action_exec_cnts[env_id] = 0
                    self.policy_prediction_ready[env_id] = True
                    self.action_preds[env_id] = None
                    self.past_action_deques[env_id] = deque(maxlen = self.cfg['DATA']['PAST_ACTION_LEN'])
                    self.past_action_deques[env_id].append(self.norm_init_action)
                
        # Update from hand_manipulate to hand_postprocess or top_schedule
        if policy_status_predict != None:
            update_to_hand_postprocess_flag = (self.envi_status == self.status_dict['hand_manipulate']) & (policy_status_predict == 1)
            update_to_top_schedule_flag = (self.envi_status == self.status_dict['hand_manipulate']) & (policy_status_predict == 2)
            self.envi_status[update_to_hand_postprocess_flag] = self.status_dict['hand_postprocess']
            self.envi_status[update_to_top_schedule_flag] = self.status_dict['top_schedule']
            for env_id, env_update_flag in enumerate(update_to_hand_postprocess_flag):
                if env_update_flag: 
                    self.action_exec_cnts[env_id] = 0
                    self.postprocess_actions[env_id] = None
                    
        # Update from hand_postprocess to top_schedule
        hand_postprocess_flags = (self.envi_status == self.status_dict['hand_postprocess'])  # Left shape: (num_envs,)
        for env_id, flag in enumerate(hand_postprocess_flags):
            if not flag: continue
            if self.postprocess_actions[env_id] != None and self.action_exec_cnts[env_id] >= self.postprocess_actions[env_id][0].shape[0]:
                self.action_exec_cnts[env_id] = 0
                self.postprocess_actions[env_id] = None
                self.envi_status[env_id] = self.status_dict['top_schedule']
            
    def prepare_data_batch(self, vision_obs_list, envs_target_repr):
        env_id = 0
        sample_dict = copy.deepcopy(vision_obs_list[env_id])
        sample_dict['tgt_repr'] = envs_target_repr[env_id].cpu().numpy()
        sample_dict['action'] = np.concatenate((self.translation_offset, self.rotation_offset, np.array([self.gripper_ctrl,])), axis = 0)  # self.gripper_ctrl is between 0 and 1
        sample_dict['hand_joint_pose'] = self.envi.dof_pos[env_id, :, 0].cpu().numpy()   # Left shape: (num_joints,)
        sample_dict['hand_end_pose'] = self.envi.rb_states[self.envi.hand_idxs][0].cpu().numpy()    # Left shape: (13,). 0-2: xyz, 3-6: quaternion, 7-9: linear speed, 10-12: angular speed
        return sample_dict
    
    def prepare_policy_input(self, envs_target_repr):
        repr_idxs = []
        for repr_range in self.cfg['DATA']['INPUT_REPR_KEY']:
            repr_idxs += list(range(repr_range[0], repr_range[1]))      
        batch_reprs = []
        for repr in envs_target_repr:
            repr = repr.cpu().numpy()
            repr = repr[:, repr_idxs]
            batch_reprs.append(repr)
        
        batch_past_action = []
        batch_past_action_is_pad = []
        for past_action_deque in self.past_action_deques:
            past_action_array = np.array(past_action_deque)
            if past_action_array.ndim == 1: past_action_array = past_action_array.reshape(-1, self.cfg['POLICY']['STATE_DIM'])  # shape: (past_action_valid_len, state_dim)
            past_action_valid_len = past_action_array.shape[0]
            past_action_len, past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
            past_action = np.zeros((past_action_len, self.cfg['POLICY']['STATE_DIM']), np.float32)
            if past_action_valid_len - 1 >= (past_action_len - 1) * past_action_interval:
                past_action = past_action_array[past_action_valid_len - 1 - (past_action_len - 1) * past_action_interval : past_action_valid_len : past_action_interval]
                past_action_len = self.cfg['DATA']['PAST_ACTION_LEN']
            else:
                valid_past_num = (past_action_valid_len - 1) // past_action_interval
                st = (past_action_valid_len - 1) - valid_past_num * past_action_interval
                past_action[-valid_past_num - 1:] = past_action_array[st : past_action_valid_len : past_action_interval]
                past_action_len = valid_past_num + 1
            past_action_is_pad = np.zeros(self.cfg['DATA']['PAST_ACTION_LEN'])
            past_action_is_pad[:-past_action_len] = 1   # Invalid past action
            batch_past_action.append(torch.Tensor(past_action).cuda())
            batch_past_action_is_pad.append(torch.tensor(past_action_is_pad, dtype = torch.bool).cuda())
        batch_past_action = torch.stack(batch_past_action, dim = 0) # Left shape: (num_envs, past_action_len, state_dim). batch_past_action has been normed.
        batch_past_action_is_pad = torch.stack(batch_past_action_is_pad, dim = 0) # Left shape: (num_envs, past_action_len)
        
        return batch_reprs, batch_past_action, batch_past_action_is_pad
        
    def dict_to_cudatensor(self, vis_obs):
        '''
        Description:
            Convert all the numpy arrays in vision observation dict as torch tensors on GPUs.
        '''
        keys = vis_obs[0].keys()
        obs_dict = {}
        for key in keys:
            obs_dict[key] = np.stack([ele[key] for ele in vis_obs], axis = 0)
        return {k: torch.Tensor(v).cuda() for k, v in obs_dict.items()}
    
    def get_top_target_3d(self, inst_seg_result, top_depth):
        '''
        Description:
            Get the 3D center of the target object scheduled based on the top camera in every environment. If no suitable target is found or the environment is not in the top camera 
                scheduling phase, the 3D center coordinate is (NaN, NaN, NaN).
        Input: 
            inst_seg_result: a list. list dim: num_env, num_obj, 2 (color name, mask)
            top_depth: shape: (num_env, img_h, img_w)
        '''
        world_coors = self.envi.get_top_cam_world_coors(top_depth)   # Left shape: (num_envs, img_height, img_width, 3)
        envs_target3d = []
        envs_center3d = []
        envs_seg_mask = []
        for envi_id, envi_segs in enumerate(inst_seg_result):
            
            '''if self.envi_status[envi_id] != self.status_dict['top_schedule']:  # Not in the top camera scheduling phase
                envs_target3d.append(torch.zeros((3,)).cuda() * float('nan'))
                envs_seg_mask.append(torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda())
                continue'''
            
            if len(envi_segs) != 0:
                seg_result_an_envi = torch.stack([seg_result for mask_id, seg_result in envi_segs], dim = 0) # Left shape: (num_obj, img_h, img_w)
            else:
                seg_result_an_envi = torch.zeros((0, self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()

            # erosion operation
            seg_result_an_envi = (1 - F.max_pool2d(1 - seg_result_an_envi.unsqueeze(1).float(), kernel_size=5, stride=1, padding=2)).bool().squeeze(1)  # Left shape: (num_obj, img_h, img_w)
            mask_world_coors = seg_result_an_envi.unsqueeze(-1) * world_coors[envi_id:envi_id+1]    # Left shape: (num_obj, img_h, img_w, 3)
            obj_center_coor3ds = mask_world_coors.sum(dim = (1, 2)) / seg_result_an_envi.sum(dim = (1, 2)).unsqueeze(-1)    # Left shape: (num_obj, 3)
            obj_target_coor3ds = obj_center_coor3ds.clone()
            for obj_id in range(obj_target_coor3ds.shape[0]):
                valid_obj_coor3ds = mask_world_coors[obj_id, (mask_world_coors[obj_id] != 0).all(dim = -1)]   # Left shape: (obj_valid_points, 3)
                if valid_obj_coor3ds.shape[0] != 0: obj_target_coor3ds[obj_id, 2] = valid_obj_coor3ds[:, 2].max()
            
            env_target3d, select_obj_idx = self.schedule_on_objects(obj_target_coor3ds)   # Decide the target object.
            if select_obj_idx != None:
                env_center3d = obj_center_coor3ds[select_obj_idx]
                env_seg_mask = seg_result_an_envi[select_obj_idx] # Left shape: (img_h, img_w)
            else:
                env_center3d = torch.zeros((3,)).cuda() * float('nan')
                env_seg_mask = torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
            envs_target3d.append(env_target3d)
            envs_center3d.append(env_center3d)
            envs_seg_mask.append(env_seg_mask)
            
        envs_target3d = torch.stack(envs_target3d, dim = 0) # Left shape: (num_envs, 3)
        envs_center3d = torch.stack(envs_center3d, dim = 0) # Left shape: (num_envs, 3)
        envs_seg_mask = torch.stack(envs_seg_mask, dim = 0) # Left shape: (num_envs, img_h, img_w)
        return envs_target3d, envs_center3d, envs_seg_mask
    
    def get_hand_target_3d(self, target_handcam_uv, hand_rgb, hand_depth):
        '''
        Description:
            Given the target handcam uv coordinate, segment the target image region, derive the 3D points, and then get the new 3D center coordinates in the environment world coordinate system.
        Input:
            target_handcam_uv shape: (num_envs, 2)
            hand_rgb shape: (num_envs, img_h, img_w, 3)
            hand_depth shape: (num_envs, img_h, img_w)
        '''
        seg_results = self.hand_cam_point_seg(target_handcam_uv, hand_rgb)
        hand_coors = self.envi.get_hand_cam_cam_coors(hand_depth) # Left shape: (num_envs, img_h, img_w, 3)
        hand_normals = normal_from_cross_product(hand_coors)    # Left shape: (num_envs, img_h, img_w, 3)
        envs_target_cls = []
        for seg in seg_results:
            cls_mask = torch.zeros((hand_coors.shape[1], hand_coors.shape[2], 1), dtype = torch.float32).cuda() # background cls id is 0
            if seg[0] != None:
                cls_mask[seg[1]] = 1 # Foreground object id
                envs_target_cls.append(cls_mask)
            else:
                envs_target_cls.append(cls_mask)
        envs_target_cls = torch.stack(envs_target_cls, dim = 0) # Left shape: (num_envs, img_h, img_w, 1)
        # repr definition: 0-2: coors, 3: depth, 4: cls, 5-7: rgb, 8-11: normals
        envs_repr = torch.cat((hand_coors, hand_depth.unsqueeze(-1), envs_target_cls, hand_rgb, hand_normals), dim = -1)   # Left shape: (num_envs, img_h, img_w, 11)
        
        envs_seg_mask = []
        envs_target_repr = []
        for envi_id, (seg_color, envi_seg) in enumerate(seg_results):
            if seg_color != None:
                dilation_envi_seg = envi_seg[None, :, :, None].float()  # Left shape: (1, img_h, img_w, 1)
                dilation_envi_seg = F.max_pool2d(dilation_envi_seg, kernel_size=5, stride=1, padding=2)[0, :, :, 0].bool()  # Left shape: (img_h, img_w). Dilation operation.
                env_target_repr = envs_repr[envi_id][dilation_envi_seg]
                envs_seg_mask.append(envi_seg)
            else:
                env_target_repr = torch.zeros((0, 11), dtype = torch.float32).cuda()
                envs_seg_mask.append(torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda())
            envs_target_repr.append(env_target_repr)
        envs_seg_mask = torch.stack(envs_seg_mask, dim = 0) # Left shape: (num_envs, img_h, img_w)
        return envs_target_repr, envs_seg_mask
    
    def schedule_on_objects(self, mean_obj_coor3ds):
        '''
        Description:
            Given the 3D centers of objects in the world coordinate system, decide which object as the target object.
        Input:
            mean_obj_coor3ds shape: (num_objs, 3)
        Output:
            env_target3d shape: (3,)
        '''
        objs_in_range_flag = (mean_obj_coor3ds[:, 0] > self.operation_range['x'][0]) & (mean_obj_coor3ds[:, 0] < self.operation_range['x'][1]) \
            & (mean_obj_coor3ds[:, 1] > self.operation_range['y'][0]) & (mean_obj_coor3ds[:, 1] < self.operation_range['y'][1])
            #& (mean_obj_coor3ds[:, 2] > self.operation_range['z'][0]) & (mean_obj_coor3ds[:, 2] < self.operation_range['z'][1]) # objs_in_range_flag shape: (num_obj,)
        in_range_mean_obj_coor3ds = mean_obj_coor3ds[objs_in_range_flag]
        if in_range_mean_obj_coor3ds.shape[0] > 0:
            in_range_select_obj_idx = in_range_mean_obj_coor3ds[:, 1].argmin()
            env_target3d = in_range_mean_obj_coor3ds[in_range_select_obj_idx]
            select_obj_idx = objs_in_range_flag.nonzero()[in_range_select_obj_idx, 0]
        else:
            select_obj_idx = None
            env_target3d = torch.zeros((3,)).cuda() * float('nan')  # If no object is in the operation range, return nan
        return env_target3d, select_obj_idx
    
    def compute_handcam_target_pos3d(self, target_pos, cam_ori, cam_target_dist):
        """
        Description:
            calculate the moving target positions of cameras.
        Input:
            target_pos: The 3D positions of targets. shape: (n, 3)
            cam_ori: The target orientation quaternion of the cameras. shape: (n, 4)
            cam_target_dist: The distance between cameras and targets. shape: (n, )
        """
        rotation_matrices = quaternion_to_rotation_matrix(cam_ori)
        # Compute the direction vector from camera to target (unit vector along the z-axis in camera frame)
        direction_vector = torch.tensor([1.0, 0.0, 0.0], device=target_pos.device).repeat(target_pos.shape[0], 1)
        # Transform the direction vector to the world frame
        direction_vector_world = torch.bmm(rotation_matrices, direction_vector.unsqueeze(2)).squeeze(2)
        # Normalize the direction vector to ensure it is a unit vector
        direction_vector_world = direction_vector_world / direction_vector_world.norm(dim=1, keepdim=True)
        # Compute the camera positions by moving backwards from the target positions along the direction vector
        camera_world_position = target_pos - direction_vector_world * cam_target_dist.unsqueeze(1)
        x_offset = 0.08
        camera_world_position[:, 0] += x_offset
        
        return camera_world_position
    
    def get_handcam_guidance_2d_points(self, top_target_world3d):
        '''
        Description:
            Get 2D points in hand cam images that can be used to guide target segmentation in hand cam images. Invalid 2D points are set to NaN.  
        '''
        target_handcam_uv = self.envi.project_target_world3d_to_hand_camuv(top_target_world3d)  # Left shape: (num_envs, 2)
        in_range_flag = (target_handcam_uv[:, 0] > self.hand_cam_proper_range[0][0]) & (target_handcam_uv[:, 1] > self.hand_cam_proper_range[0][1]) & (target_handcam_uv[:, 0] < self.hand_cam_proper_range[1][0]) \
            & (target_handcam_uv[:, 1] < self.hand_cam_proper_range[1][1])
        target_handcam_uv[~in_range_flag] = float('nan') # Set the projected target centers that are out of the proper region in the hand camera image to nan
        return target_handcam_uv
    
    def motion_estimate(self, cur_time):
        handcam_estimate_uv = torch.zeros((self.num_envs, 2), dtype = torch.float32).cuda() # Left shape: (num_envs, 2)
        
        handcam_estimate_pos3d = []
        for env_id in range(self.num_envs):
            if self.cam_target3d_estimator.has_ele(envi_id = env_id):
                pos3d = self.cam_target3d_estimator.predict(envi_id = env_id, pred_time = cur_time)
                
                objs_in_range_flag = (pos3d[0] > self.operation_range['x'][0]) & (pos3d[0] < self.operation_range['x'][1]) & (pos3d[1] > self.operation_range['y'][0]) \
                    & (pos3d[1] < self.operation_range['y'][1]) & (pos3d[2] > self.operation_range['z'][0]) & (pos3d[2] < self.operation_range['z'][1])
                if not objs_in_range_flag:
                    pos3d = torch.zeros((3,), dtype = torch.float32).cuda() * float('nan')
                handcam_estimate_pos3d.append(pos3d)
            else:
                handcam_estimate_pos3d.append(torch.zeros((3,), dtype = torch.float32).cuda() * float('nan'))
        handcam_estimate_pos3d = torch.stack(handcam_estimate_pos3d, dim = 0) # Left shape: (num_envs, 3)
        handcam_estimate_uv = self.envi.project_target_world3d_to_hand_camuv(handcam_estimate_pos3d)
            
        return handcam_estimate_pos3d, handcam_estimate_uv
    
    def compute_handcam2handend_transform(self,):
        '''
        Description:
            Get the pose transformation from the hand camera pose to the hand end pose.
        '''
        offset, ori = self.envi.handcam_local_transform.p, self.envi.handcam_local_transform.r
        offset_tensor =  -torch.Tensor([offset.x, offset.y, offset.z]).cuda()
        ori_tensor = quaternion_conjugate(torch.Tensor([ori.x, ori.y, ori.z, ori.w]).cuda())
        return offset_tensor, ori_tensor

    
    def visualize_obs(self, vision_obs_list, target_world3d, top_seg_masks, hand_seg_masks):
        '''
        Description:
            Visualize the observation of the environment.
            target_world3d: shape (num_envs, 3)
        '''
        cmap = plt.get_cmap('Spectral_r')
        vis_envi_id = 0
        
        vision_obs = vision_obs_list[vis_envi_id] # The first environment
        top_rgb_image = vision_obs['top_rgb']
        top_depth = vision_obs['top_depth']
        hand_rgb_image = vision_obs['hand_rgb']
        hand_depth = vision_obs['hand_depth']
        
        top_bgr_image = cv2.cvtColor(top_rgb_image, cv2.COLOR_RGBA2BGR)
        top_seg_mask = top_seg_masks[vis_envi_id].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy().astype(np.uint8) * 255
        '''norm_top_depth = (top_depth - np.min(top_depth)) / max((np.max(top_depth) - np.min(top_depth)), 1e-6)
        top_depth_vis = np.ascontiguousarray(cmap(norm_top_depth)[:, :, :3])
        top_depth_vis = (top_depth_vis * 255).astype(np.uint8)'''
        hand_bgr_image = cv2.cvtColor(hand_rgb_image, cv2.COLOR_RGBA2BGR)
        hand_seg_mask = hand_seg_masks[vis_envi_id].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy().astype(np.uint8) * 255
        norm_hand_depth = (hand_depth - np.min(hand_depth)) / min(max(np.max(hand_depth) - np.min(hand_depth), 1e-6), 1e3)
        hand_depth_vis = np.ascontiguousarray(cmap(norm_hand_depth)[:, :, :3])
        hand_depth_vis = (hand_depth_vis * 255).astype(np.uint8)
        
        i_world3d = torch.cat((target_world3d, torch.ones((target_world3d.shape[0], 1), dtype = torch.float32).cuda()), dim = -1)   # Left shape: (num_envs, 4)
        # Projection the target point to the top camera
        top_cam_projs, top_cam_views = self.envi.get_cameras_parameters(cam_name = 'top_camera')
        top_cam_projs = torch.Tensor(top_cam_projs).cuda()    # Left shape: (num_envs, 4, 4)
        top_cam_views = torch.Tensor(top_cam_views).cuda()    # Left shape: (num_envs, 4, 4)
        P_camera = top_cam_views @ i_world3d.unsqueeze(-1)   # Left shape: (num_envs, 4, 1)
        P_projected = (top_cam_projs @ P_camera).squeeze(-1) # Left shape: (num_envs, 4)
        uv_ndc = P_projected[:, 0:2] / P_projected[:, 3:4]   # Left shape: (num_envs, 2)
        u_pixel = (uv_ndc[:, 0:1] + 1) * self.envi.img_width / 2   # Left shape: (num_envs, 1)
        v_pixel = (1 - uv_ndc[:, 1:2]) * self.envi.img_height / 2  # Left shape: (num_envs, 1)
        uv_pixel = torch.cat((u_pixel, v_pixel), dim = -1) # Left shape: (num_envs, 2)
        top_vis_pixel = uv_pixel[vis_envi_id].cpu().numpy().astype(np.int32)
        cv2.circle(top_bgr_image, (top_vis_pixel[0], top_vis_pixel[1]), 5, (0, 0, 255), -1)
        # Projection the target point to the hand camera
        hand_cam_projs, hand_cam_views = self.envi.get_cameras_parameters(cam_name = 'hand_camera')
        hand_cam_projs = torch.Tensor(hand_cam_projs).cuda()    # Left shape: (num_envs, 4, 4)
        hand_cam_views = torch.Tensor(hand_cam_views).cuda()    # Left shape: (num_envs, 4, 4)
        P_camera = hand_cam_views @ i_world3d.unsqueeze(-1)   # Left shape: (num_envs, 4, 1)
        P_projected = (hand_cam_projs @ P_camera).squeeze(-1) # Left shape: (num_envs, 4)
        uv_ndc = P_projected[:, 0:2] / P_projected[:, 3:4]   # Left shape: (num_envs, 2)
        u_pixel = (uv_ndc[:, 0:1] + 1) * self.envi.img_width / 2   # Left shape: (num_envs, 1)
        v_pixel = (1 - uv_ndc[:, 1:2]) * self.envi.img_height / 2  # Left shape: (num_envs, 1)
        uv_pixel = torch.cat((u_pixel, v_pixel), dim = -1) # Left shape: (num_envs, 2)
        hand_vis_pixel = uv_pixel[vis_envi_id].cpu().numpy().astype(np.int32)
        cv2.circle(hand_bgr_image, (hand_vis_pixel[0], hand_vis_pixel[1]), 5, (0, 0, 255), -1)

        vis_frame = np.concatenate((top_bgr_image, hand_bgr_image, hand_depth_vis, hand_seg_mask), axis = 1)
        cv2.imshow('vis', vis_frame)
        
    def close_gripper_expand(self, actions, k, threshold):
        '''
        Description:
            If the gripper ctrl of a action timestamp is smaller than threshold, expand this timestamp of action to k timestamps and place it at the original relative position in actions.
        Input:
            action: torch tensor with the shape of (num_envs, num_action_timestamps, 8)
            k: int
            threshold: float
        '''
        expand_action = []
        for env_id in range(actions.shape[0]):
            expand_action.append([])
            for t in range(actions.shape[1]):
                if actions[env_id, t, -1] < threshold:
                    expand_action[env_id].append(actions[env_id, t : t+1, :].expand(k, -1))
                else:
                    expand_action[env_id].append(actions[env_id, t : t+1, :])
            expand_action[env_id] = torch.cat(expand_action[env_id], dim = 0)
        return expand_action
        
    def vis_marker(self, env_id, marker_pose):
        marker_idx = self.envi.marker_idxs[env_id]
        root_state_tensor  = self.envi.gym.acquire_actor_root_state_tensor(self.envi.sim)
        root_state_tensor = gymtorch.wrap_tensor(root_state_tensor)
        marker_state = torch.cat((marker_pose, torch.zeros((6,), dtype = torch.float32).cuda()), dim = -1)  # Left shape: (13,)
        root_state_tensor[marker_idx] = marker_state
        self.envi.gym.set_actor_root_state_tensor_indexed(self.envi.sim, gymtorch.unwrap_tensor(root_state_tensor), gymtorch.unwrap_tensor(torch.tensor([marker_idx], dtype = torch.int32).cuda()), 1)
        
class AllColorInstSegment():
    '''
    Description:
        Given an image batch (b, h, w, 3), segment all instances in this batch based on color filtering (there is at most one object for each color in the same image).
    '''
    def __init__(self, ):
        filter_cfg = {
            'red': [(-15, 0.3, 0.2), (15, 1.0, 1.0)],   # The correct hue range is 0~15 and 345~360
            'green': [(100, 0.3, 0.2), (140, 1.0, 1.0)],
            'blue': [(220, 0.3, 0.2), (260, 1.0, 1.0)],
            'purple': [(280, 0.3, 0.2), (320, 1.0, 1.0)],
            'yellow': [(40, 0.3, 0.2), (80, 1.0, 1.0)],
            'cyan': [(160, 0.3, 0.2), (200, 1.0, 1.0)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}

    def __call__(self, imgs):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, img_h, img_w, 3)
        '''
        norm_img = imgs / 255   # Left shape: (bs, img_h, img_w, 3)
        hsv_imgs = torch_rgb_to_hsv(norm_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(1)  # Left shape: (bs, 1, img_h, img_w, 3)
        hsv_imgs[[hsv_imgs[..., 0] > 345]] -= 360   # Convert the hue range of red from 345~360 to -15~0
        
        color_thre_min = torch.stack([ele[0] for ele in self.filter_cfg.values()])[None, :, None, None, :]  # Left shape: (1, num_color, 1, 1, 3)
        color_thre_max = torch.stack([ele[1] for ele in self.filter_cfg.values()])[None, :, None, None, :]   # Left shape: (1, num_color, 1, 1, 3)
        obj_mask = (hsv_imgs >= color_thre_min) & (hsv_imgs <= color_thre_max) # Left shape: (bs, num_color, img_h, img_w, 3)
        obj_mask = torch.all(obj_mask, dim = -1) # Left shape: (bs, num_color, img_h, img_w)
        
        inst_seg_results = []
        for bs_id in range(obj_mask.shape[0]):
            inst_seg_results.append([])
            for color_id in range(obj_mask.shape[1]):
                if obj_mask[bs_id, color_id].sum() > 3:
                    inst_seg_results[bs_id].append((list(self.filter_cfg.keys())[color_id], obj_mask[bs_id, color_id]))
        return inst_seg_results
      
      
class PointBasedInstSegment():
    '''
    Description:
        Given an image batch (b, h, w, 3) and one pixel coordinate for each image, segment the instance that the pixel belongs to for each image. The pixel could be None.
    '''
    def __init__(self, ):
        filter_cfg = {
            'red': [(-15, 0.3, 0.2), (15, 1.0, 1.0)],   # The correct hue range is 0~15 and 345~360
            'green': [(100, 0.3, 0.2), (140, 1.0, 1.0)],
            'blue': [(220, 0.3, 0.2), (260, 1.0, 1.0)],
            'purple': [(280, 0.3, 0.2), (320, 1.0, 1.0)],
            'yellow': [(40, 0.3, 0.2), (80, 1.0, 1.0)],
            'cyan': [(160, 0.3, 0.2), (200, 1.0, 1.0)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}  
        self.inst_seg = AllColorInstSegment()
        
    def __call__(self, points, images):
        '''
        Input:
            points: torch tensor with the shape of (bs, 2)
            images: normalized torch tensor with the shape of (bs, img_h, img_w, 3)
        '''
        assert points.shape[0] == images.shape[0], 'points and images should have the same batch size'
        seg_results = []
        bs, h, w, _ = images.shape
        inst_seg_results = self.inst_seg(images)
        for env_id, (point, inst_seg_result) in enumerate(zip(points, inst_seg_results)):
            if len(inst_seg_result) == 0:   # No object found in the hand camera observation of this environment
                seg_results.append((None, torch.zeros((h, w), dtype = torch.bool).cuda()))
                continue
            
            mask_min_area = 1e6
            seg_mask = torch.zeros((h, w), dtype = torch.bool).cuda()
            seg_mask_color_name = None
            for color_name, seg_result in inst_seg_result:
                if self.point_inside_mask(point, seg_result):  # The point is within this mask
                    if seg_result.sum() < mask_min_area:
                        mask_min_area = seg_result.sum()
                        seg_mask = seg_result
                        seg_mask_color_name = color_name
                    
            seg_results.append((seg_mask_color_name, seg_mask))
        return seg_results
    
    def point_inside_mask(self, point, mask):
        if point.isnan().any(): return False
        point = point.cpu().numpy()
        mask = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [max(contours, key=lambda ele: ele.shape[0])]
        contours = [cv2.convexHull(c, returnPoints=True) for c in contours]
        assert len(contours) == 1, 'The mask should only have one contour'
        contour = contours[0]
        result = cv2.pointPolygonTest(contour, point, False)
        if result > 0:  # The point is inside the contour
            return True
        else:
            return False
        
        
def torch_rgb_to_hsv(image):
    """
    Input:
        image: normalized image tensor with the shape of (B, 3, H, W). Pixel value range: (0, 1)
    Output:
        hsv: The converted HSV image with the shape of (B, 3, H, W). Hue range is (0, 360) and the ranges of value and Saturation are (0, 1)
    """
    max_val, _ = torch.max(image, dim=1, keepdim=True)  # (B, 1, H, W)
    min_val, _ = torch.min(image, dim=1, keepdim=True)  # (B, 1, H, W)
    # Compute Value
    V = max_val  # V = max(R, G, B)
    # Compute Saturation
    delta = max_val - min_val
    S = delta / (max_val + 1e-7)
    S[delta == 0] = 0
    # Compute Hue
    R, G, B = image[:, 0:1, :, :], image[:, 1:2, :, :], image[:, 2:3, :, :]
    H = torch.zeros_like(V)

    mask = (max_val == R)
    H[mask] = 60 * ((G[mask] - B[mask]) / (delta[mask] + 1e-7) % 6)
    mask = (max_val == G)
    H[mask] = 60 * ((B[mask] - R[mask]) / (delta[mask] + 1e-7) + 2)
    mask = (max_val == B)
    H[mask] = 60 * ((R[mask] - G[mask]) / (delta[mask] + 1e-7) + 4)
    H[delta == 0] = 0
    hsv_image = torch.cat([H, S, V], dim=1)
    return hsv_image

class position_estimator():
    def __init__(self, num_envs, deque_len, move_speed, move_acc):
        self.num_envs = num_envs
        self.deque_len = deque_len
        self.move_speed = torch.Tensor(move_speed).cuda()
        self.move_acc = torch.Tensor(move_acc).cuda()

        self.cam_target3d_deque_list = [deque(maxlen = deque_len) for _ in range(self.num_envs)]
        self.time_deque_list = [deque(maxlen = deque_len) for _ in range(self.num_envs)]
        
    def append_value(self, envi_id, pos3d, time):
        self.cam_target3d_deque_list[envi_id].append(pos3d)
        self.time_deque_list[envi_id].append(time)
        
    def reset_deque(self, envi_id):
        self.cam_target3d_deque_list[envi_id] = deque(maxlen = self.deque_len)
        self.time_deque_list[envi_id] = deque(maxlen = self.deque_len)
    
    def reset_all_deque(self,):
        for envi_id in range(self.num_envs):
            self.cam_target3d_deque_list[envi_id] = deque(maxlen = self.deque_len)
            self.time_deque_list[envi_id] = deque(maxlen = self.deque_len)
        
    def has_ele(self, envi_id):
        return len(self.cam_target3d_deque_list[envi_id]) > 0
    
    def len(self, envi_id):
        return len(self.cam_target3d_deque_list[envi_id])
    
    def predict(self, envi_id, pred_time):
        target_pos3d = self.cam_target3d_deque_list[envi_id]
        time_deque = self.time_deque_list[envi_id]
        x_traj = np.array([ele[0].cpu().numpy() for ele in target_pos3d])
        y_traj = np.array([ele[1].cpu().numpy() for ele in target_pos3d])
        z_traj = np.array([ele[2].cpu().numpy() for ele in target_pos3d])
        t_traj = np.array([ele for ele in time_deque]).reshape(-1, 1)
        
        kernel = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp_x = GaussianProcessRegressor(kernel=kernel)
        gp_y = GaussianProcessRegressor(kernel=kernel)
        gp_z = GaussianProcessRegressor(kernel=kernel)
        gp_x.fit(t_traj, x_traj)
        gp_y.fit(t_traj, y_traj)
        gp_z.fit(t_traj, z_traj)
        t_tgt = np.array([[pred_time,],])
        x_pred = gp_x.predict(t_tgt)
        y_pred = gp_y.predict(t_tgt)
        z_pred = gp_z.predict(t_tgt)

        model_pred_pos3d = target_pos3d[-1] + (pred_time - time_deque[-1]) * self.move_speed + (pred_time - time_deque[-1]) ** 2 * self.move_acc / 2
        pred_pos3d = torch.Tensor(np.concatenate((x_pred, y_pred, z_pred), axis = -1)).cuda()   # Left shape: (3,)

        return pred_pos3d
    
class PositionPID():
    def __init__(self, kp, ki, kd, output_range):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
        self.output_range = output_range
        
    def __call__(self, tgt_pos, cur_pos, time, stable_speed):
        '''
        Input:
            tgt_pos shape: (3,)
            cur_pos shape: (3,)
            time shape: a float
            stable_speed shape: (3,) 
        Output:
            velo_tgt shape: (3)
        '''
        error = tgt_pos - cur_pos
        time_interval = time - self.prev_time
        self.integral += error * time_interval
        derivative = (error - self.prev_error) / time_interval
        velo_tgt = self.kp * error + self.ki * self.integral + self.kd * derivative
        velo_tgt = velo_tgt + stable_speed
        self.prev_error = error
        self.prev_time = time
        velo_tgt = torch.clamp(velo_tgt, self.output_range[0], self.output_range[1])
        return velo_tgt
        
    def reset(self, time):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
        
class VelocityPID():
    def __init__(self, kp, ki, kd, output_range):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
        self.output_range = output_range
        
    def __call__(self, tgt_vel, cur_vel, cur_pos, time):
        '''
        Input:
            tgt_vel shape: (3,)
            cur_vel shape: (3,)
            cur_pos shape: (3,)
            time shape: a float
        Output:
            ctrl shape: (3)
        '''
        error = tgt_vel - cur_vel
        time_interval = time - self.prev_time
        self.integral += error * time_interval
        derivative = (error - self.prev_error) / time_interval
        ctrl = self.kp * error + self.ki * self.integral + self.kd * derivative
        ctrl = torch.clamp(ctrl, self.output_range[0], self.output_range[1])
        ctrl = ctrl + cur_pos
        self.prev_error = error
        self.prev_time = time
        return ctrl
        
    def reset(self, time):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
        
class AngularPID():
    def __init__(self, kp, ki, kd, output_range):
        """
        Initialize the Angular PID Controller
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param output_range: Output range limit (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_range = output_range  # Output range limit
        self.integral = 0  # Integral term initialized as zero vector
        self.prev_error = 0  # Previous error initialized as zero vector
        self.prev_time = 0.0  # Previous time initialized as zero

    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw)
        :param q: Quaternion, shape (4,)
        :return: Euler angles, shape (3,)
        """   
        x, y, z, w = q
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = torch.asin(2 * (w * y - z * x))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return torch.Tensor([roll, pitch, yaw]).to(q.device)

    def __call__(self, tgt_ori, cur_ori, time):
        """
        Calculate the control signal quaternion
        :param tgt_ori: Target quaternion orientation, shape (4,)
        :param cur_ori: Current quaternion orientation, shape (4,)
        :param time: Current time (float)
        :return: Control signal quaternion, shape (4,)
        """
        # Convert quaternions to Euler angles
        tgt_euler = self.quaternion_to_euler(tgt_ori)
        cur_euler = self.quaternion_to_euler(cur_ori)

        # Calculate Euler angle errors
        error = tgt_euler - cur_euler

        # Normalize errors to [-pi, pi]
        error = torch.remainder(error + torch.pi, 2 * torch.pi) - torch.pi

        # Calculate time interval
        time_interval = time - self.prev_time

        # Update integral term
        self.integral += error * time_interval

        # Calculate derivative term
        derivative = (error - self.prev_error) / time_interval

        # Calculate control signal
        control_delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Clip the control signal to the specified range
        control_delta = torch.clamp(control_delta, self.output_range[0], self.output_range[1])
        control_signal = cur_euler + control_delta
        # Update previous error and time
        self.prev_error = error
        self.prev_time = time
        # Convert control signal to quaternion
        control_quaternion = self.control_signal_to_quaternion(control_signal)
        return control_quaternion

    def control_signal_to_quaternion(self, control_signal):
        """
        Convert the control signal (Euler angles) to a quaternion
        :param control_signal: Control signal (Euler angles), shape (3,)
        :return: Control quaternion, shape (4,)
        """
        roll, pitch, yaw = control_signal
        cr = torch.cos(roll / 2)
        sr = torch.sin(roll / 2)
        cp = torch.cos(pitch / 2)
        sp = torch.sin(pitch / 2)
        cy = torch.cos(yaw / 2)
        sy = torch.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.tensor([x, y, z, w]).cuda()

    def reset(self, time):
        """
        Reset the PID controller
        :param time: Current time (float)
        """
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
        
class VisualizeManager():
    def __init__(self, data_item_num = 3, visualize_len = 100):
        self.data_item_num = data_item_num
        self.visualize_len = visualize_len
        self.visualize_data = []
        for i in range(self.data_item_num):
            self.visualize_data.append([])
        
    def add_data(self, *data):
        for i in range(self.data_item_num):
            self.visualize_data[i].append(data[i])
        if len(self.visualize_data[0]) >= self.visualize_len:
            self.visualize()
            
    def visualize(self):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.visualize_data[0]))
        for i, y_values in enumerate(self.visualize_data):
            plt.subplot(len(self.visualize_data), 1, i + 1)
            plt.plot(x, y_values, label=f"Line {i+1}", marker="o")
            plt.title(f"Line {i+1}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()
        sys.exit()

def pose_relative_transformation(A, B):
    """
    Description:
        Calculate the 4x4 transformation matrix from the pose A to the pose B.
    Input:
        A: A batch of 7D poses. The first three elements are position and the last four elements are quaternion rotation. shape: (n, 7)
        B: A batch of 7D poses. The first three elements are position and the last four elements are quaternion rotation. shape: (n, 7)
    Output:
        T_AB: A batch of 4x4 transformation matrices. shape: (n, 4, 4)
    """
    p_A, q_A = A[:, :3], A[:, 3:]  # shapes: (n, 3), (n, 4)
    p_B, q_B = B[:, :3], B[:, 3:]  # shapes: (n, 3), (n, 4)
    p_AB = p_B - p_A  # shape: (n, 3)
    q_A_inv = torch.stack([-q_A[:, 0], -q_A[:, 1], -q_A[:, 2], q_A[:, 3]], dim=-1)  # shape: (n, 4)
    q_AB = torch.stack([
        q_B[:, 0]*q_A_inv[:, 3] + q_B[:, 1]*q_A_inv[:, 2] - q_B[:, 2]*q_A_inv[:, 1] + q_B[:, 3]*q_A_inv[:, 0],
        q_B[:, 0]*q_A_inv[:, 2] + q_B[:, 1]*q_A_inv[:, 3] + q_B[:, 2]*q_A_inv[:, 0] - q_B[:, 3]*q_A_inv[:, 1],
        q_B[:, 0]*q_A_inv[:, 1] - q_B[:, 1]*q_A_inv[:, 0] + q_B[:, 2]*q_A_inv[:, 3] + q_B[:, 3]*q_A_inv[:, 2],
        q_B[:, 0]*q_A_inv[:, 0] - q_B[:, 1]*q_A_inv[:, 1] - q_B[:, 2]*q_A_inv[:, 2] - q_B[:, 3]*q_A_inv[:, 3]
    ], dim=-1)  # shape: (n, 4)
    R_AB = quaternion_to_rotation_matrix(q_AB)  # shape: (n, 3, 3)
    T_AB = torch.eye(4, dtype=A.dtype, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)  # shape: (n, 4, 4)
    T_AB[:, :3, :3] = R_AB
    T_AB[:, :3, 3] = p_AB
    return T_AB

def calculate_l1_and_giou_loss(candidate_boxes, goal_box, l1_weight = 0.01, giou_weight = 1.0):
    '''
    Input:
        candidate_boxes: shape (n, 2, 2)
        goal_box: (2, 2)
    '''
    # L1 loss
    goal_box_expanded = goal_box.unsqueeze(0).expand_as(candidate_boxes)  # (n, 2, 2)
    l1_loss = torch.abs(candidate_boxes - goal_box_expanded).sum(dim=(1, 2))  # (n,)
    
    # GIOU loss
    candidate_x1y1 = candidate_boxes[:, 0, :]  # (n, 2)
    candidate_x2y2 = candidate_boxes[:, 1, :]  # (n, 2)
    goal_x1y1 = goal_box[0, :]  # (2,)
    goal_x2y2 = goal_box[1, :]  # (2,)
    inter_x1y1 = torch.max(candidate_x1y1, goal_x1y1)  # (n, 2)
    inter_x2y2 = torch.min(candidate_x2y2, goal_x2y2)  # (n, 2)
    # Calculate intersection area
    inter_width_height = torch.clamp(inter_x2y2 - inter_x1y1, min=0)  # (n, 2)
    inter_area = inter_width_height[:, 0] * inter_width_height[:, 1]  # (n,)
    # Calculate union area
    candidate_area = (candidate_x2y2[:, 0] - candidate_x1y1[:, 0]) * (candidate_x2y2[:, 1] - candidate_x1y1[:, 1])  # (n,)
    goal_area = (goal_x2y2[0] - goal_x1y1[0]) * (goal_x2y2[1] - goal_x1y1[1])  # scalar
    union_area = candidate_area + goal_area - inter_area  # (n,)
    # Calculate enclosing area
    enclosing_x1y1 = torch.min(candidate_x1y1, goal_x1y1)  # (n, 2)
    enclosing_x2y2 = torch.max(candidate_x2y2, goal_x2y2)  # (n, 2)
    enclosing_width_height = enclosing_x2y2 - enclosing_x1y1  # (n, 2)
    enclosing_area = enclosing_width_height[:, 0] * enclosing_width_height[:, 1]  # (n,)
    # Calculate GIOU
    iou = inter_area / union_area  # (n,)
    giou = iou - (enclosing_area - union_area) / enclosing_area  # (n,)
    giou_loss = 1 - giou    # Left shape: (n,)
    
    loss = l1_weight * l1_loss + giou_weight * giou_loss
    return loss

class CubicTrajectoryPlanner:
    def __init__(self, time_interval=0.02):
        self.time_interval = time_interval

    def cubic_interpolation(self, p0, v0, pf, vf, T):
        T = T.item() if isinstance(T, np.ndarray) else T
        a = np.zeros(3)
        b = np.zeros(3)
        c = np.zeros(3)
        d = np.zeros(3)
        for i in range(3):
            A = np.array([
                [T**3, T**2],
                [3*T**2, 2*T]
            ], dtype=np.float64)

            B = np.array([
                pf[i] - p0[i] - v0[i] * T,
                vf[i] - v0[i]
            ], dtype=np.float64).reshape(-1, 1)
            ab = np.linalg.pinv(A) @ B
            a[i], b[i] = ab.flatten()
            c[i] = v0[i]
            d[i] = p0[i]

        return a, b, c, d

    def cubic_velocity_constraint(self, T, p0, v0, pf, vf, v_max):
        T = T.item() if isinstance(T, np.ndarray) else T
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T)
        t = np.linspace(0, T, 100)
        v = 3 * a * t[:, np.newaxis]**2 + 2 * b * t[:, np.newaxis] + c
        speed = np.linalg.norm(v, axis=1)
        return v_max - np.max(speed)  # Ensure speed <= v_max

    def cubic_velocity_constraint_min(self, T, p0, v0, pf, vf, v_max):
        T = T.item() if isinstance(T, np.ndarray) else T
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T)
        t = np.linspace(0, T, 100)
        v = 3 * a * t[:, np.newaxis]**2 + 2 * b * t[:, np.newaxis] + c
        speed = np.linalg.norm(v, axis=1)
        return np.min(speed) + v_max  # Ensure speed >= -v_max

    def plan_trajectory(self, p0, v0, pf, vf, v_max):
        distance = np.linalg.norm(pf - p0)
        T_guess = max(distance / v_max, 1e-3)
        result = scipy.optimize.minimize(
            lambda T: T[0],
            T_guess,
            constraints=[
                {'type': 'ineq', 'fun': lambda T: self.cubic_velocity_constraint(T[0], p0, v0, pf, vf, v_max)},
            ],
            bounds=[(1e-6, None)],  # T > 0
            method='SLSQP',
            options={'maxiter': 100}
        )
        T_opt = result.x[0]
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T_opt)
        num_points = int(T_opt / self.time_interval) + 1
        t = np.linspace(0, T_opt, num_points)
        t = t[:, np.newaxis]
        positions = a * t**3 + b * t**2 + c * t + d  # shape (num_points, 3)
        velocities = 3 * a * t**2 + 2 * b * t + c    # shape (num_points, 3)
        return positions, velocities
        
if __name__ == '__main__':
    manager = IsaacMovePickManager(cfg = None, mode = 'teleoperation', num_envs = 2, belt_speed = -0.2, obj_init_range = ((-0.15, 0.15), (0.1, 0.12)), move_mode = 'linear', ctrl_per_steps = 1, cam_visualization = True)
    while True:
        manager.run_one_step()
        if cv2.waitKey(1) == ord('q'): break
    manager.clean_up()