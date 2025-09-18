import numpy as np
import torch
import os
import random
import copy
import pickle
import cv2
import logging
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

class NavImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_dir, ids_map_dict, indices, is_train):
        super().__init__()
        self.cfg = cfg
        self.dataset_dir = dataset_dir
        self.ids_key_list = list(ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in ids_map_dict.values()])
        self.indices = indices
        self.is_train = is_train
        self.camera_names = self.cfg['DATA']['CAMERA_NAMES']
        self.logger = logging.getLogger("GEN")

    def __len__(self):
        return len(self.indices)
    
    def map_value_to_letter(self, index):
        range_id = np.searchsorted(self.ids_start_array, index, side = 'right') - 1
        hdf5_file = self.ids_key_list[range_id]
        return hdf5_file, index - self.ids_start_array[range_id]
    
    def load_video_frame(self, video_path, frame_id):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame_rgb

    def __getitem__(self, index):
        indice_index  = self.indices[index]
        hdf5_file_name, hdf5_frame_id =  self.map_value_to_letter(indice_index)
        h5py_path = os.path.join(self.dataset_dir, 'h5py', hdf5_file_name)
        
        with h5py.File(h5py_path, 'r') as root:
            sample_root = root[f'/samples/sample_{hdf5_frame_id}']
            
            cmd_linear = sample_root['target_linear'][:].astype(np.float32)  # Left shape: (3,)
            cmd_angular = sample_root['target_angular'][:].astype(np.float32)  # Left shape: (3,)
            ctrl_cmd = np.concatenate((cmd_linear, cmd_angular), axis=0)  # Left shape: (6,)
            cur_linear_acc = sample_root['cur_linear_acc'][:].astype(np.float32)    # Left shape: (3,)
            cur_angular_vel = sample_root['cur_angular_vel'][:].astype(np.float32)  # Left shape: (3,)
            cur_status = np.concatenate((cur_linear_acc, cur_angular_vel), axis=0)  # Left shape: (6,)
            global_plan = sample_root['global_plan'][:].astype(np.float32)  # Left shape: (unfixed_n, 7)
            padded_global_plan_mask = np.zeros((self.cfg['DATA']['GLOBAL_PLAN_LENGTH'],), dtype=np.bool)    # Left shape: (GLOBAL_PLAN_LENGTH,)
            if global_plan.shape[0] > self.cfg['DATA']['GLOBAL_PLAN_LENGTH']:
                padded_global_plan = global_plan[:self.cfg['DATA']['GLOBAL_PLAN_LENGTH']]   # Left shape: (GLOBAL_PLAN_LENGTH, 7)
            else:
                padded_global_plan = np.concatenate((global_plan, np.zeros((self.cfg['DATA']['GLOBAL_PLAN_LENGTH'] - global_plan.shape[0], \
                    global_plan.shape[1]), dtype=np.float32)), axis=0)  # Left shape: (GLOBAL_PLAN_LENGTH, 7)
                padded_global_plan_mask[global_plan.shape[0]:] = True
                
            image_list = []
            for camera_name in self.camera_names:
                image_list.append(sample_root[f'{camera_name}_rgb'][:].astype(np.float32))  # Left shape: (H, W, 3)
            image_array = np.stack(image_list, axis=0)  # Left shape: (N, H, W, 3)
        
        ctrl_cmd = torch.from_numpy(ctrl_cmd).float()   # left shape: (6,)
        cur_status = torch.from_numpy(cur_status).float()   # left shape: (6,)
        padded_global_plan = torch.from_numpy(padded_global_plan).float()   # left shape: (GLOBAL_PLAN_LENGTH, 7)
        padded_global_plan_mask = torch.from_numpy(padded_global_plan_mask).bool()    # left shape: (GLOBAL_PLAN_LENGTH,)
        image_array = torch.from_numpy(image_array).float()    # Left shape: (N, H, W, 3)

        return ctrl_cmd, cur_status, padded_global_plan, padded_global_plan_mask, image_array
    
def NavImage_collate_fn(batch):
    batch_keys = ['ctrl_cmd', 'cur_status', 'padded_global_plan', 'padded_global_plan_mask', 'image_array']
    all_data = {}
    stack_data_list = []
    for key in batch_keys:
        all_data[key] = []
    for one_batch_data in batch:
        assert len(one_batch_data) == len(batch_keys)
        for tensor, key in zip(one_batch_data, batch_keys):
            all_data[key].append(tensor)
    
    for key in batch_keys:
        stack_data = torch.stack(all_data[key], dim=0)
        stack_data_list.append(stack_data)

    return stack_data_list