import numpy as np
import torch
import os
import random
import pickle
import logging
import tqdm
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

from utils import comm
from utils.samplers import distributed_sampler
from utils.dataset_loaders.nav_image_dataset import NavImageDataset, NavImage_collate_fn
from utils.dataset_loaders.navdp_dataset import NavDP_Base_Datset, navdp_collate_fn
from utils.dataset_loaders.concat_dataset import CustomConcatDataset

def get_norm_stats(dataset_dir, norm_keys, norm_max_len = -1):
    norm_data_dict = {key: [] for key in norm_keys}
    mean_std_dict = {}
    for key in norm_keys:
        mean_std_dict[key + '_mean'] = []
        mean_std_dict[key + '_std'] = []
    
    data_file_list = get_hdf5_list(os.path.join(dataset_dir, 'h5py'))
    if norm_max_len > 0 and len(data_file_list) > norm_max_len:
        data_file_list = data_file_list[:norm_max_len]

    for data_file in data_file_list:
        dataset_path = os.path.join(dataset_dir, 'h5py', data_file)
        with h5py.File(dataset_path, 'r') as root:
            for norm_key in norm_keys:
                norm_data_dict[norm_key].append(torch.from_numpy(root[norm_key][()]))

    for norm_key in norm_keys:
        norm_data_dict[norm_key] = torch.cat(norm_data_dict[norm_key], axis = 0)
        mean_std_dict[norm_key + '_mean'] = norm_data_dict[norm_key].mean(dim=0).float()
        mean_std_dict[norm_key + '_std'] = norm_data_dict[norm_key].std(dim=0).float()
        mean_std_dict[norm_key + '_std'] = torch.clip(mean_std_dict[norm_key + '_std'], 1e-2, np.inf) # avoid the std to be too small.
    return mean_std_dict

def get_hdf5_list(path):
    hdf5_list = []
    for file_name in os.listdir(path):
        if file_name.endswith('.hdf5'): 
            hdf5_list.append(file_name)

    if len(hdf5_list) == 0:
        raise Exception("No hdf5 file found in the path {}".format(path))
    
    return sorted(hdf5_list)

def load_data(cfg):
    if cfg['POLICY']['POLICY_NAME'] == 'GEN_navdp':
        return load_navdp_data(cfg)
    else:
        return load_my_data(cfg)

def load_navdp_data(cfg):
    if cfg['IS_DEBUG']:
        num_workers = 0
    else:
        num_workers = cfg['TRAIN']['NUM_WORKERS']
    train_sample_per_gpu = cfg['TRAIN']['BATCH_SIZE_PER_GPU']
    collate_fn = navdp_collate_fn
    train_dataset = NavDP_Base_Datset(
        cfg = cfg,
        root_dirs = cfg['DATA']['DATASET_DIR'][0],
        preload_path='/dataset/nav-e2e/data_navdp/NavDPData/navdp_dataset_GEN.json',
        preload=True,
        memory_size=1,
        predict_size=24,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
    )
    
    if comm.is_main_process():
        logger = logging.getLogger("GEN")
        logger.info("Training set sample number: {}.".format(len(train_dataset)))
        
    train_batch_sampler = DistributedSampler(train_dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK']), shuffle=True, seed=1234)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_sample_per_gpu,
        sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_dataloader, None

def load_my_data(cfg):
    dataset_ids_map_dict_list, dataset_indices_list = [], []
    for dataset_path in cfg['DATA']['DATASET_DIR']:
        dataset_ids_map_dict, dataset_max_idx = get_ids_map(dataset_path)   # dataset_ids_map_dict: {'file1_name': (start_id, end_id), 'file2_name': (start_id, end_id), ...}
        dataset_indices = np.arange(dataset_max_idx).tolist()   # dataset_indices: [1, 2, ..., sample_total_num]
        dataset_ids_map_dict_list.append(dataset_ids_map_dict)
        dataset_indices_list.append(dataset_indices)
        
    if cfg['IS_DEBUG']:
        num_workers = 0
    else:
        num_workers = cfg['TRAIN']['NUM_WORKERS']
    train_sample_per_gpu = cfg['TRAIN']['BATCH_SIZE_PER_GPU']
    collate_fn = torch.utils.data.dataloader.default_collate
    dataset_list = []
    for i, dataset_dir in enumerate(cfg['DATA']['DATASET_DIR']):
        if cfg['DATA']['MAIN_MODALITY'] == 'image':
            train_dataset = NavImageDataset(cfg = cfg, dataset_dir = dataset_dir, ids_map_dict = dataset_ids_map_dict_list[i], indices = dataset_indices_list[i], is_train = True)
            collate_fn = NavImage_collate_fn
        elif cfg['DATA']['MAIN_MODALITY'] == 'point':
            raise NotImplementedError
        else:
            raise NotImplementedError
        dataset_list.append(train_dataset)

    train_dataset = CustomConcatDataset(dataset_list)
    
    if comm.is_main_process():
        logger = logging.getLogger("GEN")
        logger.info("Training set sample number: {}.".format(len(train_dataset)))
    
    if cfg['TRAIN']['DATA_SAMPLE_MODE'] == 'random':
        train_sampler = distributed_sampler.InfiniteTrainingSampler(len(train_dataset))
    elif cfg['TRAIN']['DATA_SAMPLE_MODE'] == 'balance':
        train_sampler = distributed_sampler.InfiniteBalanceTrainingSampler(sub_cumulative_sizes = train_dataset.cumulative_sizes)
    train_batch_sampler = torch.utils.data.sampler.BatchSampler(train_sampler, train_sample_per_gpu, drop_last=True)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    
    return train_dataloader, None

def get_ids_map(dataset_dir):
    data_file_list = get_hdf5_list(os.path.join(dataset_dir, 'h5py'))
    idx_start = 0
    ids_map = {}
    for idx, data_file in enumerate(data_file_list):
        with h5py.File(os.path.join(dataset_dir, 'h5py', data_file), 'r') as root:
            sample_num = len(root['samples'].keys())
            ids_map[data_file] = (idx_start, idx_start + sample_num - 1)
            idx_start += sample_num
    return ids_map, idx_start

def get_sequence_indices(ids_map_dict, chunk_size):
    indices = []
    for key, value in ids_map_dict.items():
        start, end = value
        indices += [ele for ele in range(start, end, chunk_size)]
    return indices

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

