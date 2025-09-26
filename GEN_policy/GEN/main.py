import numpy as np
import os
import pdb
import time
import pickle
import datetime
import logging
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_utils import load_data
from utils.dataset_utils import compute_dict_mean
from utils.utils import set_seed
from utils.engine import launch
from utils import comm
from utils.optimizer import make_optimizer, make_scheduler
from utils.metric_logger import MetricLogger
from utils.logger import setup_logger
from configs.utils import load_yaml_with_base
from utils.model_zoo import make_policy, load_policy
from utils.inference.isaac_navigation import IsaacNavEnviManager

def main(args):
    # Initialize logger
    if comm.get_rank() == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    exp_start_time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    rank = comm.get_rank()
    logger = setup_logger(args.save_dir, rank, file_name="log_{}.txt".format(exp_start_time))
    if comm.is_main_process():
        logger.info("Using {} GPUs".format(comm.get_world_size()))
        logger.info("Collecting environment info")
        logger.info(args) 
        logger.info("Loaded configuration file {}".format(args.config_name+'.yaml'))
    
    # Initialize cfg
    cfg = load_yaml_with_base(os.path.join('configs', args.config_name+'.yaml'))
    cfg['IS_EVAL'] = args.eval
    cfg['CKPT_DIR'] = args.save_dir
    cfg['IS_DEBUG'] = args.debug
    cfg['NUM_NODES'] = args.num_nodes
    
    if cfg['SEED'] >= 0:
        set_seed(cfg['SEED'])

    if cfg['IS_EVAL']:
        if args.load_dir != '':
            ckpt_paths = [args.load_dir]
        else:
            ckpt_paths = [os.path.join(cfg['CKPT_DIR'], 'policy_latest.ckpt')]
        results = []
        for ckpt_path in ckpt_paths:
            eval_bc(cfg, ckpt_path)
        logger.info("Evaluation completed!")
        exit()
    
    train_dataloader, val_dataloader = load_data(cfg)
    
    train_bc(train_dataloader, val_dataloader, cfg, load_dir = args.load_dir, load_pretrain = args.load_pretrain)

def eval_bc(cfg, ckpt_path): 
    ckpt_dir = cfg['CKPT_DIR']
    ckpt_name = ckpt_path.split('/')[-1]
    policy_class = cfg['POLICY']['POLICY_NAME']
    
    policy = load_policy(ckpt_path, policy_class, cfg)
    envi_manager = IsaacNavEnviManager(cfg, policy)
    envi_manager.inference()

def forward_pass(data, policy, cfg, iter_cnt):
    if cfg['DATA']['MAIN_MODALITY'] == 'image':
        ctrl_cmd, padded_global_plan, padded_global_plan_mask, image_array = data
        ctrl_cmd, padded_global_plan, padded_global_plan_mask, envi_obs = ctrl_cmd.cuda(), padded_global_plan.cuda(), padded_global_plan_mask.cuda(), image_array.cuda()
        envi_obs = envi_obs.permute(0, 1, 4, 2, 3)
        return policy(ctrl_cmd, padded_global_plan, padded_global_plan_mask, envi_obs)
    else:
        raise NotImplementedError

def train_bc(train_dataloader, val_dataloader, cfg, load_dir = '', load_pretrain = ''):
    logger = logging.getLogger("GEN")

    num_iterations = cfg['TRAIN']['NUM_ITERATIONS']
    ckpt_dir = cfg['CKPT_DIR']
    seed = cfg['SEED']
    policy_name = cfg['POLICY']['POLICY_NAME']

    policy = make_policy(policy_name, cfg)
    if load_pretrain != '':
        load_dict = torch.load(load_pretrain)['model']
        filter_dict = {key:value for key, value in load_dict.items() if 'model.backbone' in key or 'model.transformer' in key}
        loading_status = policy.load_state_dict(filter_dict, strict = False)
    if load_dir != '':
        load_dict = torch.load(load_dir)
        loading_status = policy.load_state_dict(load_dict['model'], strict = True)
        start_iter = load_dict['iter']
    else:
        start_iter = 0
    policy.cuda()
    optimizer = make_optimizer(policy, cfg)
    scheduler, warmup_scheduler = make_scheduler(optimizer, cfg=cfg)
    
    main_thread = comm.get_rank() == 0
    if main_thread:
        tb_writer = SummaryWriter(os.path.join(ckpt_dir, 'tb/{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))

    if cfg['TRAIN']['LR_WARMUP']:
        assert warmup_scheduler is not None
        warmup_iters = cfg['TRAIN']['WARMUP_STEPS']
    else:
        warmup_iters = -1

    min_val_loss = np.inf
    end = time.time()
    train_meters = MetricLogger(delimiter=", ", )
    
    for data, iter_cnt in zip(train_dataloader, range(start_iter, num_iterations)):
        data_time = time.time() - end

        # training
        policy.train()
        optimizer.zero_grad()
        total_loss, loss_dict = forward_pass(data, policy, cfg, iter_cnt)
        # backward
        total_loss.backward()
        optimizer.step()
        if iter_cnt < warmup_iters:
            warmup_scheduler.step(iter_cnt)
        else:
            scheduler.step(iter_cnt)

        train_meters.update(**loss_dict)
        batch_time = time.time() - end
        end = time.time()
        train_meters.update(time=batch_time, data=data_time)
        eta_seconds = train_meters.time.global_avg * (num_iterations - iter_cnt)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # log
        if main_thread and (iter_cnt % cfg['TRAIN']['LOG_INTERVAL'] == 0 or iter_cnt == num_iterations - 1):
            logger.info(
                train_meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f} \n",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iter_cnt,
                    meters=str(train_meters),
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            
            tb_writer.add_scalar('loss/total_loss', total_loss.item(), iter_cnt)
            for loss_key in loss_dict.keys():
                tb_writer.add_scalar(f'loss/{loss_key}', loss_dict[loss_key], iter_cnt)
            tb_writer.add_scalar('state/lr', optimizer.param_groups[0]["lr"], iter_cnt)

        # validation
        if cfg['EVAL']['DATA_EVAL_RATIO'] > 0 and main_thread and iter_cnt % cfg['EVAL']['EVAL_INTERVAL'] == 0 and iter_cnt != 0:
            logger.info("Start evaluation at iteration {}...".format(iter_cnt))
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []

                eval_total_iter_num = len(val_dataloader)
                if eval_total_iter_num > cfg['EVAL']['MAX_VAL_SAMPLE_NUM']:
                    eval_total_iter_num = cfg['EVAL']['MAX_VAL_SAMPLE_NUM']

                for eval_data, eval_iter_cnt in zip(val_dataloader, range(eval_total_iter_num)):
                    total_loss, loss_dict = forward_pass(eval_data, policy, cfg)
                    epoch_dicts.append(loss_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (iter_cnt, min_val_loss, deepcopy(policy.state_dict()))
            summary_string = 'Evaluation result:'
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            logger.info(summary_string)

        # Save checkpoint
        if main_thread and iter_cnt % cfg['TRAIN']['SAVE_CHECKPOINT_INTERVAL'] == 0 and iter_cnt != 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_latest.ckpt')
            if os.path.exists(ckpt_path):
                os.rename(ckpt_path, os.path.join(ckpt_dir, f'policy_previous.ckpt'))
            save_model(policy, ckpt_path, iter_cnt)
            
    if main_thread:
        ckpt_path = os.path.join(ckpt_dir, f'policy_latest.ckpt')
        save_model(policy, ckpt_path, iter_cnt)
        logger.info(f'Training finished!')

    comm.synchronize()
    if main_thread:
        tb_writer.close()
    for handler in logging.root.handlers:
        handler.close()

def save_model(model, save_path, iter_cnt):
    model_ckpt = model.state_dict()
    save_dict = {
        'iter': iter_cnt,
        'model': model_ckpt
    }
    torch.save(save_dict, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', action='store', type=str, help='configuration file name', required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='saving directory', required=True)
    parser.add_argument('--load_dir', action='store', type=str, default = '', help='The path to weight',)
    parser.add_argument('--load_pretrain', action='store', type=str, default = '', help='The path to pre-trained weight')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_nodes', default = 1, type = int, help = "The number of nodes.")
    args = parser.parse_args()
    
    launch(main, args)
