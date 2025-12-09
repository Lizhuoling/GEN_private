import pdb
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from GEN.utils.detr.models.GEN_navdp_model import get_GEN_navdp_model

class GEN_navdp_Policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = get_GEN_navdp_model(cfg)
        self.model = model.cuda()
        
        self.l1_loss_weight = 5.0

    def __call__(self, data):
        if 'batch_labels' in data.keys(): # training or validation time
            means, variances, mixture_weights = self.model(data, is_train = True)
            action_label = data['batch_labels'].cuda() # Left shape: (B, chunk_size, state_dim)
            batch_labels_invalid = data['batch_labels_invalid'].cuda()    # Left shape: (B, chunk_size)
            loss_dict = dict()
            
            # means shape: (num_dec, B, chunk_size, num_mixture, state_dim)
            # variances shape: (num_dec, B, chunk_size, num_mixture, state_dim)
            # mixture_weights shape: (num_dec, B, chunk_size, num_mixture)
            # ctrl_cmd shape: (B, state_dim)
            diff = action_label[None, :, :, None, :] - means # Left shape: (num_dec, B, chunk_size, num_mixture, state_dim)
            precisions = 1.0 / (variances + 1e-8) # Left shape: (num_dec, B, chunk_size, num_mixture, state_dim)
            exp_term = -0.5 * (diff**2) * precisions    # Left shape: (num_dec, B, chunk_size, num_mixture, state_dim)
            norm_term = 0.5 * (torch.log(precisions + 1e-8) - math.log(2 * torch.pi)).sum(dim=-1)   # Left shape: (num_dec, B, chunk_size, num_mixture)
            log_probs = exp_term.sum(dim=-1) + norm_term  # Left shape: (num_dec, B, chunk_size, num_mixture)
            weighted_log_probs = log_probs + torch.log(mixture_weights + 1e-8)  # Left shape: (num_dec, B, chunk_size, num_mixture)
            log_likelihood = torch.logsumexp(weighted_log_probs, dim=-1)    # Left shape: (num_dec, B, chunk_size)
            likehood_loss = (-log_likelihood * (~batch_labels_invalid[None])).sum(-1) / torch.clamp((~batch_labels_invalid[None]).sum(-1), min = 1) # Left shape: (num_dec, B)
            avg_likehood_loss = likehood_loss.mean(dim = -1)    # Left shape: (num_dec,)
            total_loss = avg_likehood_loss.sum()
            for dec_id in range(avg_likehood_loss.shape[0]):
                loss_dict[f'dec_{dec_id}'] = avg_likehood_loss[dec_id].item()
            loss_dict['total_loss'] = total_loss.item()
            
            # Real loss items for recording
            last_dec_diff = torch.abs(diff[-1]).sum(-1).detach() # Left shape: (B, chunk_size, num_mixture)
            _, k_indices = torch.max(mixture_weights[-1], dim=-1)   # k_indices shape: (B, chunk_size)
            absolute_diff = torch.gather(last_dec_diff, 1, k_indices.unsqueeze(2)).squeeze(-1)  # Left shape: (B, chunk_size)
            batch_real_loss = ((~batch_labels_invalid) * absolute_diff).sum(1) / torch.clamp((~batch_labels_invalid).sum(1), min = 1)   # Left shape: (B,)
            batch_real_loss = torch.nan_to_num(batch_real_loss, nan=0.0)
            real_loss = batch_real_loss.mean()
            loss_dict['real_loss'] = real_loss.item()
            return total_loss, loss_dict
        else: # inference time
            means, variances, mixture_weights = self.model(padded_global_plan, padded_global_plan_mask, envi_obs, is_train = False)
            _, k_indices = torch.max(mixture_weights, dim=-1)  # k_indices shape: (B, chunk_size)
            B, chunk_size, num_mixture, action_dim = means.shape
            indices = k_indices[:, :, None, None].expand(-1, -1, 1, action_dim)   # Left shape: (B, chunk_size, 1, action_dim)
            a_hat = torch.gather(means, 2, indices).squeeze(2)  # means shape: (B, chunk_size, num_mixture, state_dim), a_hat shape: (B, chunk_size, state_dim)
            return a_hat
