# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import pdb
import copy
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

class GEN_model(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, state_dim, chunk_size, camera_names, cfg):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.num_mixture = cfg['POLICY']['MIXTURE_GAUSSIAN_NUM']
        self.output_dim = self.num_mixture * (2 * state_dim + 1)
        self.camera_names = camera_names
        
        if len(self.camera_names) > 0:
            self.backbone = backbone
            self.obs_embed = nn.Embedding(len(cfg['DATA']['CAMERA_NAMES']), hidden_dim) 
            self.input_proj = nn.Linear(self.backbone.num_features, hidden_dim)
            
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.global_plan_proj = nn.Linear(7, hidden_dim)
        self.global_plan_embed = nn.Embedding(self.cfg['DATA']['GLOBAL_PLAN_LENGTH'], hidden_dim)
        
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.output_dim) # Decode transformer output as action.

    def forward(self, padded_global_plan, padded_global_plan_mask, envi_obs, is_train = False):
        """
        padded_global_plan: (batch, max_plan_len, 7), float32
        padded_global_plan_mask: (batch, max_plan_len), bool
        envi_obs: (batch, N, H, W, 3), float32
        is_train: bool
        """
        bs = padded_global_plan.shape[0]
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # Left shape: (num_query, B, C)
        
        if len(self.camera_names) > 0:
            if self.cfg['TRAIN']['LR_BACKBONE'] > 0:
                feature, feature_is_pad = self.backbone(envi_obs)   # feature shape: (B, L, C), feature_is_pad shape: (B, L)
            else:
                with torch.no_grad():
                    feature, feature_is_pad = self.backbone(envi_obs)
                    feature, feature_is_pad = feature.detach(), feature_is_pad.detach()
            src = self.input_proj(feature).permute(1, 0, 2)  # Left shape: (L, B, C)
            obs_embed = self.obs_embed.weight[:, None, None]   # Left shape: (num_cam, 1, 1, C)
            pos = obs_embed.expand(-1, src.shape[0] // obs_embed.shape[0], bs, -1).reshape(-1, bs, self.hidden_dim)   # Left shape: (L, B, C)
            mask = feature_is_pad.clone()   # Left shape: (B, L)
        else:
            src = torch.zeros((0, bs, self.hidden_dim)).to(envi_obs.device)
            pos = torch.zeros((0, bs, self.hidden_dim)).to(envi_obs.device)
            mask = torch.zeros((bs, 0)).to(envi_obs.device)
        
        global_plan_src = self.global_plan_proj(padded_global_plan).permute(1, 0, 2)   # Left shape: (max_plan_len, B, C)
        global_plan_embed = self.global_plan_embed.weight   # Left shape: (max_plan_len, C)
        global_plan_embed = global_plan_embed[:, None].expand(-1, bs, -1)   # Left shape: (max_plan_len, B, C)
        src = torch.cat([src, global_plan_src], dim=0)   # Left shape: (L+1+max_plan_len, B, C)
        pos = torch.cat([pos, global_plan_embed], dim=0)   # Left shape: (L+1+max_plan_len, B, C)
        mask = torch.cat([mask, padded_global_plan_mask], dim=1)   # Left shape: (B, L+1+max_plan_len)
    
        hs = self.transformer(src, mask, query_emb, pos) # Left shape: (num_dec, B, num_query, C)

        output = self.action_head(hs)    # left shape: (num_dec, B, num_query, output_dim)
        means, vars_log, pi_logits = torch.split(output, [self.num_mixture * self.state_dim, self.num_mixture * self.state_dim, self.num_mixture], dim=-1)  # means shape: (num_dec, B, num_query, num_mixture * state_dim)
        means = means.view(means.shape[0], means.shape[1], means.shape[2], self.num_mixture, self.state_dim)  # means shape: (num_dec, B, num_query, num_mixture, state_dim)
        variances = torch.exp(vars_log).view(vars_log.shape[0], vars_log.shape[1], vars_log.shape[2], self.num_mixture, self.state_dim) # variances shape: (num_dec, B, num_query, num_mixture, state_dim)
        mixture_weights = F.softmax(pi_logits, dim=-1)  # mixture_weights shape: (num_dec, B, num_query, num_mixture)
        
        if not is_train:
            means = means[-1]   # Left shape: (B, num_query, num_mixture, state_dim)
            variances = variances[-1]   # Left shape: (B, num_query, num_mixture, state_dim)
            mixture_weights = mixture_weights[-1]  # mixture_weights shape: (B, num_query, num_mixture)

        return means, variances, mixture_weights
    
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(cfg):
    d_model = cfg['POLICY']['HIDDEN_DIM'] # 256
    dropout = cfg['POLICY']['DROPOUT'] # 0.1
    nhead = cfg['POLICY']['NHEADS'] # 8
    dim_feedforward = cfg['POLICY']['DIM_FEEDFORWARD'] # 2048
    num_encoder_layers = cfg['POLICY']['ENC_LAYERS']
    normalize_before = False # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def get_GEN_model(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = GEN_model(
        backbone,
        transformer,
        state_dim=cfg['POLICY']['STATE_DIM'],
        chunk_size=cfg['POLICY']['CHUNK_SIZE'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
