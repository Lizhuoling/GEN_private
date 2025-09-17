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
from torch.autograd import Variable
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from torch.nn import functional as F

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VIRT_DecOnly(nn.Module):
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
        self.camera_names = camera_names
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        if self.cfg['POLICY']['STATUS_PREDICT']:
            query_num = 1 + chunk_size
        else:
            query_num = chunk_size
        
        dataset_dir = cfg['DATA']['DATASET_DIR']
        if type(dataset_dir) == str:
            self.skill_num = 1
            self.skill_types = [None,]
        elif type(dataset_dir) == list:
            self.skill_types = list(set([ele[0] for ele in dataset_dir]))
            self.skill_num = len(self.skill_types)
        self.skill_embed = nn.Embedding(self.skill_num, hidden_dim)
        
        self.input_proj = nn.Linear(self.backbone.num_features, hidden_dim)
        self.pc_flag_embed = nn.Embedding(1, hidden_dim) 
        self.query_embed = nn.Embedding(query_num, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, state_dim) # Decode transformer output as action.
        self.past_action_mlp = nn.Linear(state_dim, hidden_dim)  # Past action information encoding
        #self.past_action_len, self.past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
        #self.past_action_pos_enc = PositionalEncoding(d_model = hidden_dim, max_len = self.past_action_len)
        
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            self.uncern_head = nn.Linear(hidden_dim, 1)
        if self.cfg['POLICY']['STATUS_PREDICT']:
            self.status_head = nn.Linear(hidden_dim, self.cfg['POLICY']['STATUS_NUM'])

    def forward(self, repr, past_action, action, past_action_is_pad, action_is_pad, status, task_instruction_list, dataset_type):
        """
        repr: (batch, n_point, feat_len)
        past_action: (batch, past_action_len, action_dim)
        action: (batch, chunk_size, action_dim)
        past_action_is_pad: (batch, past_action_len)\
        action_is_pad: (batch, chunk_size)
        status: (batch,)
        task_instruction_list: A list with the length of batch, each element is a string.
        """
        is_training = action is not None # train or val
        bs = past_action.shape[0]
        
        if self.cfg['TRAIN']['LR_BACKBONE'] > 0:
            feature, repr_is_pad = self.backbone(repr)
        else:
            with torch.no_grad():
                feature, repr_is_pad = self.backbone(repr)
                feature, repr_is_pad = feature.detach(), repr_is_pad.detach()
        src = self.input_proj(feature).permute(1, 0, 2)  # Left shape: (L, B, C)
        pc_flag_embed = self.pc_flag_embed.weight   # Left shape: (1, C)
        pos = pc_flag_embed[None].expand(src.shape[0], bs, -1)  # Left shape: (L, B, C)
        mask = repr_is_pad.clone()   # Left shape: (B, L)

        # past action
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # Left shape: (num_query, B, C)
        past_action_src = self.past_action_mlp(past_action).permute(1, 0, 2)   # (past_action_len, B, C)
        past_action_src = past_action_src[-1:]  # Left shape: (1, B, C)
        query_emb = query_emb + past_action_src # Left shape: (num_query, B, C)
        
        # Skill type
        all_skill_embed = self.skill_embed.weight[None]   # Left shape: (1, skill_num, C)
        if self.skill_num == 1:
            skill_embed = all_skill_embed.expand(-1, bs, -1)  # Left shape: (1, 1, C)
        else:
            skill_idx = [self.skill_types.index(ele) for ele in dataset_type]
            skill_embed = all_skill_embed[:, skill_idx] # Left shape: (1, B, C)
        query_emb = query_emb + skill_embed
    
        hs = self.transformer(src, mask, query_emb, pos) # Left shape: (num_dec, B, num_query, C)
        if self.cfg['POLICY']['STATUS_PREDICT']:
            status_hs = hs[:, :, 0] # Left shape: (num_dec, B, C)
            hs = hs[:, :, 1:]
            status_pred = self.status_head(status_hs)  # left shape: (num_dec, B, num_status)
            if not is_training: status_pred = status_pred[-1].argmax(dim = -1)  # Left shape: (B,)
        else:
            status_pred = None
        
        if not is_training: hs = hs[-1] # Left shape: (B, num_query, C)

        a_hat = self.action_head(hs)    # left shape: (num_dec, B, num_query, action_dim)
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            a_hat_uncern = self.uncern_head(hs) # left shape: (num_dec, B, num_query, 1)
            a_hat_uncern = torch.clamp(a_hat_uncern, min = self.cfg['POLICY']['UNCERTAINTY_RANGE'][0], max = self.cfg['POLICY']['UNCERTAINTY_RANGE'][1])
        else:
            a_hat_uncern = None
        
        return a_hat, a_hat_uncern, status_pred
    
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
    num_encoder_layers = cfg['POLICY']['ENC_LAYERS'] # 4 # TODO shared with VAE decoder
    normalize_before = False # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def get_VIRT_DecOnly_model(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = VIRT_DecOnly(
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe
        
        self.register_buffer('pe', pe)

    def forward(self, action_len):
        return self.pe[-action_len:]
