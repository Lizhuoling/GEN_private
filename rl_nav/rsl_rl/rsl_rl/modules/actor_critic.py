# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from .lidar_backbone import TNetTiny

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        inperception_dim, 
        perception_in_dim,
        perception_out_dim,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        activation = resolve_nn_activation(activation)

        self.a_en_num = perception_out_dim
        self.c_en_num = perception_out_dim
        self.in_obs_num = inperception_dim

        if self.a_en_num is None:
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs
        else:
            mlp_input_dim_a = self.a_en_num + inperception_dim  
            mlp_input_dim_c = self.c_en_num + inperception_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        # lidar encoder
        if self.a_en_num is None:
            self.lidar_encoder = None
            self.critic_lidar_encoder = None
        else:
            self.lidar_encoder = TNetTiny(input_dim=perception_in_dim, output_dim=self.a_en_num)
            self.critic_lidar_encoder = TNetTiny(input_dim=perception_in_dim,  output_dim=self.a_en_num)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Actor LidarEncoder MLP: {self.lidar_encoder}")
        print(f"Critic MLP: {self.critic}")
        print(f"Critic LidarEncoder MLP: {self.critic_lidar_encoder}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):

        if self.lidar_encoder is not None:
            in_obs, out_obs = observations[:, :self.in_obs_num], observations[:, self.in_obs_num:].view(observations.shape[0], -1, 3)
            encoded_out_obs = self.lidar_encoder(out_obs)
            actor_input = torch.cat((in_obs, encoded_out_obs), dim=-1)
        else:
            actor_input = observations

        mean = self.actor(actor_input)

        out1 = 0.7 * torch.tanh(mean[:, 0])
        out2 = 1.0 * torch.tanh(mean[:, 1])
        out3 = 1.0 * math.pi / 2 * torch.tanh(mean[:, 2])
        mean = torch.stack([out1, out2, out3], dim=1)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):

        if self.lidar_encoder is not None:
            in_obs, out_obs = observations[:, :self.in_obs_num], observations[:, self.in_obs_num:].view(observations.shape[0], -1, 3)
            encoded_out_obs = self.lidar_encoder(out_obs)
            actor_input = torch.cat((in_obs, encoded_out_obs), dim=-1)
        else:
            actor_input = observations
    
        actions_mean = self.actor(actor_input)

        out1 = 0.7 * torch.tanh(actions_mean[:, 0])
        out2 = 1.0 * torch.tanh(actions_mean[:, 1])
        out3 = 1.0 * math.pi / 2 * torch.tanh(actions_mean[:, 2])
        actions_mean = torch.stack([out1, out2, out3], dim=1)

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):

        if self.critic_lidar_encoder is not None:
            in_obs, out_obs = critic_observations[:, :self.in_obs_num], critic_observations[:, self.in_obs_num:].view(critic_observations.shape[0], -1, 3)
            encoded_out_obs = self.critic_lidar_encoder(out_obs)
            value_input = torch.cat((in_obs, encoded_out_obs), dim=-1)
        else:
            value_input = critic_observations

        value = self.critic(value_input)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
