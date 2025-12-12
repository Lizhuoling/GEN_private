import torch
import torch.nn as nn
import torch.nn.functional as F

class MapBackbone(nn.Module):
    def __init__(self, num_one_step_observations=10, scandots_output_dim=64, num_frames=1) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.encoder = nn.Sequential(
            # [1, 100, 100]
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            # [32, 96, 96]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 48, 48]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            activation,
            # [64, 23, 23]
            nn.Flatten(),
            nn.Linear(64 * 23 * 23, 256),
            activation,
            nn.Linear(256, scandots_output_dim),
            activation
        )
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(scandots_output_dim + num_one_step_observations, 128),
                                    activation,
                                    nn.Linear(128, 128)
                                )
        self.output_mlp = nn.Sequential(
                                nn.Linear(128, scandots_output_dim),
                                last_activation
                            )

    def forward(self, obs, hmap):
        hmap_feature = self.encoder(hmap)
        hmap_latent = self.combination_mlp(torch.cat((hmap_feature, obs), dim=-1))
        hmap_latent = self.output_mlp(hmap_latent)
        return hmap_latent

class LidarBackbone(nn.Module):
    def __init__(self, input_dim=50*50, num_one_step_observations=10, scandots_output_dim=64) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.encoder_mlp = nn.Sequential(
                                    nn.Linear(input_dim, 1250),
                                    activation,
                                    nn.Linear(1250, 640),
                                    activation,
                                    nn.Linear(640, 256),
                                    activation,
                                    nn.Linear(256, 64),
                                )
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(scandots_output_dim + num_one_step_observations, 128),
                                    activation,
                                    nn.Linear(128, 128)
                                )
        self.output_mlp = nn.Sequential(
                                nn.Linear(128, scandots_output_dim),
                                last_activation
                            )
        
    def forward(self, obs, hmap):
        hmap_feature =  self.encoder_mlp(hmap)
        hmap_latent = self.combination_mlp(torch.cat((hmap_feature, obs), dim=-1))
        hmap_latent = self.output_mlp(hmap_latent)
        return hmap_latent

class TNetTiny(nn.Module):

    def __init__(self, input_dim=3, output_dim=64):
        super(TNetTiny, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=64,  kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=self.output_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        x = x.transpose(2, 1)  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)

        return x

if __name__ == "__main__":
    bk = LidarBackbone()
    hmap = torch.rand(10, 2500)
    obs = torch.rand(10, 10)
    bk(obs, hmap)

