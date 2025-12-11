import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import open3d as o3d
import einops
import pdb
import sonata
from sonata.model import PointTransformerV3

class Sonata(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        level_feat_ch = [512, 384, 192]
        self.use_level_num = 3
        self.num_features = 0
        for i in range(self.use_level_num):
            self.num_features += level_feat_ch[i]
            
        # Load sonata
        ckpt = torch.load(self.cfg['POLICY']['BACKBONE_PATH'])
        ckpt['config']['enc_patch_size'] = (1024, 1024, 1024, 1024, 1024)
        ckpt['config']['enable_flash'] = True
        self.sonata = PointTransformerV3(**ckpt["config"])
        self.sonata.load_state_dict(ckpt["state_dict"])
        # Data preproceess
        self.transform = sonata.transform.GEN_navdp(cfg)
        
    def forward(self, rgb_obs, depth_obs, intrinsic):
        '''
        Input:
            rgb_obs shape: (B, H, W, 3)
            depth_obs shape: (B, H, W, 1)
            intrinsic shape: (B, 3, 3)
        '''
        B, H, W, _ = rgb_obs.shape

        # Calculate the point coord
        fx = intrinsic[:, 0, 0].view(B, 1, 1)  # (B, 1, 1)
        fy = intrinsic[:, 1, 1].view(B, 1, 1)  # (B, 1, 1)
        cx = intrinsic[:, 0, 2].view(B, 1, 1)  # (B, 1, 1)
        cy = intrinsic[:, 1, 2].view(B, 1, 1)  # (B, 1, 1)
        u_coords = torch.arange(W, device=depth_obs.device).float()  # (W,)
        v_coords = torch.arange(H, device=depth_obs.device).float()  # (H,)
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")
        u_grid = u_grid.unsqueeze(0).repeat(B, 1, 1)
        v_grid = v_grid.unsqueeze(0).repeat(B, 1, 1)
        Z = depth_obs.squeeze(-1)
        X = (u_grid - cx) * Z / fx  # (B, H, W)
        Y = (v_grid - cy) * Z / fy  # (B, H, W)
        point_coord = torch.stack([X, Y, Z], dim=-1)    # left shape: (B, H, W, 3)
        point_normal = self.normal_from_cross_product_torch(point_coord)    # left shape: (B, H, W, 3)
        point_coord = point_coord.reshape(-1, 3) # Left shape: (BHW, 3)
        point_normal = point_normal.reshape(-1, 3) # Left shape: (BHW, 3)
        point_rgb = rgb_obs.reshape(-1, 3) # Left shape: (BHW, 3)
        batch_ids = torch.arange(B, device=depth_obs.device).view(B, 1).repeat(1, H*W).flatten() # Left shape: (BHW)
        # Filter invalid points
        valid_point_mask = (depth_obs.flatten() > self.cfg['DATA']['VALID_DEPTH_RANGE'][0]) & (depth_obs.flatten() < self.cfg['DATA']['VALID_DEPTH_RANGE'][1])   # Left shape: (BHW,)
        point_coord = point_coord[valid_point_mask] # Left shape: (n, 3)
        point_normal = point_normal[valid_point_mask] # Left shape: (n, 3)
        point_rgb = point_rgb[valid_point_mask] # Left shape: (n, 3)
        batch_ids = batch_ids[valid_point_mask] # Left shape: (n,)
        
        # Check if any batch has no points after filtering, and add a dummy point (0, 0, 0) for those batches
        unique_batches = torch.unique(batch_ids).cpu().numpy() if len(batch_ids) > 0 else np.array([], dtype=np.int64)
        all_batches = np.arange(B)
        missing_batches = np.setdiff1d(all_batches, unique_batches).tolist()
        
        if len(missing_batches) > 0:
            # Add dummy points for missing batches
            device = point_coord.device
            dummy_coord = torch.zeros((len(missing_batches), 3), device=device, dtype=point_coord.dtype)
            dummy_normal = torch.zeros((len(missing_batches), 3), device=device, dtype=point_normal.dtype)
            dummy_rgb = torch.zeros((len(missing_batches), 3), device=device, dtype=point_rgb.dtype)
            dummy_batch_ids = torch.tensor(missing_batches, device=device, dtype=batch_ids.dtype)
            
            point_coord = torch.cat([point_coord, dummy_coord], dim=0)
            point_normal = torch.cat([point_normal, dummy_normal], dim=0)
            point_rgb = torch.cat([point_rgb, dummy_rgb], dim=0)
            batch_ids = torch.cat([batch_ids, dummy_batch_ids], dim=0)

        data = dict(coord=point_coord.cpu().numpy(), color=point_rgb.cpu().numpy(), normal=point_normal.cpu().numpy(), batch=batch_ids.cpu().numpy())
        # Add 'batch' to index_valid_keys so it gets downsampled along with coord, color, normal
        data['index_valid_keys'] = ['coord', 'color', 'normal', 'batch']
        data = self.transform(data)

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
                
        point = self.sonata(data)
        for _ in range(self.use_level_num - 1):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
            
        feature = point.feat    # (new_n, c)
        feature_batch_idxs = point.batch    # (new_n,)
        batch_repr_list = []
        batch_repr_is_pad_list = []
        batch_max_point_num = 0
        for i in range(B):
            batch_feat = feature[feature_batch_idxs == i]
            batch_repr_list.append(batch_feat)
            if batch_feat.shape[0] > batch_max_point_num:
                batch_max_point_num = batch_feat.shape[0]
        for i in range(B):
            pad_feat = torch.zeros((batch_max_point_num - batch_repr_list[i].shape[0], batch_repr_list[i].shape[1]), dtype = torch.float32).cuda()
            repr_is_pad = torch.cat((torch.zeros((batch_repr_list[i].shape[0],), dtype = torch.bool), torch.ones((pad_feat.shape[0],), dtype = torch.bool)), dim = 0).cuda()
            batch_repr_is_pad_list.append(repr_is_pad)
            batch_repr_list[i] = torch.cat((batch_repr_list[i], pad_feat), dim = 0)
        batch_repr = torch.stack(batch_repr_list, dim = 0)    # (bs, max_n, c)
        batch_repr_is_pad = torch.stack(batch_repr_is_pad_list, dim = 0)    # (bs, max_n)

        return batch_repr, batch_repr_is_pad 
    
    def normal_from_cross_product_torch(self, points_coord):
        """
        Input:
            points_coord: torch tensor of shape (B, H, W, 3)
        """
        B, H, W, C = points_coord.shape
        assert C == 3
        pad_xyz = F.pad(points_coord.permute(0, 3, 1, 2), (0, 1, 0, 1), mode='reflect')
        pad_xyz = pad_xyz.permute(0, 2, 3, 1)          # Left shape: (B, H+1, W+1, 3)
        xyz_ver = (pad_xyz[:, :, :-1, :] - pad_xyz[:, :, 1:, :])[:, :-1, :, :]  # (B, H, W, 3)
        xyz_hor = (pad_xyz[:, :-1, :, :] - pad_xyz[:, 1:, :, :])[:, :, :-1, :]  # (B, H, W, 3)
        xyz_normal = torch.cross(xyz_hor, xyz_ver, dim=-1)
        norm = torch.linalg.norm(xyz_normal, dim=-1, keepdim=True)  # (B, H, W, 1)
        mask = norm > 0
        xyz_normal = torch.where(mask, xyz_normal / (norm + 1e-12), torch.zeros_like(xyz_normal))    # Left shape: (B, H, W, 3)
        return xyz_normal

def get_sonata(cfg):
    return Sonata(cfg)