import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Vectorized Farthest Point Sampling (FPS) implementation in pure PyTorch.
    Args:
        xyz: (N, 3) tensor
        npoint: number of points to sample
    Returns:
        indices: (npoint,) tensor of selected indices
    """
    device = xyz.device
    N, C = xyz.shape
    
    # Initialize centroids and distance tracking
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    
    # Randomly pick the first point to start
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        # Compute squared Euclidean distance from the new centroid to all points
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distance if the new distance is smaller than existing distance
        mask = dist < distance
        distance[mask] = dist[mask]
        # Pick the point that is farthest from all currently selected centroids
        farthest = torch.max(distance, -1)[1]
        
    return centroids

def pad_or_truncate_point_cloud(points: torch.Tensor, target_size: int, features: torch.Tensor = None):
    """
    强行将点云对齐到 target_size。自带防崩溃安全校验。
    """
    N = points.shape[0]
    
    # [安全补丁] 如果该 Patch 完全没有点 (N=0)
    if N == 0:
        dummy_points = torch.zeros((target_size, 3), dtype=points.dtype, device=points.device)
        if features is not None:
            # 假设特征的最后一维是 D
            D = features.shape[-1]
            dummy_features = torch.zeros((target_size, D), dtype=features.dtype, device=features.device)
            return dummy_points, dummy_features
        return dummy_points

    if N == target_size:
        if features is not None:
            return points, features
        return points

    if N > target_size:
        # 策略 A: Farthest Point Sampling (FPS) 下采样
        # 相比于随机下采样，FPS 能够更好地保留几何骨架
        indices = farthest_point_sample(points, target_size)
    else:
        # 策略 B: 随机有放回上采样
        indices = torch.randint(0, N, (target_size,))
        
    sampled_points = points[indices]
    
    if features is not None:
        sampled_features = features[indices]
        return sampled_points, sampled_features
        
    return sampled_points

class UnderwaterPatchDataset(Dataset):
    """
    加载点云数据的核心 Dataset
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data_dict = torch.load(file_path, weights_only=True)
        
        x_raw = data_dict['noisy_points']
        features = data_dict['features']
        x_gt = data_dict['gt_points']
        
        # 动态重采样 (超参数定死：输入 512，真值 4096)
        x_raw_fixed, features_fixed = pad_or_truncate_point_cloud(x_raw, target_size=512, features=features)
        x_gt_fixed = pad_or_truncate_point_cloud(x_gt, target_size=4096)
        
        return {
            'noisy_points': x_raw_fixed,
            'features': features_fixed,
            'gt_points': x_gt_fixed
        }