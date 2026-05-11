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


def _compute_unit_diag_normalize(gt_pts: torch.Tensor):
    """Compute per-sample (center, scale) so that gt fits in a unit-radius ball.

    Args:
        gt_pts: (K, 3) float tensor, used as the *reference* geometry.
            We normalize against GT (not the noisy SfM cloud) because GT is
            clean and densely covers the surface — its bounding box is a
            stable definition of "the scene". Both noisy and gt are then
            transformed by the same affine to stay co-registered.

    Returns:
        center: (3,) float tensor — the AABB midpoint of GT
        scale:  scalar tensor — half the AABB diagonal length

    Rationale: cad_to_gt.py already does an AABB normalize with
    scale = max(|coord|), which leaves long objects (pipes) with diag≈2 on
    the long axis but ≈0.4 on short axes. That anisotropy made the
    densifier's isotropic ε-ball over-perturb thin directions, which is the
    proximate cause of the "short-axis fattening, long-axis collapse"
    failure mode visible in benchmark.

    Using diag/2 as scale forces every scene into a *unit-radius bounding
    sphere*, so the model sees objects of the same relative size regardless
    of category. This makes the densifier's ε meaningful as a fraction of
    object size, and it makes per-scene CD comparable across categories.
    """
    aabb_min = gt_pts.amin(dim=0)                               # (3,)
    aabb_max = gt_pts.amax(dim=0)                               # (3,)
    center = (aabb_min + aabb_max) / 2.0                        # (3,)
    diag = torch.linalg.norm(aabb_max - aabb_min)               # ()
    # half-diag so the points sit inside a unit sphere; clamp for degenerate
    # scenes (single-point clouds, etc.) to avoid div-by-zero.
    scale = torch.clamp(diag / 2.0, min=1e-6)
    return center, scale


class UnderwaterPatchDataset(Dataset):
    """加载点云数据的核心 Dataset.

    Two design changes vs the legacy version:

    1. **Point-count alignment.** Noisy target is now 1024 (was 512). The
       densifier multiplies by k=4, so x_0 ends up at 4096 — same as the GT
       target. With the old 512→2048 path, every prediction had half the
       point count of GT, capping F-Score recall at ~50% regardless of how
       good the model was. Aligning the counts lets OT and downstream metrics
       see a fair comparison.

    2. **Per-sample unit-sphere normalize.** Even after cad_to_gt's AABB
       normalize, pipes have anisotropic bboxes (long axis ≈ 2, short axis
       ≈ 0.4). The flow-matching ε-ball and learned velocity field implicitly
       assume isotropic scale; the anisotropy was the proximate cause of the
       "short-axis fattening, long-axis collapse" pathology in benchmark.
       We compute (center, scale=diag/2) from GT and apply the same affine
       to both noisy and gt, so every scene is in a unit-radius sphere
       during training and inference. The transform is returned as
       `normalize_runtime` so stage_infer can de-normalize predictions back
       to the original frame for ply export.
    """

    # Target point counts. NOISY_TARGET × densifier.k must equal GT_TARGET
    # so that x_0 and x_gt have matching cardinality at OT time. With
    # densifier k=4 (see models/cufm_net.py), 1024 × 4 = 4096.
    NOISY_TARGET = 1024
    GT_TARGET = 4096

    def __init__(self, data_dir: str, normalize_per_sample: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.normalize_per_sample = normalize_per_sample
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

        # Per-sample isotropic normalize (see docstring). We must compute
        # this *before* resampling, otherwise the FPS picks change the AABB
        # subtly and noisy/gt drift apart.
        if self.normalize_per_sample:
            center, scale = _compute_unit_diag_normalize(x_gt)
            x_raw = (x_raw - center) / scale
            x_gt = (x_gt - center) / scale
        else:
            # Identity transform, but keep the keys present so stage_infer
            # doesn't have to branch on existence.
            center = torch.zeros(3, dtype=x_gt.dtype)
            scale = torch.ones((), dtype=x_gt.dtype)

        # 动态重采样 — 1024 noisy / 4096 gt so densifier×4 matches gt count.
        x_raw_fixed, features_fixed = pad_or_truncate_point_cloud(
            x_raw, target_size=self.NOISY_TARGET, features=features
        )
        x_gt_fixed = pad_or_truncate_point_cloud(x_gt, target_size=self.GT_TARGET)

        return {
            'noisy_points': x_raw_fixed,
            'features': features_fixed,
            'gt_points': x_gt_fixed,
            # Affine that maps normalized → original frame:
            #     original = normalized * scale + center
            # stage_infer reads these to de-normalize predictions before
            # exporting .ply / running benchmark.
            'normalize_center': center,
            'normalize_scale': scale,
        }
