import torch
import torch.nn as nn
import math


# ──────────────────────── Time Embedding ────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional encoding for continuous time t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) or (B,) → (B, dim)"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)  # guard div-by-zero when dim <= 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.view(-1, 1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


# ──────────────────────── Helper Functions ────────────────────────

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Batched Farthest Point Sampling.
    xyz:    (B, N, 3)
    Returns (B, npoint) indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)   # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)             # (B, N)
        improved = dist < distance
        distance[improved] = dist[improved]
        farthest = distance.argmax(-1)
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gathers features by index.
    points: (B, N, C)
    idx:    (B, S) or (B, S, K)
    Returns (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    flat_idx  = idx.reshape(B, -1)                                          # (B, S*K)
    batch_idx = torch.arange(B, device=points.device).unsqueeze(1).expand_as(flat_idx)
    gathered  = points[batch_idx, flat_idx]                                 # (B, S*K, C)
    return gathered.reshape(*idx.shape, points.shape[-1])


def query_ball_point(radius: float, nsample: int,
                     xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    For each centroid in new_xyz, find up to nsample nearest neighbors in xyz within radius.
    Positions with no valid neighbor are filled with the nearest point's index.
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    Returns  (B, S, nsample) indices into xyz
    """
    dists = torch.cdist(new_xyz, xyz)                                       # (B, S, N)
    sorted_dists, sorted_idx = dists.sort(dim=-1)                           # (B, S, N)
    group_idx   = sorted_idx[:, :, :nsample].clone()                        # (B, S, nsample)
    group_dists = sorted_dists[:, :, :nsample]                              # (B, S, nsample)
    # Replace out-of-ball entries with the nearest neighbor (sorted_idx[:,:,0])
    invalid = group_dists > radius
    group_idx[invalid] = group_idx[:, :, 0:1].expand_as(group_idx)[invalid]
    return group_idx


# ──────────────────────── Set Abstraction ────────────────────────

class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction: FPS → ball query → grouped MLP → max-pool.
    in_channel: feature channels EXCLUDING xyz (xyz is always prepended inside groups).
    """
    def __init__(self, npoint, radius, nsample, in_channel: int, mlp_channels: list,
                 group_all: bool = False):
        super().__init__()
        self.npoint    = npoint
        self.radius    = radius
        self.nsample   = nsample
        self.group_all = group_all

        layers  = []
        last_ch = in_channel + 3   # relative xyz is concatenated to every grouped point
        for out_ch in mlp_channels:
            layers += [nn.Conv2d(last_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(True)]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_ch

    def forward(self, xyz: torch.Tensor, points):
        """
        xyz:    (B, N, 3)
        points: (B, N, C) or None
        Returns:
            new_xyz:    (B, npoint, 3)       [or (B, 1, 3) if group_all]
            new_points: (B, npoint, mlp[-1]) [or (B, 1, mlp[-1])]
        """
        if self.group_all:
            B, _, _ = xyz.shape
            new_xyz     = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)                                  # (B, 1, N, 3)
            if points is not None:
                grouped_pts = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)  # (B,1,N,3+C)
            else:
                grouped_pts = grouped_xyz
        else:
            fps_idx     = farthest_point_sample(xyz, self.npoint)           # (B, npoint)
            new_xyz     = index_points(xyz, fps_idx)                        # (B, npoint, 3)
            ball_idx    = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B,S,nsample)
            grouped_xyz = index_points(xyz, ball_idx)                       # (B, S, nsample, 3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)                # relative coords
            if points is not None:
                grouped_pts = index_points(points, ball_idx)                # (B, S, nsample, C)
                grouped_pts = torch.cat([grouped_xyz, grouped_pts], dim=-1) # (B, S, nsample, 3+C)
            else:
                grouped_pts = grouped_xyz                                   # (B, S, nsample, 3)

        # Conv2d shared MLP then max-pool over the sample dimension
        grouped_pts = grouped_pts.permute(0, 3, 2, 1)                      # (B, 3+C, nsample, S)
        new_points  = self.mlp(grouped_pts)                                 # (B, mlp[-1], nsample, S)
        new_points  = new_points.max(dim=2)[0]                              # (B, mlp[-1], S)
        new_points  = new_points.permute(0, 2, 1)                           # (B, S, mlp[-1])
        return new_xyz, new_points


# ──────────────────────── Feature Propagation ────────────────────────

class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation: IDW interpolation (k=3) + skip connection + shared MLP.
    in_channel: concatenated channels of skip features + interpolated features.
    """
    def __init__(self, in_channel: int, mlp_channels: list):
        super().__init__()
        layers  = []
        last_ch = in_channel
        for out_ch in mlp_channels:
            layers += [nn.Conv1d(last_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(True)]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1, points2: torch.Tensor) -> torch.Tensor:
        """
        xyz1:    (B, N, 3)  target resolution (higher)
        xyz2:    (B, S, 3)  source resolution (lower)
        points1: (B, N, C1) skip-connection features, or None
        points2: (B, S, C2) features to upsample
        Returns: (B, N, mlp[-1])
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.expand(B, N, -1)                        # (B, N, C2)
        else:
            dists, knn_idx = torch.cdist(xyz1, xyz2).topk(3, dim=-1, largest=False)  # (B,N,3) each
            inv_d    = 1.0 / (dists + 1e-8)                                # (B, N, 3)
            weights  = inv_d / inv_d.sum(dim=-1, keepdim=True)             # (B, N, 3) normalized
            knn_feat = index_points(points2, knn_idx)                      # (B, N, 3, C2)
            interpolated = (weights.unsqueeze(-1) * knn_feat).sum(dim=2)   # (B, N, C2)

        fused = torch.cat([points1, interpolated], dim=-1) if points1 is not None \
                else interpolated                                            # (B, N, C1+C2)

        out = self.mlp(fused.permute(0, 2, 1))                             # (B, mlp[-1], N)
        return out.permute(0, 2, 1)                                         # (B, N, mlp[-1])


# ──────────────────────── PointNet++ Backbone ────────────────────────

class PointNet2(nn.Module):
    """
    PointNet++ encoder-decoder for per-point feature extraction.
    Architecture: SA1 → SA2 → SA3(global) → FP3 → FP2 → FP1.
    Input first 3 channels must be XYZ; remaining are treated as additional features.
    Output: same point count as input, with out_dim channels per point.

    radii / nsamples: tune to match point cloud scale (physical units or normalized).
    """
    def __init__(self, in_dim: int, out_dim: int = 256,
                 npoints:  tuple = (512, 128),
                 radii:    tuple = (0.1, 0.2),
                 nsamples: tuple = (32, 64)):
        super().__init__()
        feat_dim = in_dim - 3   # additional feature channels beyond xyz

        # Encoder
        self.sa1 = PointNetSetAbstraction(npoints[0], radii[0], nsamples[0],
                                          feat_dim, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoints[1], radii[1], nsamples[1],
                                          128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None,
                                          256, [256, 512, 1024], group_all=True)

        # Decoder
        self.fp3 = PointNetFeaturePropagation(1024 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256  + 128, [256, 128])
        fp1_in   = 128 + feat_dim if feat_dim > 0 else 128
        self.fp1 = PointNetFeaturePropagation(fp1_in,     [128, 128, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim)  — first 3 dims are xyz
        Returns: (B, N, out_dim)
        """
        xyz    = x[:, :, :3]                                               # (B, N, 3)
        points = x[:, :, 3:] if x.shape[-1] > 3 else None                 # (B, N, feat_dim)

        # Encoder
        l1_xyz, l1_pts = self.sa1(xyz,    points)  # (B, 512, 3),  (B, 512, 128)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)  # (B, 128, 3),  (B, 128, 256)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)  # (B, 1,   3),  (B, 1,   1024)

        # Decoder (skip connections from each encoder level)
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)  # (B, 128, 256)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)  # (B, 512, 128)
        l0_pts = self.fp1(xyz,    l1_xyz, points,  l1_pts)  # (B, N,   out_dim)

        return l0_pts
