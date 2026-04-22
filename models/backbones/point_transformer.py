"""
Point Transformer backbone (Zhao et al., ICCV 2021).

Key difference from PointNet++:
  - Uses k-NN (not ball query) for local neighborhood
  - Vector self-attention with per-channel softmax weights
  - Position encoding on relative 3D coordinates feeds into both
    attention weight computation and value aggregation

Same interface as PointNet2: (B, N, in_dim) → (B, N, out_dim)
First 3 input channels must be XYZ coordinates.

Memory note: PointTransformerLayer at full resolution computes
O(B·N²) pairwise distances. For N=2048, B=8 this is ~128 MB —
reduce k_attn or batch size if GPU memory is tight.
"""

import torch
import torch.nn as nn

from .pointnet2 import farthest_point_sample, index_points
from .pointnet2 import SinusoidalTimeEmbedding  # re-exported for callers that import from this module


# ──────────────────────── KNN Helper ────────────────────────

def knn_query(k: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """
    K nearest neighbor query (no radius constraint).
    xyz:       (B, N, 3)  reference points
    query_xyz: (B, S, 3)  query points
    Returns:   (B, S, k)  indices into xyz
    """
    dists = torch.cdist(query_xyz, xyz)                        # (B, S, N)
    _, idx = dists.topk(k, dim=-1, largest=False)              # (B, S, k)
    return idx


# ──────────────────────── Core Attention Layer ────────────────────────

class PointTransformerLayer(nn.Module):
    """
    Point Transformer self-attention layer.

    For each point p_i with feature f_i and k-NN {p_j, f_j}:
      q_i = W_q(f_i),   k_j = W_k(f_j),   v_j = W_v(f_j)
      δ_j = φ(p_i − p_j)                  (position encoding MLP)
      α_j = softmax_k( ρ(q_i − k_j + δ_j) )   (vector attention over k)
      y_i = Σ_j  α_j ⊙ (v_j + δ_j)       (weighted sum with position bias)
      output = LayerNorm(y_i + f_i)        (residual)
    """
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k      = k
        self.w_q    = nn.Linear(dim, dim, bias=False)
        self.w_k    = nn.Linear(dim, dim, bias=False)
        self.w_v    = nn.Linear(dim, dim, bias=False)
        # φ: relative 3D → dim  (position encoding)
        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(True), nn.Linear(dim, dim)
        )
        # ρ: dim → dim  (attention weight MLP, per-channel softmax)
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        xyz:  (B, N, 3)
        feat: (B, N, dim)
        Returns: (B, N, dim)
        """
        _, _, D = feat.shape

        knn_idx  = knn_query(self.k, xyz, xyz)                              # (B, N, k)
        knn_xyz  = index_points(xyz,  knn_idx)                              # (B, N, k, 3)
        knn_feat = index_points(feat, knn_idx)                              # (B, N, k, D)

        q = self.w_q(feat).unsqueeze(2).expand(-1, -1, self.k, -1)         # (B, N, k, D)
        k = self.w_k(knn_feat)                                              # (B, N, k, D)
        v = self.w_v(knn_feat)                                              # (B, N, k, D)

        rel_pos = xyz.unsqueeze(2) - knn_xyz                                # (B, N, k, 3)
        pos_enc = self.pos_enc(rel_pos)                                     # (B, N, k, D)

        # Per-channel softmax over k neighbors (vector attention)
        attn = torch.softmax(self.attn_mlp(q - k + pos_enc), dim=2)        # (B, N, k, D)
        out  = ((v + pos_enc) * attn).sum(dim=2)                            # (B, N, D)

        return self.norm(out + feat)


# ──────────────────────── Transition Down ────────────────────────

class TransitionDown(nn.Module):
    """
    Downsampling: FPS centroids → KNN grouping → shared MLP → max-pool.
    Reduces point count N → npoint while expanding feature dim in_dim → out_dim.
    """
    def __init__(self, npoint: int, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.npoint = npoint
        self.k      = k
        # Shared MLP on grouped (relative_xyz ∥ feature) per point
        self.mlp = nn.Sequential(
            nn.Linear(3 + in_dim, out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """
        xyz:  (B, N, 3)
        feat: (B, N, in_dim)
        Returns:
            new_xyz:  (B, npoint, 3)
            new_feat: (B, npoint, out_dim)
        """
        fps_idx  = farthest_point_sample(xyz, self.npoint)                  # (B, npoint)
        new_xyz  = index_points(xyz,  fps_idx)                              # (B, npoint, 3)
        knn_idx  = knn_query(self.k, xyz, new_xyz)                          # (B, npoint, k)
        grp_xyz  = index_points(xyz,  knn_idx) - new_xyz.unsqueeze(2)       # (B, npoint, k, 3)
        grp_feat = index_points(feat, knn_idx)                              # (B, npoint, k, in_dim)
        grouped  = torch.cat([grp_xyz, grp_feat], dim=-1)                   # (B, npoint, k, 3+in_dim)

        B, S, K, C = grouped.shape
        new_feat = self.mlp(grouped.reshape(B * S * K, C))                  # (B*S*K, out_dim)
        new_feat = new_feat.reshape(B, S, K, -1).max(dim=2)[0]             # (B, npoint, out_dim)
        return new_xyz, self.norm(new_feat)


# ──────────────────────── Transition Up ────────────────────────

class TransitionUp(nn.Module):
    """
    Upsampling: IDW interpolation (k=3) → skip connection → MLP.
    Mirrors PointNet++ Feature Propagation adapted for Point Transformer.
    """
    def __init__(self, low_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(low_dim + skip_dim, out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, xyz_hi: torch.Tensor, xyz_lo: torch.Tensor,
                feat_skip: torch.Tensor, feat_lo: torch.Tensor) -> torch.Tensor:
        """
        xyz_hi:    (B, N, 3)        target coordinates (higher resolution)
        xyz_lo:    (B, S, 3)        source coordinates (lower resolution)
        feat_skip: (B, N, skip_dim) skip-connection from encoder
        feat_lo:   (B, S, low_dim)  features to upsample
        Returns:   (B, N, out_dim)
        """
        B, N, _ = xyz_hi.shape
        _, S, _ = xyz_lo.shape

        if S == 1:
            interpolated = feat_lo.expand(B, N, -1)                        # (B, N, low_dim)
        else:
            dists, knn_idx = torch.cdist(xyz_hi, xyz_lo).topk(3, dim=-1, largest=False)
            inv_d    = 1.0 / (dists + 1e-8)                                # (B, N, 3)
            weights  = inv_d / inv_d.sum(dim=-1, keepdim=True)             # (B, N, 3)
            knn_feat = index_points(feat_lo, knn_idx)                      # (B, N, 3, low_dim)
            interpolated = (weights.unsqueeze(-1) * knn_feat).sum(dim=2)   # (B, N, low_dim)

        fused = torch.cat([feat_skip, interpolated], dim=-1)               # (B, N, skip+low)
        return self.norm(self.fc(fused))                                    # (B, N, out_dim)


# ──────────────────────── Full Backbone ────────────────────────

class PointTransformer(nn.Module):
    """
    Point Transformer encoder-decoder backbone for per-point feature extraction.

    Architecture (3-stage, symmetric):
      Input → Stem → PT₀ → TransDown → PT₁ → TransDown → PT₂  (encoder)
                                    ↕ skip        ↕ skip
                              TransUp → PT₃ → TransUp → PT₄    (decoder)
      → head → (B, N, out_dim)

    Compared to PointNet2:
      - No ball-query radius sensitivity; robust to varying point cloud scales
      - Vector self-attention captures richer local structure for fine geometry
      - Slightly higher memory cost at full resolution due to N×N distance matrix
    """
    def __init__(self, in_dim: int, out_dim: int = 256,
                 npoints: tuple = (512, 128),
                 dims:    tuple = (64, 128, 256),
                 k_attn:  int   = 16,
                 k_down:  int   = 16):
        """
        npoints: (npoint1, npoint2) centroids per TransitionDown stage
        dims:    (d0, d1, d2) feature channels at each encoder level
        k_attn:  k-NN size for PT self-attention (memory: O(B·N·k_attn))
        k_down:  k-NN size for TransitionDown grouping
        """
        super().__init__()
        d0, d1, d2 = dims

        # Stem: project in_dim → d0 (handles arbitrary in_dim including xyz+consensus+time)
        self.stem = nn.Sequential(nn.Linear(in_dim, d0), nn.LayerNorm(d0), nn.ReLU(True))

        # Encoder
        self.pt_enc0 = PointTransformerLayer(d0, k=k_attn)                 # (B, N,    d0)
        self.td1     = TransitionDown(npoints[0], d0, d1, k=k_down)        # (B, np0,  d1)
        self.pt_enc1 = PointTransformerLayer(d1, k=k_attn)                 # (B, np0,  d1)
        self.td2     = TransitionDown(npoints[1], d1, d2, k=k_down)        # (B, np1,  d2)
        self.pt_enc2 = PointTransformerLayer(d2, k=k_attn)                 # (B, np1,  d2)

        # Decoder (skip connections from each encoder level)
        self.tu1     = TransitionUp(d2, d1, d1)                            # (B, np0,  d1)
        self.pt_dec1 = PointTransformerLayer(d1, k=k_attn)                 # (B, np0,  d1)
        self.tu2     = TransitionUp(d1, d0, d0)                            # (B, N,    d0)
        self.pt_dec2 = PointTransformerLayer(d0, k=k_attn)                 # (B, N,    d0)

        # Output projection
        self.head = nn.Sequential(nn.Linear(d0, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim)  — first 3 dims must be xyz
        Returns: (B, N, out_dim)
        """
        xyz = x[:, :, :3]                                                   # (B, N, 3)

        # Stem + encoder level 0
        f0 = self.stem(x)                                                   # (B, N, d0)
        f0 = self.pt_enc0(xyz, f0)                                          # (B, N, d0)

        # Encoder level 1
        xyz1, f1 = self.td1(xyz, f0)                                        # (B, 512, d1)
        f1 = self.pt_enc1(xyz1, f1)                                         # (B, 512, d1)

        # Encoder level 2
        xyz2, f2 = self.td2(xyz1, f1)                                       # (B, 128, d2)
        f2 = self.pt_enc2(xyz2, f2)                                         # (B, 128, d2)

        # Decoder level 1: 128 → 512
        f1_up = self.tu1(xyz1, xyz2, f1, f2)                                # (B, 512, d1)
        f1_up = self.pt_dec1(xyz1, f1_up)                                   # (B, 512, d1)

        # Decoder level 2: 512 → N
        f0_up = self.tu2(xyz, xyz1, f0, f1_up)                              # (B, N, d0)
        f0_up = self.pt_dec2(xyz, f0_up)                                    # (B, N, d0)

        return self.head(f0_up)                                             # (B, N, out_dim)
