import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard Positional Encoding for continuous time t.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, 1) or (B,) time values
        Returns: (B, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.view(-1, 1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimplePointNet(nn.Module):
    """
    Base conditional PointNet++ placeholder.
    Currently implemented as a shared MLP operating on concatenated node features.
    """
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        # Using a shared MLP for point-wise operations
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, M, in_dim)
        Returns: Tensor of shape (B, M, out_dim)
        """
        return self.net(x)
