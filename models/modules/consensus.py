import torch
import torch.nn as nn

class ConsensusMLP(nn.Module):
    """
    Multi-modal fusion MLP to project raw features into a static embedding space.
    Input: F_raw (N, D)
    Output: C_i (N, d)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, N, D) or (N, D)
        Returns:
            c_i: Tensor of shape (B, N, d) or (N, d)
        """
        return self.net(features)
