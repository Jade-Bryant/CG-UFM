import torch
import torch.nn as nn

class Densifier(nn.Module):
    """
    Epsilon-ball soft mass injection.
    Takes sparse anchors and creates k neighbor points, copying the C_i feature.
    """
    def __init__(self, k: int = 4, epsilon: float = 0.05):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        
    def forward(self, x_raw: torch.Tensor, c_i: torch.Tensor):
        """
        Args:
            x_raw: Raw points, shape (B, N, 3)
            c_i: Consensus features, shape (B, N, d)
        Returns:
            x_0: Densified points, shape (B, M, 3) where M = N * k
            c_dense: Replicated features, shape (B, M, d)
        """
        B, N, _ = x_raw.shape
        d = c_i.shape[-1]
        
        # Duplicate anchors k times -> (B, N, k, 3)
        x_expanded = x_raw.unsqueeze(2).expand(-1, -1, self.k, -1)
        
        # Add uniform noise within epsilon ball
        # Generate random offsets [-1, 1] scaled by epsilon
        noise = (torch.rand_like(x_expanded) * 2 - 1) * self.epsilon
        
        # Isotropic oversampling
        x_0 = x_expanded + noise
        
        # Replicate features -> (B, N, k, d)
        c_expanded = c_i.unsqueeze(2).expand(-1, -1, self.k, -1)
        
        # Reshape to dense format -> (B, N*k, 3) and (B, N*k, d)
        M = N * self.k
        x_0 = x_0.reshape(B, M, 3)
        c_dense = c_expanded.reshape(B, M, d)
        
        return x_0, c_dense
