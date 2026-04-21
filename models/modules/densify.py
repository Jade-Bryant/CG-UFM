import torch
import torch.nn as nn
import torch.nn.functional as F

class Densifier(nn.Module):
    """
    Strict Epsilon-ball soft mass injection.
    Takes sparse anchors and performs isotropic volumetric oversampling, 
    strictly inheriting the consensus feature C_i.
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
        device = x_raw.device
        
        # 1. Duplicate anchors k times -> (B, N, k, 3)
        x_expanded = x_raw.unsqueeze(2).expand(-1, -1, self.k, -1)
        
        # 2. Strict Uniform Spherical Volume Sampling (Isotropic)
        # Step A: Generate random directions isotropically (Gaussian -> Normalize)
        directions = torch.randn_like(x_expanded)
        directions = F.normalize(directions, p=2, dim=-1)
        
        # Step B: Generate radii matching the cubic volume growth (u^(1/3))
        u = torch.rand((B, N, self.k, 1), device=device)
        radii = (u ** (1.0 / 3.0)) * self.epsilon
        
        # Step C: Scale directions by radii
        noise = directions * radii
        
        # 3. Apply the strictly bounded epsilon-ball noise
        x_0 = x_expanded + noise
        
        # 4. Replicate consensus features implicitly -> (B, N, k, d)
        c_expanded = c_i.unsqueeze(2).expand(-1, -1, self.k, -1)
        
        # 5. Reshape to flat batch format
        M = N * self.k
        x_0 = x_0.reshape(B, M, 3)
        c_dense = c_expanded.reshape(B, M, d)
        
        return x_0, c_dense