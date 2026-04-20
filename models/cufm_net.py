import torch
import torch.nn as nn

from .backbones.pointnet2 import SimplePointNet, SinusoidalTimeEmbedding
from .modules.consensus import ConsensusMLP
from .modules.densify import Densifier

class CG_UFM_Network(nn.Module):
    """
    Main model wrapping backbone + dual heads for Consensus-Guided Unbalanced Flow Matching.
    This acts as the continuous vector field f(x_t, t, c) for the ODE solver.
    """
    def __init__(self, 
                 feature_dim: int = 6, 
                 c_dim: int = 64, 
                 time_emb_dim: int = 64, 
                 backbone_dim: int = 256):
        super().__init__()
        
        # Modules for feature preprocessing (can be called before ODE solver)
        self.consensus_mlp = ConsensusMLP(input_dim=feature_dim, output_dim=c_dim)
        self.densifier = Densifier(k=4, epsilon=0.05)
        
        # Time conditioning
        self.time_embedder = SinusoidalTimeEmbedding(dim=time_emb_dim)
        
        # Backbone input dimension: 3 (xyz) + c_dim (consensus) + time_emb_dim
        in_dim = 3 + c_dim + time_emb_dim
        self.backbone = SimplePointNet(in_dim=in_dim, out_dim=backbone_dim)
        
        # Dual-Head ODE Outputs
        # 1. Spatial Velocity v_t
        self.velocity_head = nn.Linear(backbone_dim, 3)
        # 2. Survival (Mass) Logit \alpha_t
        self.survival_head = nn.Linear(backbone_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        Defines the continuous vector field for the ODE solver.
        
        Args:
            x_t: Current spatial coordinates, shape (B, M, 3)
            t: Current time step, shape (B, 1) or (B,)
            c: Consensus embedding condition, shape (B, M, d)
            
        Returns:
            velocity: Spatial velocity v_t, shape (B, M, 3)
            survival_logit: Survival mass logit \alpha_t, shape (B, M, 1)
        """
        B, M, _ = x_t.shape
        
        # Compute time embedding: (B, time_emb_dim)
        t_emb = self.time_embedder(t)
        
        # Expand time embedding to match number of points: (B, M, time_emb_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(B, M, -1)
        
        # Concatenate features along the feature dimension
        node_features = torch.cat([x_t, c, t_emb_expanded], dim=-1)
        
        # Extract deep features using backbone
        backbone_features = self.backbone(node_features)
        
        # Compute Dual-Head outputs
        velocity = self.velocity_head(backbone_features)
        survival_logit = self.survival_head(backbone_features)
        
        return velocity, survival_logit
