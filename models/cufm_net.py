import torch
import torch.nn as nn

from .backbones.pointnet2 import SinusoidalTimeEmbedding
from .backbones.point_transformer import PointTransformer
from .modules.consensus import ConsensusMLP
from .modules.densify import Densifier


class CG_UFM_Network(nn.Module):
    """
    Main model wrapping backbone + dual heads for Consensus-Guided Unbalanced Flow Matching.
    Acts as the continuous vector field f(x_t, t, c) for the ODE solver.

    Backbone: PointTransformer with FiLM modulation by the consensus tensor c.
    The consensus does NOT enter the backbone via input concat (which would be
    diluted by the encoder); it conditions every encoder stage through γ/β.
    """
    def __init__(self,
                 feature_dim: int = 6,
                 c_dim: int = 64,
                 time_emb_dim: int = 64,
                 backbone_dim: int = 256):
        super().__init__()
        self.c_dim = c_dim

        # Modules for feature preprocessing (can be called before ODE solver)
        self.consensus_mlp = ConsensusMLP(input_dim=feature_dim, output_dim=c_dim)
        self.densifier = Densifier(k=4, epsilon=0.05)

        # Time conditioning
        self.time_embedder = SinusoidalTimeEmbedding(dim=time_emb_dim)

        # Backbone input is xyz ⊕ time_emb only — c is injected via FiLM.
        in_dim = 3 + time_emb_dim
        self.backbone = PointTransformer(
            in_dim=in_dim, out_dim=backbone_dim, c_dim=c_dim
        )

        # Dual-Head ODE Outputs
        self.velocity_head = nn.Linear(backbone_dim, 3)        # v_t
        self.survival_head = nn.Linear(backbone_dim, 1)        # α_t logit

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        Defines the continuous vector field for the ODE solver.

        Args:
            x_t: Current spatial coordinates, shape (B, M, 3)
            t:   Current time step, shape (B, 1) or (B,)
            c:   Consensus embedding condition, shape (B, M, c_dim)

        Returns:
            velocity:        (B, M, 3)
            survival_logit:  (B, M, 1)
        """
        B, M, _ = x_t.shape

        t_emb = self.time_embedder(t)                          # (B, time_emb_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(B, M, -1)   # (B, M, time_emb_dim)

        # c is fed separately via FiLM, not concatenated into input.
        node_features = torch.cat([x_t, t_emb_expanded], dim=-1)
        backbone_features = self.backbone(node_features, c=c)

        velocity = self.velocity_head(backbone_features)
        survival_logit = self.survival_head(backbone_features)
        return velocity, survival_logit

class CG_UFM_ODEWrapper(nn.Module):
    """
    The mathematical wrapper that strictly conforms to torchdiffeq's `odeint` signature.
    It encapsulates the static condition `c` and couples (position, mass) into a 4D state.
    """
    def __init__(self, net: CG_UFM_Network, condition_c: torch.Tensor):
        super().__init__()
        self.net = net
        self.c = condition_c # Static consensus embedding (B, M, d)

    def forward(self, t: torch.Tensor, state_4d: torch.Tensor):
        """
        Signature required by torchdiffeq: f(t, y) -> dy/dt
        
        Args:
            t: Current time scalar or 1D tensor
            state_4d: The coupled state [x_t, m_t], shape (B, M, 4)
            
        Returns:
            dstate_dt: The coupled derivative [dx_t/dt, dm_t/dt], shape (B, M, 4)
        """
        # torchdiffeq often passes `t` as a 0D scalar. We format it to (B, 1) for the network.
        B, M, _ = state_4d.shape
        if t.dim() == 0:
            t_batch = t.expand(B, 1).to(state_4d.device)
        elif t.dim() == 1:
            t_batch = t.unsqueeze(1).to(state_4d.device)
        else:
            t_batch = t
            
        # Uncouple the 4D state back into 3D position and 1D mass
        x_t = state_4d[..., :3]  # (B, M, 3)
        # m_t = state_4d[..., 3:] # (B, M, 1) -> Mass is tracked but not input to this specific backbone
        
        # Forward pass through the core network
        # velocity is dx/dt, survival_rate is dm/dt
        velocity, survival_rate = self.net(x_t, t_batch, self.c)
        
        # Recouple the derivatives into 4D
        dstate_dt = torch.cat([velocity, survival_rate], dim=-1) # (B, M, 4)
        
        return dstate_dt
        