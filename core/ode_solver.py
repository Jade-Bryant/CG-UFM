import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEVectorField(nn.Module):
    """
    Wrapper for the CG_UFM_Network to match the torchdiffeq signature:
    d(x_t)/dt = f(t, x_t, c)
    """
    def __init__(self, cufm_net, c_embed):
        super().__init__()
        self.cufm_net = cufm_net
        self.c_embed = c_embed  # Static condition C_i throughout integration

    def forward(self, t: torch.Tensor, x_t: torch.Tensor):
        """
        Args:
            t: Scalar time tensor
            x_t: Current spatial positions, shape (B, M, 3)
        Returns:
            dx/dt: Velocity, shape (B, M, 3)
        """
        # torchdiffeq passes scalar `t`, expand it to (B, 1)
        B = x_t.shape[0]
        t_batch = t.expand(B, 1).to(x_t.device)
        
        # We only need the velocity for ODE integration of spatial coordinates
        velocity, _ = self.cufm_net(x_t, t_batch, self.c_embed)
        return velocity

class ODESolver:
    """
    Handles the forward ODE integration from t=0 to t=1 using `torchdiffeq`.
    """
    def __init__(self, method: str = 'euler', step_size: float = 0.1):
        """
        Args:
            method: ODE solver method ('euler', 'dopri5', 'rk4', etc.)
            step_size: Step size for fixed-step solvers like 'euler'
        """
        self.method = method
        self.step_size = step_size

    def integrate(self, cufm_net, x_0: torch.Tensor, c_embed: torch.Tensor, t_span=None):
        """
        Integrates the vector field from t=0 to t=1.
        
        Args:
            cufm_net: The main CG_UFM_Network model
            x_0: Initial densified points, shape (B, M, 3)
            c_embed: Consensus features, shape (B, M, d)
            t_span: Optional integration time steps
            
        Returns:
            x_1: Final integrated points at t=1, shape (B, M, 3)
            alpha_1: Final survival logits at t=1, shape (B, M, 1)
        """
        device = x_0.device
        if t_span is None:
            # Integrate from 0 to 1
            t_span = torch.tensor([0.0, 1.0], device=device)
            
        vector_field = ODEVectorField(cufm_net, c_embed)
        
        # Optional kwargs for solver
        options = {}
        if self.method in ['euler', 'rk4']:
            options['step_size'] = self.step_size

        # Solve ODE for spatial coordinates
        # traj shape: (len(t_span), B, M, 3)
        traj = odeint(vector_field, x_0, t_span, method=self.method, options=options)
        
        # Get final spatial positions at t=1
        x_1 = traj[-1]
        
        # Compute final survival logits at t=1
        t_final = torch.ones((x_0.shape[0], 1), device=device)
        _, alpha_1 = cufm_net(x_1, t_final, c_embed)
        
        return x_1, alpha_1
