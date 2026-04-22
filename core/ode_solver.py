import torch
from torchdiffeq import odeint

from models.cufm_net import CG_UFM_ODEWrapper


class ODESolver:
    """
    Handles the forward ODE integration from t=0 to t=1 using `torchdiffeq`.
    Integrates the full 4D state [x_t, m_t] as required by the UOT formulation.
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
        Integrates the 4D vector field [dx/dt, dm/dt] from t=0 to t=1.

        Args:
            cufm_net: The main CG_UFM_Network model
            x_0: Initial densified points, shape (B, M, 3)
            c_embed: Consensus features, shape (B, M, d)
            t_span: Optional integration time steps

        Returns:
            x_1: Final integrated positions at t=1, shape (B, M, 3)
            alpha_1: Final survival logits at t=1, shape (B, M, 1)
        """
        device = x_0.device
        B, M, _ = x_0.shape

        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=device)

        # Initial mass logit = 0 (sigmoid(0) = 0.5, neutral survival prior)
        m_0 = torch.zeros(B, M, 1, device=device)
        state_0 = torch.cat([x_0, m_0], dim=-1)  # (B, M, 4)

        wrapper = CG_UFM_ODEWrapper(cufm_net, c_embed)

        options = {}
        if self.method in ['euler', 'rk4']:
            options['step_size'] = self.step_size

        # traj shape: (len(t_span), B, M, 4)
        traj = odeint(wrapper, state_0, t_span, method=self.method, options=options)

        state_1 = traj[-1]  # (B, M, 4)
        x_1 = state_1[..., :3]   # (B, M, 3)
        alpha_1 = state_1[..., 3:]  # (B, M, 1)

        return x_1, alpha_1
