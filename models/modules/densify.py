import torch
import torch.nn as nn
import torch.nn.functional as F


class Densifier(nn.Module):
    """Strict Epsilon-ball soft mass injection.

    Takes sparse anchors and performs isotropic volumetric oversampling,
    strictly inheriting the consensus feature C_i.

    Epsilon semantics:
      - `epsilon` is the *relative* radius, expressed as a fraction of the
        per-batch point-cloud extent (max coordinate spread / 2).
      - With per-sample unit-sphere normalization in the dataset, the
        cloud half-extent is ≈ 1, so eps_relative=0.05 → physical radius
        0.05 in normalized space — same as the legacy absolute value 0.05.
      - For un-normalized inputs (legacy callers, sanity tests), this falls
        back to the absolute interpretation if `adaptive_epsilon=False`.

    Why adaptive: the old absolute ε=0.05 was the proximate cause of the
    "short-axis fattening" failure. Long pipes have bbox z-extent ≈ 2 but
    short-axis extent ≈ 0.4 after cad_to_gt's AABB normalize. An isotropic
    ε=0.05 noise was 2.5% of the long axis but 12.5% of the short axis,
    so the densified initial points were already smeared sideways. Scaling
    ε by per-sample extent (after dataset normalization) keeps the noise
    proportional to the object.
    """

    def __init__(self, k: int = 4, epsilon: float = 0.05,
                 adaptive_epsilon: bool = True):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon

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

        # Determine per-batch epsilon.
        if self.adaptive_epsilon:
            # Robust per-batch half-extent: use 5th/95th percentile spread
            # instead of full min/max. SfM clouds carry outliers far outside
            # the actual object (background residuals, mismatched tracks),
            # so raw bbox would overestimate the cloud's effective size and
            # inflate ε. The unit-sphere normalization in the dataset is
            # defined from GT, so the *real* object scale is ≈ 1 — using
            # robust percentiles recovers that even when noisy carries
            # outliers.
            with torch.no_grad():
                lo = torch.quantile(x_raw, 0.05, dim=1)             # (B, 3)
                hi = torch.quantile(x_raw, 0.95, dim=1)             # (B, 3)
                half_extent = (hi - lo).norm(dim=1) / 2.0           # (B,)
                # Lower bound stops degenerate single-point clouds from
                # nuking ε to 0; upper bound caps un-normalized legacy
                # callers so one big scene doesn't dominate.
                half_extent = half_extent.clamp(min=1e-3, max=2.0)
            eps_per_batch = (self.epsilon * half_extent).view(B, 1, 1, 1)
        else:
            eps_per_batch = self.epsilon  # scalar broadcast

        radii = (u ** (1.0 / 3.0)) * eps_per_batch

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
