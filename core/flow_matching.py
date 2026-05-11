import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────
# Batched log-domain Unbalanced Sinkhorn (Chizat et al., NeurIPS 2018)
# ─────────────────────────────────────────────────────────────────────

def _sinkhorn_step(K_log, log_a, log_b, log_u, log_v, tau):
    """One Sinkhorn iteration. Pulled out so we can wrap it in
    `torch.utils.checkpoint` and avoid storing the (B, M, K) broadcasts
    for backward (they get recomputed on the backward pass instead)."""
    log_Kv = torch.logsumexp(K_log + log_v.unsqueeze(1), dim=2)
    log_u = tau * (log_a - log_Kv)
    log_KTu = torch.logsumexp(K_log + log_u.unsqueeze(2), dim=1)
    log_v = tau * (log_b - log_KTu)
    return log_u, log_v


def _batched_sinkhorn_unbalanced_log(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    reg: float,
    reg_m: float,
    num_iter: int,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """
    Vectorized log-domain unbalanced Sinkhorn.

    Implements the KL-relaxed marginal updates (eq. 4.4 in Chizat 2018):
        u^(n+1) = (a / Kv^n)^τ ,    v^(n+1) = (b / Kᵀu^(n+1))^τ
        where τ = reg_m / (reg_m + reg)   ∈ (0, 1].

    Runs on the original device, fully autograd-compatible (gradients flow
    back through `a`, `b`, and `cost`). Works with arbitrary batch dim B
    in a single CUDA stream — no Python loop over batch elements.

    Args:
        a:    (B, M)        source marginals (positive)
        b:    (B, K) or (K,) target marginals (positive); 1D is broadcast
        cost: (B, M, K)     pairwise costs (already normalized as desired)
        reg:                entropic regularization ε > 0
        reg_m:              marginal-relaxation strength λ > 0
        num_iter:           number of Sinkhorn iterations

    Returns:
        pi: (B, M, K) transport plan
    """
    B = a.shape[0]
    if b.dim() == 1:
        b = b.unsqueeze(0).expand(B, -1)

    tau = reg_m / (reg_m + reg)

    log_a = torch.log(a.clamp_min(1e-30))                     # (B, M)
    log_b = torch.log(b.clamp_min(1e-30))                     # (B, K)
    K_log = -cost / reg                                        # (B, M, K)

    log_u = torch.zeros_like(log_a)                           # (B, M)
    log_v = torch.zeros_like(log_b)                           # (B, K)

    for _ in range(num_iter):
        if use_checkpoint and torch.is_grad_enabled():
            # `use_reentrant=False` is required so that we can checkpoint a
            # function whose inputs (log_u, log_v) are produced by an earlier
            # checkpoint. This trades one extra forward of the iteration
            # for not having to store the (B, M, K) broadcasts in the graph.
            log_u, log_v = checkpoint(
                _sinkhorn_step,
                K_log, log_a, log_b, log_u, log_v, tau,
                use_reentrant=False,
            )
        else:
            log_u, log_v = _sinkhorn_step(K_log, log_a, log_b,
                                           log_u, log_v, tau)

    return torch.exp(log_u.unsqueeze(2) + K_log + log_v.unsqueeze(1))


# ─────────────────────────────────────────────────────────────────────
class FlowMatchingLoss(nn.Module):
    """
    Consensus-Guided Unbalanced Flow Matching loss.

    Two-headed supervision:
      • Velocity head — straight-flow MSE against the OT-matched target,
        weighted by the (detached) survival probability.
      • Survival head — supervised by both BCE on the normalized OT marginal
        AND a transport-cost term that backprops through the differentiable
        Sinkhorn plan (so α gradient flows from the OT objective itself).

    The differentiable UOT path uses sigmoid(α_pred) as the source marginal,
    closing the loop required by the "Consensus-Guided UFM" formulation
    (background.md §3.1 Gap 3): the predicted survival mass directly shapes
    the transport plan rather than being a passive output.

    Implementation note: Sinkhorn is solved with a hand-rolled log-domain
    batched solver (`_batched_sinkhorn_unbalanced_log`), no per-batch Python
    loop. POT is no longer a dependency.
    """

    def __init__(self,
                 lambda_vel: float = 1.0,
                 lambda_surv: float = 1.0,
                 lambda_ot: float = 0.1,
                 reg_ot: float = 0.05,
                 reg_mass: float = 1.0,
                 num_iter_diff: int = 5,
                 num_iter_static: int = 1000):
        """
        lambda_vel:       weight for velocity MSE
        lambda_surv:      weight for survival BCE on normalized OT marginal
        lambda_ot:        weight for transport-cost term (the differentiable
                          path through which α learns from OT)
        reg_ot:           Sinkhorn entropy regularization ε
        reg_mass:         marginal-relaxation strength λ for unbalanced OT
                          (1.0 tolerates the sum(a)≈M·0.5 vs sum(b)=1 mismatch
                          when using sigmoid(α) as the source marginal)
        num_iter_diff:    Sinkhorn iterations on the differentiable path.
                          Kept small (5) because the iteration is wrapped in
                          `torch.utils.checkpoint`, so the autograd graph for
                          this path is essentially flat regardless of the
                          count — but each iteration still costs a forward
                          recompute in backward. 5 is enough for unbalanced
                          Sinkhorn to give a useful gradient signal.
        num_iter_static:  Sinkhorn iterations on the no_grad path used to
                          precompute matched_x_gt for x_t construction
        """
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_surv = lambda_surv
        self.lambda_ot = lambda_ot
        self.reg_ot = reg_ot
        self.reg_mass = reg_mass
        self.num_iter_diff = num_iter_diff
        self.num_iter_static = num_iter_static

    # ─────────────────────────────────────────────────────────────────
    def compute_ot_assignment(self, x_0: torch.Tensor, x_gt: torch.Tensor,
                              alpha_pred: torch.Tensor = None):
        """
        Solve the (unbalanced) entropic OT between x_0 and x_gt.

        Args:
            x_0:        (B, M, 3) source point cloud
            x_gt:       (B, K, 3) target point cloud
            alpha_pred: (B, M, 1) survival logit. If None, uses uniform source
                        marginal a=1/M (no_grad path, returned tensors are not
                        differentiable). If provided, uses sigmoid(soft_clamp(α))
                        as the source marginal — gradients flow back into α
                        through the entire returned tuple.

        Returns:
            matched_x_gt:   (B, M, 3) barycentric projection of x_gt under π
            surviving_mass: (B, M, 1) row-sum of π, scaled to [0,1] by ·M and
                            clamped (matches legacy semantics so BCE targets
                            stay valid)
            pi:             (B, M, K) full transport plan
            cost:           (B, M, K) normalized cost matrix (detached from x_0)

        Cost is always built from `detach()`-ed coordinates so that OT only
        backprops through α (positions are supervised by the velocity head).
        """
        B, M, _ = x_0.shape
        _, K, _ = x_gt.shape
        device = x_0.device

        use_diff = alpha_pred is not None
        num_iter = self.num_iter_diff if use_diff else self.num_iter_static

        # Cost matrix — always detached so OT solely backprops through α
        src = x_0.detach()
        tgt = x_gt.detach()
        cost = torch.cdist(src, tgt, p=2) ** 2                      # (B, M, K)
        cost = cost / (cost.amax(dim=(1, 2), keepdim=True) + 1e-8)

        # Source marginal
        if use_diff:
            # Soft clamp keeps gradients alive in the saturation band.
            alpha_soft = 10.0 * torch.tanh(alpha_pred.squeeze(-1) / 10.0)
            a = torch.sigmoid(alpha_soft)                           # (B, M) ∈ (0,1)
        else:
            a = torch.full((B, M), 1.0 / M, device=device)

        # Target marginal: uniform within batch, but total mass matched to a.
        # Without this, sum(a) ≈ M/2 vs sum(b) = 1 puts the KL-relaxation in
        # an impossible regime — the plan can transport ≪ a, so surv/a ≪ 1
        # for almost every point regardless of correspondence quality.
        # Matching totals lets surv ≈ a for points with good geometric matches
        # and surv ≪ a for points with bad ones — what the survival head
        # should actually learn from.
        b = a.sum(dim=1, keepdim=True).expand(-1, K) / K            # (B, K)

        # Batched log-domain unbalanced Sinkhorn.
        # On the differentiable path we wrap each iteration in
        # `torch.utils.checkpoint` so the (B, M, K) broadcasts aren't kept in
        # the autograd graph — critical when M=K=4096, B≈8, since each such
        # broadcast is ~512 MiB and the loop produces two per iteration.
        pi = _batched_sinkhorn_unbalanced_log(
            a, b, cost,
            reg=self.reg_ot, reg_m=self.reg_mass, num_iter=num_iter,
            use_checkpoint=use_diff,
        )                                                            # (B, M, K)

        # Marginal mass per source point.
        # Normalize by the source marginal `a` itself: surv_b / a_i is the
        # fraction of the mass we sent that actually got transported. This is
        # well-defined for both legacy (a=1/M) and diff (a=sigmoid(α)) paths,
        # and bounded in [0, 1+ε] post-KL-relaxation. Clamp to [0,1] so BCE
        # targets are valid logits.
        surv = pi.sum(dim=2)                                        # (B, M)
        survival_score = torch.clamp(
            surv / (a + 1e-8), min=0.0, max=1.0
        ).unsqueeze(-1)

        # Barycentric projection
        matched_x_gt = torch.bmm(pi, tgt) / (surv.unsqueeze(-1) + 1e-8)

        return matched_x_gt, survival_score, pi, cost

    # ─────────────────────────────────────────────────────────────────
    def forward(self,
                x_0: torch.Tensor,
                x_gt: torch.Tensor,
                v_pred: torch.Tensor,
                alpha_pred: torch.Tensor,
                t: torch.Tensor,
                matched_x_gt_precomputed: torch.Tensor):
        """
        Compute total loss for one training step.

        Caller is expected to have already computed `matched_x_gt_precomputed`
        in a no_grad pre-pass (used to build x_t = (1-t)·x_0 + t·matched).
        We reuse it here as the velocity supervision target, and run a SECOND
        OT pass with α_pred to get the differentiable plan that supervises α.

        Args:
            x_0:                       (B, M, 3) source (densified)
            x_gt:                      (B, K, 3) target
            v_pred:                    (B, M, 3) predicted velocity
            alpha_pred:                (B, M, 1) predicted survival logit
            t:                         (B, 1) time
            matched_x_gt_precomputed:  (B, M, 3) from no_grad pre-pass
        """
        # Differentiable OT — α learns from this branch
        _, surv_norm, pi, cost = self.compute_ot_assignment(x_0, x_gt, alpha_pred)

        # Velocity supervision uses the static (no_grad) match — detached so the
        # velocity loss does NOT push α through the matching pathway.
        v_target = (matched_x_gt_precomputed - x_0).detach()
        weights = torch.sigmoid(alpha_pred).detach().squeeze(-1)        # (B, M)
        sq_err = ((v_pred - v_target) ** 2).sum(dim=-1)                 # (B, M)
        loss_vel = (sq_err * weights).sum() / weights.sum().clamp(min=1.0)

        # Survival BCE on the (detached) normalized OT marginal — stable target
        loss_surv = F.binary_cross_entropy_with_logits(
            alpha_pred, surv_norm.detach()
        )

        # Transport cost — α gradient backprops through π. Normalize by total
        # transported mass so the term lives in [0, 1] regardless of sum(a),
        # otherwise lambda_ot has to be retuned every time α's distribution shifts.
        loss_transport = (pi * cost.detach()).sum() / pi.sum().clamp(min=1e-8)

        loss_total = (self.lambda_vel * loss_vel
                      + self.lambda_surv * loss_surv
                      + self.lambda_ot * loss_transport)

        metrics = {
            "loss_total": loss_total.item(),
            "loss_vel": loss_vel.item(),
            "loss_surv": loss_surv.item(),
            "loss_transport": loss_transport.item(),
            "survivor_ratio": surv_norm.mean().item(),
            "alpha_mean": torch.sigmoid(alpha_pred).mean().item(),
        }
        return loss_total, metrics


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, M, K = 2, 100, 80
    x_0 = torch.randn(B, M, 3)
    x_gt = torch.randn(B, K, 3)
    v_pred = torch.randn(B, M, 3, requires_grad=True)
    alpha_pred = torch.randn(B, M, 1, requires_grad=True)
    t = torch.rand(B, 1)

    criterion = FlowMatchingLoss()

    with torch.no_grad():
        matched, _, _, _ = criterion.compute_ot_assignment(x_0, x_gt)

    loss, metrics = criterion(x_0, x_gt, v_pred, alpha_pred, t,
                              matched_x_gt_precomputed=matched)
    loss.backward()
    print("metrics:", metrics)
    print("alpha grad norm:", alpha_pred.grad.norm().item())
    assert alpha_pred.grad.norm().item() > 0, "α has no gradient!"
    print("OK")
