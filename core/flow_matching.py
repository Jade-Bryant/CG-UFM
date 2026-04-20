import torch
import torch.nn as nn
import torch.nn.functional as F
import ot  # Python Optimal Transport library (POT)
import numpy as np

class FlowMatchingLoss(nn.Module):
    """
    Computes the Unbalanced Optimal Transport assignment and Dual-Head Loss 
    for Consensus-Guided Unbalanced Flow Matching.
    """
    def __init__(self, lambda_vel: float = 1.0, lambda_surv: float = 1.0, reg_ot: float = 0.1):
        """
        Args:
            lambda_vel: Weight for Velocity MSE Loss
            lambda_surv: Weight for Survival BCE Loss
            reg_ot: Entropy regularization parameter for Sinkhorn OT
        """
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_surv = lambda_surv
        self.reg_ot = reg_ot

    def compute_ot_assignment(self, x_0: torch.Tensor, x_gt: torch.Tensor):
        """
        Computes the unbalanced or partial optimal transport plan between x_0 and x_gt.
        Here we use POT's exact EMD or Sinkhorn for batch processing.
        
        Args:
            x_0: Source points, shape (B, M, 3)
            x_gt: Target points, shape (B, K, 3)
            
        Returns:
            matched_x_gt: The target points aligned to x_0, shape (B, M, 3)
            survival_target: Binary mask indicating if an x_0 point is successfully matched (1) or should die (0), shape (B, M, 1)
        """
        B, M, _ = x_0.shape
        _, K, _ = x_gt.shape
        device = x_0.device
        
        matched_x_gt = torch.zeros_like(x_0)
        survival_target = torch.zeros((B, M, 1), device=device)
        
        # OT is typically solved per sample in the batch
        for b in range(B):
            source = x_0[b].detach().cpu().numpy()
            target = x_gt[b].detach().cpu().numpy()
            
            # Compute cost matrix (squared Euclidean distance)
            M_cost = ot.dist(source, target, metric='sqeuclidean')
            M_cost = M_cost / M_cost.max() # Normalize cost matrix for numerical stability
            
            # Uniform mass distribution
            a, b_mass = np.ones((M,)) / M, np.ones((K,)) / K
            
            # Use exact Earth Mover's Distance (EMD) assignment 
            # (or use ot.sinkhorn for differentiable/faster approx)
            # Since this is unbalanced (M != K), we use Partial OT or relax the mass conservation.
            # For simplicity in this scaffold, we find the nearest neighbor in GT for each x_0
            # weighted by the OT plan.
            # Real implementation might use `ot.partial.partial_wasserstein` or similar.
            pi = ot.emd(a, b_mass, M_cost)
            
            # From the transport plan pi (M x K), find the most probable target for each source
            # If the max probability is below a threshold, we consider it "dead" (noise)
            max_prob_indices = np.argmax(pi, axis=1)
            max_probs = np.max(pi, axis=1)
            
            # Threshold to determine survival (heuristic for unbalanced matching)
            survival_threshold = 1e-5
            is_survived = max_probs > survival_threshold
            
            matched_x_gt[b] = x_gt[b, max_prob_indices]
            survival_target[b, is_survived, 0] = 1.0

        return matched_x_gt, survival_target

    def forward(self, 
                x_0: torch.Tensor, 
                x_gt: torch.Tensor, 
                v_pred: torch.Tensor, 
                alpha_pred: torch.Tensor, 
                t: torch.Tensor):
        """
        Computes the dual-head loss.
        
        Args:
            x_0: Initial points (densified), shape (B, M, 3)
            x_gt: Ground truth clean points, shape (B, K, 3)
            v_pred: Predicted velocity from the network, shape (B, M, 3)
            alpha_pred: Predicted survival logit from the network, shape (B, M, 1)
            t: Time step, shape (B, 1) - Used if v_target needs time-dependent scaling
            
        Returns:
            loss: Total combined loss
            loss_dict: Dictionary with individual loss components for logging
        """
        # 1. Compute UOT assignments (Static target matching)
        # Note: OT matching can be computationally heavy, often done on CPU.
        with torch.no_grad():
            matched_x_gt, survival_target = self.compute_ot_assignment(x_0, x_gt)
            
            # 2. Compute Flow Matching Target (Linear Flow path)
            # v_target = X_GT^{matched} - X_0
            v_target = matched_x_gt - x_0
            
        # 3. Velocity Head Loss (MSE)
        # We only compute velocity loss on points that are "surviving"
        mask = survival_target.expand_as(v_pred)
        # Avoid division by zero
        num_survivors = survival_target.sum().clamp(min=1.0)
        
        loss_vel = F.mse_loss(v_pred * mask, v_target * mask, reduction='sum') / num_survivors
        
        # 4. Survival Head Loss (BCE with Logits)
        loss_surv = F.binary_cross_entropy_with_logits(alpha_pred, survival_target)
        
        # Total Loss
        loss_total = self.lambda_vel * loss_vel + self.lambda_surv * loss_surv
        
        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_vel": loss_vel.item(),
            "loss_surv": loss_surv.item(),
            "survivor_ratio": (survival_target.sum() / (x_0.shape[0] * x_0.shape[1])).item()
        }
        
        return loss_total, loss_dict

# Test snippet
if __name__ == "__main__":
    B, M, K = 2, 100, 80
    x_0 = torch.randn(B, M, 3)
    x_gt = torch.randn(B, K, 3)
    v_pred = torch.randn(B, M, 3)
    alpha_pred = torch.randn(B, M, 1)
    t = torch.rand(B, 1)
    
    criterion = FlowMatchingLoss()
    loss, metrics = criterion(x_0, x_gt, v_pred, alpha_pred, t)
    print("Loss Metrics:", metrics)
