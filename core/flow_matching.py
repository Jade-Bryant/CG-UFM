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
            Computes the unbalanced optimal transport plan directly on GPU using PyTorch backend.
            Uses Barycentric projection to find the exact target coordinate and marginal mass for survival.
            """
            B, M, _ = x_0.shape
            _, K, _ = x_gt.shape
            device = x_0.device
            
            matched_x_gt = torch.zeros_like(x_0)
            survival_target = torch.zeros((B, M, 1), device=device)
            
            # UOT Hyperparameters (Can be moved to __init__ later for tuning)
            reg_entropy = self.reg_ot  # Entropy regularization (e.g., 0.05)
            reg_mass = 0.5             # Mass relaxation term (Controls how easily points can "die")
            
            for b in range(B):
                source = x_0[b].detach() # (M, 3)
                target = x_gt[b].detach() # (K, 3)
                
                # 1. Compute squared Euclidean distance matrix on GPU
                # Using torch.cdist for fast batched distance computation
                M_cost = torch.cdist(source.unsqueeze(0), target.unsqueeze(0), p=2).squeeze(0) ** 2
                
                # Normalize cost for numerical stability in Sinkhorn
                cost_max = M_cost.max() + 1e-8
                M_cost_norm = M_cost / cost_max
                
                # 2. Uniform initial mass distribution (Tensors on GPU)
                a = torch.ones(M, device=device, dtype=torch.float32) / M
                b_mass = torch.ones(K, device=device, dtype=torch.float32) / K
                
                # 3. Unbalanced Sinkhorn directly on PyTorch tensors
                # pi shape: (M, K)
                pi = ot.unbalanced.sinkhorn_unbalanced(
                    a, b_mass, M_cost_norm, 
                    reg=reg_entropy, reg_m=reg_mass, 
                    numItermax=1000, stopThr=1e-5
                )
                
                # 4. Extract survival probability from marginals
                # The sum of transported mass from each source point represents its survival score.
                # We normalize it by the initial mass (1/M) so the max score approaches 1.0.
                surviving_mass = pi.sum(dim=1) # (M,)
                survival_score = torch.clamp(surviving_mass / (1.0 / M), min=0.0, max=1.0)
                
                # 5. Barycentric projection to find the continuous target coordinate
                # Instead of a hard Nearest Neighbor, we blend target coordinates based on the OT plan.
                # Add epsilon to prevent division by zero for completely dead points.
                matched_coords = torch.matmul(pi, target) / (surviving_mass.unsqueeze(1) + 1e-8)
                
                matched_x_gt[b] = matched_coords
                survival_target[b, :, 0] = survival_score

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
