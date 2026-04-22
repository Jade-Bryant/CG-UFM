import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import os

# Imports from CG_UFM modules
from data.dataset import UnderwaterPatchDataset
from models.cufm_net import CG_UFM_Network
from core.flow_matching import FlowMatchingLoss

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Load Data
        x_raw = batch['noisy_points'].to(device)   # (B, N, 3)
        features = batch['features'].to(device)    # (B, N, D)
        x_gt = batch['gt_points'].to(device)       # (B, K, 3)
        
        # 1. Forward Pass: Features -> Consensus -> Densify
        c_i = model.consensus_mlp(features)
        x_0, c_dense = model.densifier(x_raw, c_i)
        
        # 2. Sample random time t in [0, 1)
        B, M, _ = x_0.shape
        t = torch.rand((B, 1), device=device)
        
        # --- FIX: Compute OT Target ONLY ONCE ---
        with torch.no_grad():
            matched_x_gt, survival_target = criterion.compute_ot_assignment(x_0, x_gt)
            v_target = matched_x_gt - x_0 # Flow matching linear target velocity
            
        # 3. Construct x_t via Linear Interpolation
        t_expanded = t.unsqueeze(-1).expand(-1, M, 3)
        x_t = (1 - t_expanded) * x_0 + t_expanded * matched_x_gt
        
        # 4. Forward Pass: ODE Network
        v_pred, alpha_pred = model(x_t, t, c_dense)
        
        # 5. Explicit Loss Computation (Avoids double OT computation)
        mask = survival_target.expand_as(v_pred)
        num_survivors = survival_target.sum().clamp(min=1.0)
        
        loss_vel = F.mse_loss(v_pred * mask, v_target * mask, reduction='sum') / num_survivors
        loss_surv = F.binary_cross_entropy_with_logits(alpha_pred, survival_target)
        loss = criterion.lambda_vel * loss_vel + criterion.lambda_surv * loss_surv
        
        metrics = {
            "loss_total": loss.item(),
            "loss_vel": loss_vel.item(),
            "loss_surv": loss_surv.item(),
            "survivor_ratio": (survival_target.sum() / (B * M)).item()
        }
        
        # 6. Backward Pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        wandb.log(metrics)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100 # 增加到真实训练长度
    batch_size = 8
    lr = 1e-4
    save_dir = "./weights"
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project="CG-UFM", config={"learning_rate": lr, "epochs": epochs, "batch_size": batch_size})
    
    # Initialize Dataset & DataLoader
    dataset = UnderwaterPatchDataset(data_dir="./datasets/dummy_dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Optimizer, Scheduler, Criterion
    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64, backbone_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # --- ADDED: Cosine Annealing LR Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = FlowMatchingLoss(lambda_vel=1.0, lambda_surv=2.0).to(device)
    
    print(f"Starting training on {device}...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {current_lr:.6e}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss, "lr": current_lr})
        
        # --- ADDED: Model Checkpointing ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("🌟 New best model saved!")
            
        # Save latest model every epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))

if __name__ == "__main__":
    main()