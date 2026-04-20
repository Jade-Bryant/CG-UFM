import torch
from torch.utils.data import DataLoader
import wandb

# Imports from CG_UFM modules
from CG_UFM.data.dataset import UnderwaterPatchDataset
from CG_UFM.models.cufm_net import CG_UFM_Network
from CG_UFM.core.flow_matching import FlowMatchingLoss

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Standard PyTorch training loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Load Data
        x_raw = batch['noisy_points'].to(device)   # (B, N, 3)
        features = batch['features'].to(device)      # (B, N, D)
        x_gt = batch['gt_points'].to(device)         # (B, K, 3)
        
        # 1. Forward Pass: Features -> Consensus -> Densify
        # c_i: (B, N, d)
        c_i = model.consensus_mlp(features)
        
        # x_0: (B, M, 3), c_dense: (B, M, d)
        x_0, c_dense = model.densifier(x_raw, c_i)
        
        # 2. Sample random time t in [0, 1)
        B, M, _ = x_0.shape
        t = torch.rand((B, 1), device=device)
        
        # In Flow Matching, x_t is a linear interpolation between x_0 and x_gt.
        # But here we don't have x_gt mapped point-to-point until OT is computed.
        # A common practice in FM is to compute OT on x_0 and x_gt first,
        # then construct x_t.
        with torch.no_grad():
            matched_x_gt, _ = criterion.compute_ot_assignment(x_0, x_gt)
            
        # Linear Interpolation: x_t = (1 - t) * x_0 + t * matched_x_gt
        t_expanded = t.unsqueeze(-1).expand(-1, M, 3)
        x_t = (1 - t_expanded) * x_0 + t_expanded * matched_x_gt
        
        # 3. Forward Pass: ODE Network
        v_pred, alpha_pred = model(x_t, t, c_dense)
        
        # 4. Loss Computation
        loss, metrics = criterion(x_0, x_gt, v_pred, alpha_pred, t)
        
        # 5. Backward Pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log to wandb
        wandb.log(metrics)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def main():
    # Setup hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 8
    lr = 1e-4
    
    # Initialize wandb
    wandb.init(project="CG-UFM", config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    })
    
    # Initialize Dataset & DataLoader
    # Assuming dummy data loading for now
    dataset = UnderwaterPatchDataset(data_dir="./data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64, backbone_dim=256).to(device)
    
    # Initialize Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FlowMatchingLoss(lambda_vel=1.0, lambda_surv=1.0).to(device)
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})

if __name__ == "__main__":
    main()
