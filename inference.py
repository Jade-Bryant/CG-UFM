import torch
import numpy as np

# Imports from CG_UFM modules
from data.dataset import UnderwaterPatchDataset
from models.cufm_net import CG_UFM_Network
from core.ode_solver import ODESolver
from core.aggregation import NadarayaWatsonAggregator

def infer_patch(model, solver, x_raw, features, device):
    """
    Runs the inference pipeline for a single patch.
    """
    model.eval()
    with torch.no_grad():
        x_raw = x_raw.unsqueeze(0).to(device)      # (1, N, 3)
        features = features.unsqueeze(0).to(device)  # (1, N, D)
        
        # 1. Forward Pass: Features -> Consensus -> Densify
        c_i = model.consensus_mlp(features)
        x_0, c_dense = model.densifier(x_raw, c_i)
        
        # 2. ODE Integration (Forward Euler)
        x_1, alpha_1 = solver.integrate(model, x_0, c_dense)
        
        # 3. Filter dead points (Survival Mask)
        # alpha_1 > 0 means prob > 0.5 (since it's logits)
        survival_mask = alpha_1.squeeze(0).squeeze(-1) > 0.0
        
        surviving_points = x_1.squeeze(0)[survival_mask]
        
        return surviving_points

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting inference on {device}...")
    
    # 1. Load Model
    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64, backbone_dim=256).to(device)
    # model.load_state_dict(torch.load("best_model.pth"))
    
    # 2. Initialize ODE Solver & Aggregator
    solver = ODESolver(method='euler', step_size=0.1)
    aggregator = NadarayaWatsonAggregator(bandwidth=0.1, kernel_type='gaussian')
    
    # 3. Simulate Sliding Window Inference
    # In a real scenario, you'd extract patches iteratively
    # Here we just infer a dummy patch
    dataset = UnderwaterPatchDataset(data_dir="./datasets/dummy_dataset")
    sample = dataset[0]
    x_raw = sample['noisy_points']
    features = sample['features']
    
    surviving_points = infer_patch(model, solver, x_raw, features, device)
    
    print(f"Original patch size: {x_raw.shape[0]}")
    print(f"Inferred dense patch size: {surviving_points.shape[0]}")
    
    # 4. Nadaraya-Watson Aggregation Simulation
    # Assume we have a global point cloud grid and we want to aggregate the normals/colors 
    # of the surviving points. For this example, we just show the API.
    global_points = torch.randn(1000, 3).to(device)
    # Dummy values (e.g., predicted normals) for the patch
    patch_values = torch.ones(surviving_points.shape[0], 3).to(device)
    
    aggregated_values = aggregator.aggregate(global_points, surviving_points, patch_values)
    print(f"Aggregated values shape: {aggregated_values.shape}")

if __name__ == "__main__":
    main()
