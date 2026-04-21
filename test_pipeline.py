import torch
import torch.optim as optim
import time

# 导入我们刚刚浴血奋战写出来的核心模块
from models.cufm_net import CG_UFM_Network
from models.modules.densify import Densifier
from core.flow_matching import FlowMatchingLoss

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(2026) # 预祝 CoRL 2026 中稿
    
    # 1. 硬件探测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [INIT] Testing pipeline on device: {device}")
    
    # 2. 模拟配置参数 (Hyperparameters)
    B = 2             # Batch Size
    N_raw = 512       # 初始稀疏 SfM 点云数量
    K = 4             # 过采样倍率 (Densifier k)
    M = N_raw * K     # 演化点云数量 (2048)
    N_gt = 4096       # 稠密 GT 点云数量 (非平衡状态: M != N_gt)
    feature_dim = 6   # 多模态特征维度 (Stereo + Temporal + Geo)
    
    print(f"📦 [DATA] Batch: {B}, Raw Points: {N_raw}, Densified: {M}, GT Points: {N_gt}")

    # 3. 实例化模型与 Loss (移入目标设备)
    print("🛠️  [BUILD] Initializing Networks and Loss...")
    densifier = Densifier(k=K, epsilon=0.05).to(device)
    net = CG_UFM_Network(feature_dim=feature_dim, c_dim=64, time_emb_dim=64, backbone_dim=256).to(device)
    criterion = FlowMatchingLoss(lambda_vel=1.0, lambda_surv=2.0, reg_ot=0.05).to(device)
    
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 4. 生成 Dummy Data (假数据)
    # x_raw: 模拟落在 [-1, 1] 空间的稀疏骨架
    x_raw = (torch.rand(B, N_raw, 3, device=device) * 2 - 1)
    # features: 模拟抽取的 6 维共识特征
    features = torch.randn(B, N_raw, feature_dim, device=device)
    # x_gt: 模拟极其致密的高精度干态扫描表面
    x_gt = (torch.rand(B, N_gt, 3, device=device) * 2 - 1)
    
    print("🔄 [RUN] Starting Forward-Backward test...\n")
    start_time = time.time()

    # ==========================
    # 🏋️ TRAINING LOOP SIMULATION
    # ==========================
    optimizer.zero_grad()

    # Step 1: 抽取特征并进行几何空间致密化 (Soft Mass Injection)
    c_i = net.consensus_mlp(features)       # (B, N_raw, d)
    x_0, c_dense = densifier(x_raw, c_i)    # x_0: (B, M, 3), c_dense: (B, M, d)
    
    # Step 2: [核心流匹配逻辑] 预先算出目标点，用于构造轨迹
    # 注意：计算 OT 时不需要梯度，这纯粹是为了找目标配对
    with torch.no_grad():
        matched_x_gt, survival_target = criterion.compute_ot_assignment(x_0, x_gt)
    
    # Step 3: 随机采样时间步 t ~ U(0, 1)
    t = torch.rand(B, 1, device=device)
    t_expanded = t.unsqueeze(-1).expand(-1, M, 1) # (B, M, 1) 用于广播插值
    
    # Step 4: 构造时间 t 刻的中间物理状态 x_t (Optimal Transport 直线路径)
    # 公式：x_t = (1-t)*x_0 + t*x_1
    x_t = (1 - t_expanded) * x_0 + t_expanded * matched_x_gt
    
    # Step 5: 前向传播预测速度与存活率
    v_pred, alpha_pred = net(x_t, t, c_dense)
    
    # Step 6: 计算 Loss
    # 我们的 FlowMatchingLoss 内部期望 x_0 以计算真正的 target_velocity
    loss, metrics = criterion(x_0, x_gt, v_pred, alpha_pred, t)
    
    # Step 7: 反向传播与梯度更新
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    
    # ==========================
    # 📊 REPORTING
    # ==========================
    print("✅ [SUCCESS] Pipeline execution completed without crashing!")
    print(f"⏱️  [TIME] Single step took: {end_time - start_time:.4f} seconds")
    print("📉 [LOSS METRICS]")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")
        
    # 检查梯度是否断裂
    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    print(f"⚡ [GRADIENT] Total network gradient norm: {grad_norm:.4f}")
    if grad_norm == 0:
        print("❌ [WARNING] Gradient is ZERO! The computation graph is broken.")
    else:
        print("✔️ [OK] Gradients are actively flowing through the network.")

if __name__ == "__main__":
    main()