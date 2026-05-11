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

    # Step 2: 用 no_grad UOT 预算 matched_x_gt，构造 x_t 直线路径
    with torch.no_grad():
        matched_x_gt, _, _, _ = criterion.compute_ot_assignment(x_0, x_gt)

    # Step 3: 随机采样时间步 t ~ U(0, 1)
    t = torch.rand(B, 1, device=device)
    t_expanded = t.unsqueeze(-1).expand(-1, M, 3)
    x_t = (1 - t_expanded) * x_0 + t_expanded * matched_x_gt

    # Step 4: 前向传播预测速度与存活率
    v_pred, alpha_pred = net(x_t, t, c_dense)

    # Step 5: 调用 criterion (内部跑可微分 UOT 让 α 梯度回流)
    loss, metrics = criterion(
        x_0, x_gt, v_pred, alpha_pred, t,
        matched_x_gt_precomputed=matched_x_gt,
    )

    # Step 6: 反向传播与梯度更新
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    
    # ==========================
    # 📊 REPORTING + ASSERTIONS
    # ==========================
    print("✅ [SUCCESS] Pipeline execution completed without crashing!")
    print(f"⏱️  [TIME] Single step took: {end_time - start_time:.4f} seconds")
    print("📉 [LOSS METRICS]")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")

    # ── A1. 4-tuple unpack contract on compute_ot_assignment ──
    matched_check, surv_check, pi_check, cost_check = criterion.compute_ot_assignment(x_0, x_gt)
    assert matched_check.shape == (B, M, 3)
    assert surv_check.shape == (B, M, 1)
    assert pi_check.shape == (B, M, N_gt)
    assert cost_check.shape == (B, M, N_gt)
    print("[OK] A1 compute_ot_assignment 4-tuple shapes")

    # ── A2. Critical sub-modules must receive non-zero gradient ──
    crit_modules = {
        "consensus_mlp":            net.consensus_mlp,
        "backbone.consensus_branch": net.backbone.consensus_branch,
        "backbone.film_enc0":       net.backbone.film_enc0,
        "backbone.film_enc1":       net.backbone.film_enc1,
        "backbone.film_enc2":       net.backbone.film_enc2,
        "survival_head":            net.survival_head,   # ← key: α flows through Sinkhorn
        "velocity_head":            net.velocity_head,
    }
    for name, mod in crit_modules.items():
        gn = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
        assert gn > 0, f"❌ {name} has zero gradient — {name} is detached from the graph!"
        print(f"[OK] A2 grad flows into {name}: {gn:.4e}")

    # ── A3. Survival ratio sanity (α should not collapse to ~0) ──
    # We only check the lower bound here. The upper bound (e.g. < 0.95) would
    # be the right check on real data with ghost points where good/bad sources
    # must be discriminated, but this test uses uniform-random x_raw and x_gt:
    # every source has a viable target neighbour by construction, so the OT
    # plan saturates surv ≈ a everywhere and survivor_ratio → 1.0 is correct.
    surv_mean = metrics["survivor_ratio"]
    assert surv_mean > 0.02, (
        f"❌ survivor_ratio={surv_mean:.3f} too low — α may be collapsing to -∞")
    print(f"[OK] A3 survivor_ratio={surv_mean:.3f} above collapse threshold")

    # ── A4. PointTransformer FiLM near-identity-at-init ──
    # FiLM is initialized with small Gaussian (std=0.01) — not strict identity,
    # but small enough that the FiLM-augmented network outputs a perturbed
    # version of the un-FiLMed one. Strict identity would require zero-init,
    # which would cut gradient flow into ConsensusMLP at step 0 (see FiLMLayer
    # docstring).
    from models.backbones.point_transformer import PointTransformer
    set_seed(2026)
    pt_with_film = PointTransformer(in_dim=3 + 64, out_dim=256, c_dim=64).to(device)
    max_w = max(pt_with_film.film_enc0.proj.weight.abs().max().item(),
                pt_with_film.film_enc1.proj.weight.abs().max().item(),
                pt_with_film.film_enc2.proj.weight.abs().max().item())
    assert max_w < 0.1, (
        f"❌ FiLM proj weight not in small-init regime: max |w| = {max_w:.3f}")
    print(f"[OK] A4 FiLM proj weights small at init: max |w| = {max_w:.4f}")

    print("\n🎉 ALL ASSERTIONS PASSED")

if __name__ == "__main__":
    main()