# CG-UFM: Project Knowledge Base

**Project Name:** CG-UFM (Consensus-Guided Unbalanced Flow Matching)
**Target Venue:** CoRL 2026 (Conference on Robot Learning)
**Core Domain:** Underwater Perception, Metric-Fidelity 3D Reconstruction, Robotics
**Target Object:** Sparse structures (e.g., PVC pipe networks, T-joints, L-joints, valves with handwheels and etc).

---

## 1. 项目立意 (Project Vision & Motivation)

本项目的核心目标是解决**“真实水下环境中的测量级稀疏结构三维重建与精准测量”**问题。

*   **物理痛点：** 水下光学/声学传感面临严重的散射和衰减。这导致前端 SfM (Structure-from-Motion) 输出的初始点云包含大量由“海洋雪”引起的**幽灵点 (Ghost Mass)**，以及由遮挡引起的**大面积结构断裂 (Missing Mass)**。
*   **理论痛点：** 现有的三维点云处理范式（传统回归、扩散模型 Diffusion、常微分方程流匹配 Flow Matching）在底层动力学上都受限于**“质量守恒假设 (Mass Conservation)”**。它们强制要求噪声输入和干净目标之间存在完美的点对点映射。
*   **视觉退化表现：** 
    *   传统平滑法面对稀疏管线会导致致命的**管径收缩 (Shrinkage)**。
    *   生成式方法（如 Diffusion）面对断裂区域会产生严重的**拓扑幻觉 (Topological Hallucination / 飞线)**。
*   **我们的破局点：** 抛弃纯视觉审美视角，转向**度量级保真 (Metric-Fidelity)**。引入能够允许质量发生“生灭”的**非平衡最优传输 (UOT)**，打造能够用于卡尺级物理测量的机器人感知系统。

---

## 2. 核心解决逻辑 (Problem-Solving Logic)

本算法的“引擎”是 **UFM (Unbalanced Flow Matching)**，而“导航仪”是 **Consensus-Guided (共识引导)** 机制。

### 2.1 引入 4D 状态空间与 Survival Head
为了打破质量守恒，我们将标准的空间坐标 ODE 演化，升级为包含空间位置 $x_t$ 和质量（存活概率） $m_t$ 的 4D 状态空间：
*   **速度场 $v_\theta$：** 决定点如何移动，修复局部骨架。
*   **存活头 $s_\phi$ (Survival Head)：** 允许点在演化中发生质量衰减。对于被判定为光学散射产生的“幽灵点”，网络不会尝试去将其推向管线，而是直接在 ODE 积分过程中将其质量 $m_t$ 降至 0（即“湮灭”）。

### 2.2 共识引导 (Consensus-Guided) 注入
纯空间坐标无法区分“幽灵噪点”和“真实极细管线”。系统引入来自前端传感器的**共识特征 $c_i$**（主要包括多视角 Track Length (时序特征)、重投影误差 Reprojection Error (双目基线特征)）。
*   将 $c_i$ 作为动态条件 (Condition)，通过 FiLM/AdaIN 等特征仿射层强行介入主干网络的特征流，指导 Survival Head 做出正确的生死判决。

---

## 3. 工程实现细节 (Engineering Implementation)

当前项目正处于核心逻辑打通与沙箱适配阶段。

### 3.1 核心代码架构要求
*   **环境隔离：** 采用极其严格的“隔离沙箱”策略。主环境 `cg_ufm` (高版本 PyTorch + `pyproject.toml` + CUDA 12.4) 与 Baseline 环境严格物理隔离。
*   **适配器模式 (Adapter Pattern)：** 针对 PCN、Diffusion 等基线，编写统一的 `run_xxx.py` 包装器。实现统一的流水线：读取 `.pt` -> 坐标归一化 (单位球) -> 模型推理 -> 逆归一化反算真实物理坐标 -> 输出 `.ply`。
*   **亟待修复的三个物理架构 Gap (Current Priorities)：**
    1.  **特征剥离 (Dual Branch)：** 严禁将几何 $(x,y,z)$ 与共识 $c_i$ 在输入端直接 concat。必须使用双分支分别提取，避免 $c_i$ 在 PointNet++ 的 Max-Pooling 中被稀释。
    2.  **动态调制 (FiLM/AdaIN)：** 共识特征 $c_i$ 和时间 $t$ 必须通过 MLP 生成 $\gamma$ 和 $\beta$，介入网络每一层进行物理相乘/相加，而不能只作为静态通道。
    3.  **UOT 动力学闭环：** `POT` 库的 Sinkhorn 计算必须从 `ot.sinkhorn` 替换为 `ot.unbalanced.sinkhorn_unbalanced`。且必须让 Survival Head 预测的存活概率回流到 Sinkhorn 的边际分布向量 (Marginals) 中，真正实现非平衡动力学。

### 3.2 物理数据集制备 (Dataset Pipeline)
*   **绝对 Ground Truth：** 使用高精度激光雷达 (LiDAR) 在“干态 (空气中)”扫描已固定的带有各种拓扑（直角、T型、手轮）的亚光 PVC 框架。
*   **极严尺度对齐：** 利用防水 ArUco/AprilTag 标定板作为锚点。在水池中拍摄多位姿照片并用 COLMAP 跑出 SfM 后，通过 3D-3D 匹配，求出 $R, t, Scale$，将水下极脏的 SfM 点云严格以 1:1 物理毫米级刚性对齐到 LiDAR GT 上。

---

## 4. 评测体系与假想敌 (Evaluation & Baselines)

抛弃传统的 CD (Chamfer Distance) 或 EMD，采用定制化的**拓扑感知尺寸误差 $E_{topo}$**，用于衡量“物理测量能力”。

### 4.1 核心指标
*   **$E_{caliper}$ (卡尺管径误差)：** 重点惩罚管径收缩 (Shrinkage) 现象。
*   **$E_{junc}$ (物理接头漂移误差)：** 重点惩罚 T 型、L 型接头处的飞线与几何幻觉 (Hallucination)。

### 4.2 顶级 Baseline 靶子矩阵 (2026版)
1.  **下限守门员 (梯度平滑派)：** *Score-Based Point Cloud Denoising (ICCV 2021)* 或 *PU-GAN (ICCV 2019)* —— 暴露出它们在水下环境的严重管径坍缩与过度平滑。
2.  **理论天花板 (平衡 OT 派)：** *Diffusion Bridges (ECCV 2024)* —— 证明其坚持“质量守恒”会在遇到海量幽灵点时产生严重拓扑扭曲。
3.  **分辨率霸主 (Transformer 派)：** *PointInfinity (CVPR 2024)* —— 证明无物理约束的纯概率生成会在接头处产生可怕的高频飞线幻觉。
4.  **架构新贵 (Mamba 派)：** *3DMambaIPF / PointMamba (CVPR/ECCV 2024)* —— 证明仅仅更换特征提取器无法解决底层动力学缺陷。
