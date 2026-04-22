import os
import torch
import math
from tqdm import tqdm

def generate_cylinder_points(num_points: int, radius: float, height: float, missing_angle_start: float = None, missing_angle_end: float = None):
    """生成一个可以带大面积随机缺失的圆柱体表面点云"""
    points = []
    # 稍微多生成一些，因为后面要挖洞
    while len(points) < num_points:
        # 随机采样角度和高度
        theta = torch.rand(num_points) * 2 * math.pi
        z = torch.rand(num_points) * height - (height / 2)
        
        # 如果定义了缺失角度，就挖掉这部分 (模拟随机视场盲区)
        if missing_angle_start is not None and missing_angle_end is not None:
            # 找到不在缺失范围内的点
            valid_mask = (theta < missing_angle_start) | (theta > missing_angle_end)
            theta = theta[valid_mask]
            z = z[valid_mask]
            
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        
        valid_points = torch.stack([x, y, z], dim=1)
        points.append(valid_points)
        
    points = torch.cat(points, dim=0)
    
    # 精确截取所需的点数 (点数固定，但因为面积和挖洞随机，密度分布高度随机！)
    indices = torch.randperm(len(points))[:num_points]
    return points[indices]

def generate_random_dummy_patch(num_gt=4096, num_raw=512, feature_dim=6):
    """生成物理属性高度随机化的训练对"""
    # --- 1. 物理参数随机化 ---
    radius = torch.empty(1).uniform_(0.02, 0.08).item() # 管径在 2cm 到 8cm 之间随机
    height = torch.empty(1).uniform_(0.2, 0.5).item()   # 长度在 20cm 到 50cm 之间随机
    noise_level = torch.empty(1).uniform_(0.002, 0.015).item() # 噪声标准差从 2mm 到 1.5cm 随机
    
    # 随机决定挖洞的范围 (模拟 45度 到 180度 的严重残缺)
    missing_start = torch.empty(1).uniform_(0, math.pi).item()
    missing_end = missing_start + torch.empty(1).uniform_(math.pi/4, math.pi).item()
    
    # --- 2. 生成点云 ---
    # GT 是完美的完整圆柱
    x_gt = generate_cylinder_points(num_gt, radius, height)
    
    # Raw 是残缺的圆柱
    x_raw = generate_cylinder_points(num_raw, radius, height, missing_start, missing_end)
    
    # --- 3. 施加随机的海洋雪和折射噪声 ---
    noise = torch.randn_like(x_raw) * noise_level
    x_raw = x_raw + noise
    
    # --- 4. 模拟多模态特征 ---
    features = torch.randn(num_raw, feature_dim)
    
    return {
        'noisy_points': x_raw,      
        'features': features,       
        'gt_points': x_gt           
    }

def main():
    print("🚀 [INIT] Starting RANDOMIZED Dummy Dataset Generation...")
    num_train_samples = 200  
    save_dir = "./data/dummy_dataset"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(num_train_samples), desc="Generating Random Patches"):
        data_dict = generate_random_dummy_patch(num_gt=4096, num_raw=512, feature_dim=6)
        save_path = os.path.join(save_dir, f"patch_{i:04d}.pt")
        torch.save(data_dict, save_path)
        
    print(f"✅ [SUCCESS] Generated {num_train_samples} highly randomized patches in '{save_dir}'.")

if __name__ == "__main__":
    main()