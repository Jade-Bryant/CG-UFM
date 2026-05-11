import argparse
import math
import os
from functools import partial
from multiprocessing import Pool

import torch
from tqdm import tqdm


def generate_cylinder_points(num_points: int, radius: float, height: float,
                             missing_angle_start: float = None,
                             missing_angle_end: float = None):
    """生成一个可以带大面积随机缺失的圆柱体表面点云"""
    points = []
    while len(points) < num_points:
        theta = torch.rand(num_points) * 2 * math.pi
        z = torch.rand(num_points) * height - (height / 2)

        if missing_angle_start is not None and missing_angle_end is not None:
            valid_mask = (theta < missing_angle_start) | (theta > missing_angle_end)
            theta = theta[valid_mask]
            z = z[valid_mask]

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        valid_points = torch.stack([x, y, z], dim=1)
        points.append(valid_points)

    points = torch.cat(points, dim=0)
    indices = torch.randperm(len(points))[:num_points]
    return points[indices]


def generate_random_dummy_patch(num_gt=4096, num_raw=512, feature_dim=6):
    """生成物理属性高度随机化的训练对"""
    radius = torch.empty(1).uniform_(0.02, 0.08).item()
    height = torch.empty(1).uniform_(0.2, 0.5).item()
    noise_level = torch.empty(1).uniform_(0.002, 0.015).item()

    missing_start = torch.empty(1).uniform_(0, math.pi).item()
    missing_end = missing_start + torch.empty(1).uniform_(math.pi / 4, math.pi).item()

    x_gt = generate_cylinder_points(num_gt, radius, height)
    x_raw = generate_cylinder_points(num_raw, radius, height, missing_start, missing_end)

    noise = torch.randn_like(x_raw) * noise_level
    x_raw = x_raw + noise

    features = torch.randn(num_raw, feature_dim)

    return {
        'noisy_points': x_raw,
        'features': features,
        'gt_points': x_gt,
    }


def _worker(i: int, save_dir: str, num_gt: int, num_raw: int, feature_dim: int) -> str:
    """One sample → one .pt. Pure CPU work + disk write."""
    # Each worker gets an independent torch RNG seed derived from its index
    # so that across `--workers N` runs, samples remain reproducible per index.
    torch.manual_seed(i)
    data_dict = generate_random_dummy_patch(
        num_gt=num_gt, num_raw=num_raw, feature_dim=feature_dim)
    save_path = os.path.join(save_dir, f"patch_{i:04d}.pt")
    torch.save(data_dict, save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--save-dir", default="./datasets/dummy_dataset")
    parser.add_argument("--num-gt", type=int, default=4096)
    parser.add_argument("--num-raw", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="multiprocessing pool size (CPU-bound; not GPU-bound)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"🚀 [INIT] Generating {args.num_samples} patches with {args.workers} workers...")

    fn = partial(_worker,
                 save_dir=args.save_dir,
                 num_gt=args.num_gt,
                 num_raw=args.num_raw,
                 feature_dim=args.feature_dim)

    if args.workers <= 1:
        for i in tqdm(range(args.num_samples), desc="Generating (single)"):
            fn(i)
    else:
        with Pool(processes=args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(fn, range(args.num_samples)),
                          total=args.num_samples,
                          desc="Generating (parallel)"):
                pass

    print(f"✅ [SUCCESS] Generated {args.num_samples} patches in '{args.save_dir}'.")


if __name__ == "__main__":
    main()
