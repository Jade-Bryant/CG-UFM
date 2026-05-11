"""Compose a training-ready .pt for one scene from real SfM + CAD outputs.

Schema (matches generate_dummy_data.py / data/dataset.py):
    {
        "noisy_points": (M, 3) float32,   # filtered SfM cloud  (x_0)
        "features":     (M, 6) float32,   # per-point feature vector
        "gt_points":    (K, 3) float32,   # CAD-sampled cloud   (x_1)
    }

Inputs:
    1. SfM workspace (output of run_sfm_known_pose.py + filter_sfm_background.py).
       Reads `<workspace>/sparse_filtered/points3D.ply`.
       Falls back to `<workspace>/sparse/points3D.ply` if `--use-raw` is set.
    2. CAD GT .pt produced by cad_to_gt.py.
       Reads `gt_points`, `gt_normals` and the optional `normalize` block.

Crucial behaviour:
    If the CAD GT was AABB-normalised (i.e. `normalize` is non-null), the SAME
    affine `(p - center) / scale` is applied to the SfM cloud here. Otherwise
    the two clouds end up in different scales and x_0/x_1 misalign all over
    again, undoing what known-pose triangulation just bought us.

Features:
    Default `--features random` matches the dummy generator -- per-point
    Gaussian noise of dim 6, frozen at .pt write time. The model has been
    trained against random features so far, so this preserves training
    behaviour. `--features rgb_pad6` swaps in PLY colors (R,G,B,0,0,0) for
    later experiments where you want the consensus head to actually have
    photometric signal.

Usage:
    python scripts/ply_to_pt.py \\
        dataset_test/result_known_pose \\
        datasets/gt/Elbow_30deg_004.pt \\
        datasets/training/Elbow_30deg_004.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workspace", type=Path, help="SfM workspace (with sparse_filtered/ or sparse/)")
    p.add_argument("gt_pt", type=Path, help="cad_to_gt.py output for this scene")
    p.add_argument("out_pt", type=Path, help="path to write the merged .pt")
    p.add_argument("--use-raw", action="store_true",
                   help="read sparse/points3D.ply instead of sparse_filtered/points3D.ply "
                        "(skips background filter -- not recommended)")
    p.add_argument("--features", choices=["random", "zeros", "rgb_pad6"], default="random",
                   help="how to fill the per-point feature tensor (default: random, matches dummy)")
    p.add_argument("--feature-dim", type=int, default=6,
                   help="feature dimension (default 6, matches dummy/model)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for random features (default 0). Frozen at .pt write time, so "
                        "this only affects which exact noise is stored.")
    p.add_argument("--min-noisy", type=int, default=512,
                   help="warn if noisy point count < this (matches dataset.py x_0 target)")
    p.add_argument("--min-gt", type=int, default=4096,
                   help="warn if gt point count < this (matches dataset.py x_1 target)")
    return p.parse_args()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ply_to_pt")


def find_sfm_ply(workspace: Path, use_raw: bool) -> Path:
    cand = workspace / ("sparse" if use_raw else "sparse_filtered") / "points3D.ply"
    if not cand.is_file():
        # be helpful: if user forgot --use-raw and only sparse/ exists
        fallback = workspace / "sparse" / "points3D.ply"
        if fallback.is_file() and not use_raw:
            raise SystemExit(
                f"sparse_filtered/points3D.ply not found at {cand}.\n"
                f"Run: python scripts/filter_sfm_background.py {workspace}\n"
                f"or pass --use-raw to consume the unfiltered cloud at {fallback} (NOT recommended)."
            )
        raise SystemExit(f"PLY not found at {cand}")
    return cand


def make_features(N: int, mode: str, rgb: np.ndarray | None, dim: int, seed: int) -> torch.Tensor:
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randn(N, dim, generator=g, dtype=torch.float32)
    if mode == "zeros":
        return torch.zeros(N, dim, dtype=torch.float32)
    if mode == "rgb_pad6":
        if rgb is None or rgb.size == 0:
            raise SystemExit("--features rgb_pad6 requires the PLY to carry colors (none found).")
        rgb_t = torch.as_tensor(np.asarray(rgb), dtype=torch.float32)   # (N, 3) in [0,1]
        if dim <= 3:
            return rgb_t[:, :dim].contiguous()
        pad = torch.zeros(N, dim - 3, dtype=torch.float32)
        return torch.cat([rgb_t, pad], dim=1)
    raise ValueError(f"unknown features mode: {mode}")


def main() -> int:
    args = parse_args()
    log = setup_logging()

    workspace: Path = args.workspace.resolve()
    gt_pt:      Path = args.gt_pt.resolve()
    out_pt:     Path = args.out_pt.resolve()

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")
    if not gt_pt.is_file():
        raise SystemExit(f"gt .pt not found: {gt_pt}")

    sfm_ply = find_sfm_ply(workspace, args.use_raw)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 64)
    log.info("CG-UFM merge SfM + CAD into training .pt")
    log.info("  workspace : %s", workspace)
    log.info("  sfm PLY   : %s", sfm_ply.relative_to(workspace))
    log.info("  gt .pt    : %s", gt_pt)
    log.info("  out .pt   : %s", out_pt)
    log.info("  features  : %s  dim=%d  seed=%d", args.features, args.feature_dim, args.seed)
    log.info("=" * 64)

    # Load filtered SfM cloud (deferred import).
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(sfm_ply))
    sfm_pts = np.asarray(pcd.points, dtype=np.float32)             # (M, 3)
    sfm_rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    if len(sfm_pts) == 0:
        raise SystemExit(f"SfM PLY has no points: {sfm_ply}")

    # Load CAD GT.
    gt_data = torch.load(gt_pt, weights_only=False)
    gt_pts: torch.Tensor = gt_data["gt_points"]
    if not isinstance(gt_pts, torch.Tensor):
        gt_pts = torch.as_tensor(np.asarray(gt_pts), dtype=torch.float32)
    gt_pts = gt_pts.to(torch.float32)

    # Apply CAD's normalize affine to SfM (keeps x_0 and x_1 in the same frame).
    norm = gt_data.get("normalize")
    if norm is not None:
        center = np.asarray(norm["center"], dtype=np.float32).reshape(1, 3)   # (1, 3)
        scale  = float(norm["scale"])
        log.info("[normalize] re-applying CAD's AABB transform to SfM:  center=%s  scale=%.4f",
                 center.flatten().tolist(), scale)
        sfm_pts = (sfm_pts - center) / scale
    else:
        log.info("[normalize] CAD .pt has no normalize block -- both clouds kept in raw Blender frame")

    # Sanity stats on the post-normalization clouds.
    def stats(arr, name):
        mn = arr.min(0); mx = arr.max(0); cm = arr.mean(0); rd = np.linalg.norm(arr - cm, axis=1).max()
        log.info("[stats] %-12s  N=%5d  AABB [%.3f..%.3f, %.3f..%.3f, %.3f..%.3f]  centroid %s  radius %.3f",
                 name, len(arr), mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], cm.round(3).tolist(), rd)
    stats(sfm_pts, "noisy")
    stats(gt_pts.numpy(), "gt")

    # Features.
    feats = make_features(
        N=len(sfm_pts),
        mode=args.features,
        rgb=sfm_rgb if sfm_rgb is not None else None,
        dim=args.feature_dim,
        seed=args.seed,
    )

    # Sanity warnings -- dataset.py will pad to 512/4096 via random-w-replacement if short.
    if len(sfm_pts) < args.min_noisy:
        log.warning("[warn] noisy points (%d) < %d -- dataset.py will create duplicates via "
                    "sample-with-replacement.", len(sfm_pts), args.min_noisy)
    if len(gt_pts) < args.min_gt:
        log.warning("[warn] gt points (%d) < %d -- dataset.py will create duplicates via "
                    "sample-with-replacement.", len(gt_pts), args.min_gt)

    # Write merged .pt.
    out = {
        "noisy_points": torch.as_tensor(sfm_pts, dtype=torch.float32),     # (M, 3)
        "features":     feats,                                              # (M, dim)
        "gt_points":    gt_pts.contiguous(),                                # (K, 3)
    }
    # Forward the CAD-derived per-scene physical diameter if cad_to_gt
    # populated it. benchmark.py prefers this over the global --gt-diameter
    # fallback, so E_caliper is computed against the actual scene's pipe
    # size instead of a one-size-fits-all constant.
    if "gt_diameter" in gt_data:
        out["gt_diameter"] = float(gt_data["gt_diameter"])
    torch.save(out, out_pt)
    log.info("[write] merged .pt -> %s   (noisy=%d, features=%dx%d, gt=%d)",
             out_pt, len(sfm_pts), len(sfm_pts), args.feature_dim, len(gt_pts))
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
