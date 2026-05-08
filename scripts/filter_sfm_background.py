"""Drop obvious far-background points from a known-pose SfM workspace.

Keeps points within a sphere of radius `K * object_bounding_radius` around
`object_center` (both read from the scene's transforms.json). Default K=2.0
removes floor/wall/cabinet matches without touching near-object ghosts -- those
are the training signal CG-UFM's mass head is supposed to learn to kill.

Why mild and not aggressive:
    UFM's dual-head ODE (CLAUDE.md §2.1) evolves both position AND survival
    logit. To learn the mass-killing dynamics it needs ghost-mass examples
    INSIDE the cloud. Stripping the cloud back to "only on-surface" leaves the
    mass head with no training signal. So we only chop the obvious far stuff
    that drowns the gradient otherwise.

Usage:
    python scripts/filter_sfm_background.py <workspace> [--scene-dir DIR]
    # auto-detects scene_dir / transforms.json from the workspace's log file.

Output:
    <workspace>/sparse_filtered/points3D.ply   <-- the filtered cloud, this is x_0
    <workspace>/sparse_filtered/filter_stats.json
    <workspace>/sparse_filtered/points3D_colored.ply (kept=green, dropped=red, for inspection)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workspace", type=Path, help="output of run_sfm_known_pose.py")
    p.add_argument("--scene-dir", type=Path, default=None,
                   help="scene folder with transforms.json. If omitted, parsed from workspace log.")
    p.add_argument("--radius-factor", type=float, default=2.0,
                   help="keep points where ||p - object_center|| < radius_factor * object_bounding_radius "
                        "(default 2.0; tighten to 1.5 for cleaner clouds at the cost of mass-head signal)")
    p.add_argument("--min-points", type=int, default=512,
                   help="warn if filtered cloud has fewer points than this (matches dataset.py x_0 target)")
    return p.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def detect_scene_dir(workspace: Path) -> Path:
    """Parse the most recent run_sfm_known_pose log for the `transforms:` line."""
    logs = sorted(workspace.glob("run_sfm_known_pose_*.log"))
    if not logs:
        raise SystemExit(
            f"No run_sfm_known_pose log under {workspace}. "
            f"Pass --scene-dir explicitly."
        )
    text = logs[-1].read_text()
    m = re.search(r"transforms\s*:\s*(\S+)/transforms\.json", text)
    if not m:
        raise SystemExit(f"Could not parse transforms path from {logs[-1]}.")
    return Path(m.group(1))


def main() -> int:
    args = parse_args()
    workspace: Path = args.workspace.resolve()
    scene_dir: Path = (args.scene_dir or detect_scene_dir(workspace)).resolve()

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.is_file():
        raise SystemExit(f"transforms.json not found at {transforms_path}")

    sparse_dir = workspace / "sparse"
    src_ply = sparse_dir / "points3D.ply"
    if not src_ply.is_file():
        raise SystemExit(f"sparse PLY not found at {src_ply}")

    out_dir = workspace / "sparse_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(workspace / f"filter_sfm_background_{ts}.log")
    log = logging.getLogger("filter_sfm_background")

    # Deferred import so --help works without the corl env active.
    import open3d as o3d

    tj = json.loads(transforms_path.read_text())
    obj_center = np.asarray(tj["object_center"], dtype=np.float64)
    obj_radius = float(tj["object_bounding_radius"])
    keep_radius = args.radius_factor * obj_radius

    pcd = o3d.io.read_point_cloud(str(src_ply))
    pts = np.asarray(pcd.points, dtype=np.float64)         # (N, 3)
    cols = np.asarray(pcd.colors)                           # (N, 3) or empty

    log.info("=" * 64)
    log.info("CG-UFM SfM background filter")
    log.info("  workspace        : %s", workspace)
    log.info("  scene_dir        : %s", scene_dir)
    log.info("  src points       : %d", len(pts))
    log.info("  object_center    : %s", obj_center.round(4).tolist())
    log.info("  object_radius    : %.4f", obj_radius)
    log.info("  radius_factor    : %.2f  (keep_radius = %.4f)", args.radius_factor, keep_radius)
    log.info("=" * 64)

    d = np.linalg.norm(pts - obj_center, axis=1)            # (N,)
    keep = d < keep_radius                                   # (N,) bool

    n_keep = int(keep.sum())
    n_drop = int(len(pts) - n_keep)

    # Distance percentiles for the kept set vs the dropped set
    log.info("[stats] kept   : %d (%.1f%%)   d-percentiles "
             "p10 %.4f  p50 %.4f  p90 %.4f  max %.4f",
             n_keep, 100*n_keep/len(pts),
             np.percentile(d[keep], 10) if n_keep else float("nan"),
             np.percentile(d[keep], 50) if n_keep else float("nan"),
             np.percentile(d[keep], 90) if n_keep else float("nan"),
             d[keep].max() if n_keep else float("nan"))
    log.info("[stats] dropped: %d (%.1f%%)   d-percentiles "
             "p10 %.4f  p50 %.4f  p90 %.4f  max %.4f",
             n_drop, 100*n_drop/len(pts),
             np.percentile(d[~keep], 10) if n_drop else float("nan"),
             np.percentile(d[~keep], 50) if n_drop else float("nan"),
             np.percentile(d[~keep], 90) if n_drop else float("nan"),
             d[~keep].max() if n_drop else float("nan"))

    if n_keep < args.min_points:
        log.warning("[warn] kept points (%d) < min_points (%d)!", n_keep, args.min_points)
        log.warning("[warn] dataset.py will pad via random sampling-with-replacement, creating duplicates.")
        log.warning("[warn] consider raising --radius-factor, improving render textures, or skipping this scene.")

    # Write filtered PLY
    pcd_kept = o3d.geometry.PointCloud()
    pcd_kept.points = o3d.utility.Vector3dVector(pts[keep])
    if cols.size:
        pcd_kept.colors = o3d.utility.Vector3dVector(cols[keep])
    out_ply = out_dir / "points3D.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd_kept)
    log.info("[write] filtered cloud -> %s", out_ply)

    # Write a 2-color inspection PLY: green=kept, red=dropped
    inspect_cols = np.zeros((len(pts), 3))
    inspect_cols[keep]  = [0.0, 0.8, 0.0]
    inspect_cols[~keep] = [0.8, 0.0, 0.0]
    pcd_inspect = o3d.geometry.PointCloud()
    pcd_inspect.points = o3d.utility.Vector3dVector(pts)
    pcd_inspect.colors = o3d.utility.Vector3dVector(inspect_cols)
    inspect_ply = out_dir / "points3D_colored.ply"
    o3d.io.write_point_cloud(str(inspect_ply), pcd_inspect)
    log.info("[write] inspection PLY -> %s   (green=kept, red=dropped)", inspect_ply)

    # Stats sidecar
    stats = {
        "src_ply": str(src_ply),
        "scene_dir": str(scene_dir),
        "object_center": obj_center.tolist(),
        "object_bounding_radius": obj_radius,
        "radius_factor": args.radius_factor,
        "keep_radius": keep_radius,
        "src_points": int(len(pts)),
        "kept": n_keep,
        "dropped": n_drop,
        "min_points_warning_threshold": args.min_points,
        "warned_below_min": bool(n_keep < args.min_points),
    }
    (out_dir / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("[write] stats -> %s", out_dir / "filter_stats.json")
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
