"""Sample ground-truth point clouds from CAD meshes via Poisson Disk Sampling.

Reads .obj files (with or without companion .mtl) and writes one .pt per
mesh containing a uniformly-sampled surface point cloud, intended as the
x_1 target in the CG-UFM flow-matching pipeline.

Usage:
    # single file
    python scripts/cad_to_gt.py path/to/Elbow.obj datasets/gt/

    # batch over a directory tree (recursive)
    python scripts/cad_to_gt.py datasets/cad_root/ datasets/gt/ --recursive

Output schema (one .pt per mesh, key matches generate_dummy_data.py):
    {
        'gt_points': (N, 3) float32,        # surface points
        'gt_normals': (N, 3) float32,       # outward-facing normals
        'mesh_name': str,
        'normalize': dict | None,           # {'center': (3,), 'scale': float} if --normalize
    }

Method:
    1. Read mesh with open3d.
    2. Optional: simplify to <= max_faces for speed.
    3. Poisson Disk sample N * oversample points.
    4. Farthest Point Sampling to exactly N points (more robust than PDS
       on its own, which sometimes returns slightly fewer points).
    5. Optional: normalize to unit AABB or unit sphere.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help=".obj file or directory containing .obj files")
    p.add_argument("output_dir", type=Path)
    p.add_argument("-n", "--num-points", type=int, default=4096)
    p.add_argument("--oversample", type=float, default=2.0,
                   help="PDS samples N*oversample then FPS down to N (default 2.0)")
    p.add_argument("--max-faces", type=int, default=200_000,
                   help="decimate mesh above this face count for PDS speed (0 = never)")
    p.add_argument("--recursive", action="store_true", help="search input dir recursively")
    p.add_argument("--normalize", choices=["none", "aabb", "sphere"], default="aabb",
                   help="normalize to unit AABB (default), unit sphere, or keep CAD units")
    p.add_argument("--no-ply", action="store_true",
                   help="skip writing the companion .ply (default: write one alongside the .pt)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")],
    )


def find_meshes(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.obj" if recursive else "*.obj"
    return sorted(path.glob(pattern))


def fps_indices(points: np.ndarray, n: int) -> np.ndarray:
    """CPU farthest-point-sampling. Adequate for n <= ~16k."""
    M = points.shape[0]
    if M <= n:
        return np.arange(M)
    selected = np.empty(n, dtype=np.int64)
    selected[0] = np.random.randint(M)                                # start: random
    dist = np.linalg.norm(points - points[selected[0]], axis=1)       # (M,)
    for i in range(1, n):
        idx = int(np.argmax(dist))
        selected[i] = idx
        new_d = np.linalg.norm(points - points[idx], axis=1)
        dist = np.minimum(dist, new_d)
    return selected


def sample_one(mesh_path: Path, n: int, oversample: float, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    # Use trimesh to load: it auto-triangulates N-gon faces (quads, 450-gons, etc.)
    # that Open3D silently drops, which caused near-empty GT for T_joint / Y_joint.
    tm = trimesh.load(str(mesh_path), force='mesh')
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tm.vertices),
        o3d.utility.Vector3iVector(tm.faces),
    )
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    if max_faces and len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)

    mesh.compute_vertex_normals()
    mesh.orient_triangles()                                           # consistent normal orientation

    n_oversample = int(n * oversample)
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_oversample, use_triangle_normal=True)
    pts = np.asarray(pcd.points, dtype=np.float32)                    # (M, 3)
    nrm = np.asarray(pcd.normals, dtype=np.float32)                   # (M, 3)

    if pts.shape[0] < n:
        # PDS returned fewer than requested -- pad by area-weighted random sampling.
        deficit = n - pts.shape[0]
        extra = mesh.sample_points_uniformly(number_of_points=deficit)
        pts = np.concatenate([pts, np.asarray(extra.points, dtype=np.float32)])
        nrm = np.concatenate([nrm, np.asarray(extra.normals, dtype=np.float32)])

    idx = fps_indices(pts, n)
    return pts[idx], nrm[idx]


def normalize(pts: np.ndarray, mode: str) -> tuple[np.ndarray, dict | None]:
    if mode == "none":
        return pts, None
    center = pts.mean(axis=0)                                         # (3,)
    centered = pts - center
    if mode == "aabb":
        scale = float(np.abs(centered).max())                         # half-extent of AABB
    elif mode == "sphere":
        scale = float(np.linalg.norm(centered, axis=1).max())         # bounding sphere radius
    else:
        raise ValueError(mode)
    if scale < 1e-9:
        scale = 1.0
    return (centered / scale).astype(np.float32), {"center": center.astype(np.float32), "scale": scale}


def estimate_pipe_diameter(pts: np.ndarray) -> float:
    """Estimate the local tube/pipe diameter from sampled surface points.

    For pipe-shaped objects (elbow, reducer, straight_pipe, T-joint, Y-joint),
    the AABB short axes ≈ the outer diameter — the long axis runs along the
    pipe's main flow direction, and the two perpendicular axes both span the
    cross-section. We use the *mean of the two shortest axis extents* as a
    rough physical diameter estimate.

    This is intentionally a per-scene scalar rather than a full diameter
    field — E_caliper is designed as a single-number summary anyway. For
    reducers (which change diameter along the length) this returns an
    average; that's acceptable because E_caliper's purpose is to detect
    over-smoothing collapse, not characterise variable cross-sections.

    Returns the estimate in the *output* (post-normalize) coordinate frame,
    so benchmark.py can directly compare it to estimate_local_diameter()
    on the prediction without further bookkeeping.
    """
    aabb = pts.max(axis=0) - pts.min(axis=0)            # (3,)
    sorted_extent = np.sort(aabb)                       # short → long
    # Mean of the two short axes = mean cross-section span ≈ outer diameter.
    return float((sorted_extent[0] + sorted_extent[1]) / 2.0)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(args.output_dir / f"cad_to_gt_{ts}.log")
    log = logging.getLogger("cad_to_gt")

    meshes = find_meshes(args.input, args.recursive)
    if not meshes:
        log.error("No .obj files found at %s", args.input)
        return 1

    log.info("=" * 64)
    log.info("CG-UFM CAD -> GT point cloud (Poisson Disk + FPS)")
    log.info("  input      : %s (%d meshes)", args.input, len(meshes))
    log.info("  output_dir : %s", args.output_dir)
    log.info("  N          : %d  (oversample=%.1fx)", args.num_points, args.oversample)
    log.info("  normalize  : %s", args.normalize)
    log.info("=" * 64)

    n_ok, n_skip, n_fail = 0, 0, 0
    t0 = time.time()
    for mesh_path in meshes:
        out_path = args.output_dir / f"{mesh_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            log.info("[SKIP] %s already exists", out_path.name)
            n_skip += 1
            continue
        try:
            pts, nrm = sample_one(mesh_path, args.num_points, args.oversample, args.max_faces)
            pts, norm_meta = normalize(pts, args.normalize)
            # Estimate per-scene physical diameter on the post-normalize cloud
            # so benchmark.py's E_caliper compares apples to apples (the
            # prediction is also written in normalized frame from stage_infer).
            gt_diameter = estimate_pipe_diameter(pts)
            torch.save({
                "gt_points": torch.from_numpy(pts),                   # (N, 3)
                "gt_normals": torch.from_numpy(nrm),                  # (N, 3)
                "mesh_name": mesh_path.stem,
                "normalize": norm_meta,
                # Per-scene pipe diameter — read by benchmark.py to replace
                # the global --gt-diameter fallback. With this, E_caliper
                # is no longer comparing to a globally-tuned constant.
                "gt_diameter": float(gt_diameter),
            }, out_path)
            if not args.no_ply:
                ply_path = out_path.with_suffix(".ply")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
                o3d.io.write_point_cloud(str(ply_path), pcd)
            log.info("[OK]   %-40s -> %s  (%d pts)", mesh_path.name, out_path.name, pts.shape[0])
            n_ok += 1
        except Exception as e:
            log.exception("[FAIL] %s: %s", mesh_path, e)
            n_fail += 1

    log.info("=" * 64)
    log.info("Done in %.1fs  ok=%d  skip=%d  fail=%d", time.time() - t0, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
