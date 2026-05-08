"""Align an existing SfM workspace to the CAD/Blender world frame.

Solves a 7-DoF Sim(3) (Umeyama) between SfM camera centers and the GT
camera centers from `transforms.json`, then applies it to the entire
reconstruction (points and images) in-place. Writes a new sparse model
under `<workspace>/sparse_aligned/` plus a standalone aligned PLY.

Use this when:
    - You have an SfM workspace from `run_sfm.sh` or `run_sfm_hloc.py` and
      want to land its cloud in the CAD frame (so it can be paired with
      `cad_to_gt.py` output as `x_0` / `x_1`).
    - The renders have a `transforms.json` -- the ALIGNMENT only needs the
      GT camera centers, NOT a full known-pose triangulation.

For Blender renders that DO have transforms.json, prefer
`scripts/run_sfm_known_pose.py` instead -- that bypasses the broken
incremental mapper entirely. This script is the fallback for SfM models
that have already been computed (or for future real-data scenes where you
ran a separate calibration).

Usage:
    python scripts/align_sfm_to_cad.py <workspace> <transforms.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workspace", type=Path, help="SfM workspace (output of run_sfm[_hloc].py)")
    p.add_argument("transforms_json", type=Path, help="path to transforms.json with GT poses")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: <workspace>/sparse_aligned)")
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


def find_sparse_dir(workspace: Path) -> Path:
    """run_sfm_hloc.py writes <ws>/sparse/, run_sfm.sh writes <ws>/sparse/0/."""
    for cand in (workspace / "sparse" / "0", workspace / "sparse"):
        if (cand / "cameras.bin").is_file() or (cand / "cameras.txt").is_file():
            return cand
    raise SystemExit(f"No sparse model found under {workspace} (looked at sparse/0 and sparse/)")


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3): finds (s, R, t) s.t. dst ≈ s * R @ src + t.

    src, dst: (N, 3) point clouds in correspondence.
    Returns (scale: float, R: (3,3), t: (3,)) with R in SO(3).
    Reference: Umeyama 1991, "Least-squares estimation of transformation parameters".
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)                        # (3,)
    mu_d = dst.mean(axis=0)                        # (3,)
    src_c = src - mu_s                             # (N, 3)
    dst_c = dst - mu_d                             # (N, 3)
    var_s = (src_c ** 2).sum() / n                 # scalar
    cov = (dst_c.T @ src_c) / n                    # (3, 3) -- note dst on the left
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection guard
        S[2, 2] = -1.0
    R = U @ S @ Vt                                 # (3, 3) in SO(3)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s                        # (3,)
    return s, R, t


def main() -> int:
    args = parse_args()
    workspace: Path = args.workspace.resolve()
    transforms_json: Path = args.transforms_json.resolve()
    out_dir: Path = (args.out or workspace / "sparse_aligned").resolve()

    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")
    if not transforms_json.is_file():
        raise SystemExit(f"transforms.json not found: {transforms_json}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = workspace / f"align_sfm_to_cad_{ts}.log"
    setup_logging(log_path)
    log = logging.getLogger("align_sfm_to_cad")

    sparse_dir = find_sparse_dir(workspace)

    log.info("=" * 64)
    log.info("CG-UFM Sim(3) alignment to CAD frame")
    log.info("  workspace      : %s", workspace)
    log.info("  sparse model   : %s", sparse_dir)
    log.info("  transforms.json: %s", transforms_json)
    log.info("  output         : %s", out_dir)
    log.info("=" * 64)

    import pycolmap

    rec = pycolmap.Reconstruction(sparse_dir)
    log.info("[load] sparse model: cams=%d  imgs=%d  points=%d",
             rec.num_cameras(), rec.num_images(), rec.num_points3D())

    # SfM camera centers, keyed by image basename (matches transforms.json file_path).
    sfm_centers: dict[str, np.ndarray] = {}
    for image in rec.images.values():
        cw = image.cam_from_world()
        R = cw.rotation.matrix()                          # (3, 3)
        t = np.asarray(cw.translation, dtype=np.float64)  # (3,)
        center = -R.T @ t                                 # world coords (SfM frame)
        sfm_centers[Path(image.name).name] = center

    # GT centers from transforms.json. NOTE: in this dataset's exporter,
    # transform_matrix is the Blender/OpenGL **w2c** (despite the NeRF-style
    # field name suggesting c2w). See run_sfm_known_pose.py for the empirical
    # verification. So the camera center in Blender world is:
    #     C_world = -R_w2c^T @ t_w2c
    # The world-frame center is invariant to OpenGL-vs-COLMAP camera-side axis
    # flips, so no convention conversion is needed here.
    raw = json.loads(transforms_json.read_text())
    gt_centers: dict[str, np.ndarray] = {}
    for f in raw["frames"]:
        M = np.asarray(f["transform_matrix"], dtype=np.float64)
        R = M[:3, :3]
        t = M[:3, 3]
        gt_centers[Path(f["file_path"]).name] = -R.T @ t

    common = sorted(sfm_centers.keys() & gt_centers.keys())
    if len(common) < 4:
        log.error("Only %d image names overlap between SfM and transforms.json (need >= 4).", len(common))
        log.error("  sfm names sample: %s", sorted(sfm_centers)[:3])
        log.error("  gt  names sample: %s", sorted(gt_centers)[:3])
        return 2

    src = np.stack([sfm_centers[k] for k in common], axis=0)  # (N, 3) SfM frame
    dst = np.stack([gt_centers[k]  for k in common], axis=0)  # (N, 3) CAD frame
    log.info("[align] paired %d images for Sim(3)", len(common))

    scale, R, t = umeyama_sim3(src, dst)
    aligned = (scale * (R @ src.T)).T + t                     # (N, 3)
    rmse = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    obj_radius = float(raw.get("object_bounding_radius", 0.0))

    log.info("[align] scale = %.6f", scale)
    log.info("[align] R det = %.6f", float(np.linalg.det(R)))
    log.info("[align] t     = [%.4f, %.4f, %.4f]", *t)
    log.info("[align] RMSE  = %.4f  (object_bounding_radius=%.4f)", rmse, obj_radius)
    if obj_radius > 0 and rmse > obj_radius:
        log.warning("[align] RMSE >= object_bounding_radius. SfM is so broken that even Sim(3) can't rescue it.")

    # Apply to the whole reconstruction (points and images).
    sim3 = pycolmap.Sim3d(
        scale=scale,
        rotation=pycolmap.Rotation3d(R),
        translation=t,
    )
    rec.transform(sim3)
    log.info("[apply] Sim(3) applied to %d points + %d images", rec.num_points3D(), rec.num_images())

    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    ply_path = out_dir / "points3D.ply"
    rec.export_PLY(str(ply_path))
    log.info("[write] aligned model -> %s", out_dir)
    log.info("[write] aligned PLY   -> %s", ply_path)
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
