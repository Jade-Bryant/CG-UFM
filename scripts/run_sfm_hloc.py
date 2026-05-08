"""HLoc-based SfM for CG-UFM underwater datasets.

Replaces SIFT (which collapses under heavy scattering) with LoFTR — a
detector-free dense matcher that produces correspondences in low-texture
and scattering-degraded regions where keypoint detectors miss. Output
layout matches scripts/run_sfm.sh so downstream code does not change:

    <workspace>/
        image_list.txt
        run_sfm_hloc_<ts>.log
        features.h5
        matches.h5
        pairs-exhaustive.txt
        sparse/0/{cameras,images,points3D}.bin
        sparse/0/points3D.ply

Install (once):
    pip install hloc                       # high-level pipeline
    pip install pycolmap                   # COLMAP Python bindings
    # LoFTR weights are downloaded by hloc on first run.
    # SuperPoint/SuperGlue/LightGlue are still selectable via --matcher.

Usage:
    python scripts/run_sfm_hloc.py <image_dir> <workspace> [--fresh]

Notes:
    - Exhaustive pairs are fine for <= 200 images. For larger sets switch
      to pairs_from_retrieval (NetVLAD) -- the hloc API is identical.
    - Non-image files in <image_dir> (.obj/.mtl/.json) are filtered via
      a whitelist passed to hloc as the image_list argument.
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_dir", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument("--fresh", action="store_true", help="wipe workspace before running")
    p.add_argument(
        "--feature",
        default="superpoint_aachen",
        help="hloc feature config name; only used when --matcher is a sparse "
             "(detector-based) matcher (default: superpoint_aachen)",
    )
    p.add_argument(
        "--matcher",
        default="loftr",
        help="hloc matcher config name (default: loftr, detector-free dense). "
             "Sparse alternatives: superglue, superpoint+lightglue, disk+lightglue.",
    )
    p.add_argument(
        "--relax", action="store_true",
        help="lower COLMAP mapper/triangulation thresholds so weakly-matched scenes still produce a (noisier) sparse cloud -- ideal for the broken x_0 we want",
    )
    p.add_argument(
        "--dense-features", action="store_true",
        help="extract more SuperPoint keypoints (defaults: 6144, NMS=1, threshold=0) and lower matcher threshold to 0.05; recommended for hollow/thin objects where SuperPoint defaults under-sample edges",
    )
    p.add_argument(
        "--max-keypoints", type=int, default=6144,
        help="SuperPoint keypoints per image when --dense-features is on (default 6144; 8192 is GPU-only practical)",
    )
    p.add_argument(
        "--resize-max", type=int, default=1024,
        help="longest image side fed to SuperPoint when --dense-features is on (default 1024)",
    )
    p.add_argument(
        "--match-threshold", type=float, default=0.05,
        help="matcher confidence threshold when --dense-features is on (default 0.05; matcher default is 0.2)",
    )
    return p.parse_args()


# Looser mapper/triangulation thresholds. Standard COLMAP defaults are tuned
# for "give me a metric reconstruction"; for CG-UFM we WANT the broken,
# under-constrained sparse cloud as x_0, so we trade accuracy for survival.
# pycolmap 4.x nests options: top-level + .mapper + .triangulation.
RELAXED_MAPPER_OPTIONS: dict = {
    "min_num_matches": 8,                          # default 15  (DB-level pair filter)
    "mapper": {
        "init_min_num_inliers":      30,           # default 100
        "abs_pose_min_num_inliers":  15,           # default 30
        "abs_pose_min_inlier_ratio": 0.15,         # default 0.25
        "filter_max_reproj_error":   8.0,          # default 4.0
        "filter_min_tri_angle":      0.5,          # default 1.5  (deg)
    },
    "triangulation": {
        "min_angle":                 0.5,          # default 1.5  (deg)
        "create_max_angle_error":    4.0,          # default 2.0
        "continue_max_angle_error":  4.0,          # default 2.0
        "merge_max_reproj_error":    8.0,          # default 4.0
        "complete_max_reproj_error": 8.0,          # default 4.0
        "ignore_two_view_tracks":    False,        # default True -- KEEP 2-view tracks
    },
}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def collect_images(image_dir: Path) -> list[str]:
    names = sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not names:
        raise SystemExit(f"No images found in {image_dir}")
    return names


def export_ply(model_or_dir, ply_path: Path) -> int:
    """Dump points3D.ply from either a pycolmap.Reconstruction or its dir."""
    import pycolmap

    rec = (
        model_or_dir
        if isinstance(model_or_dir, pycolmap.Reconstruction)
        else pycolmap.Reconstruction(model_or_dir)
    )
    rec.export_PLY(str(ply_path))
    return rec.num_points3D()


def main() -> int:
    args = parse_args()
    image_dir: Path = args.image_dir.resolve()
    workspace: Path = args.workspace.resolve()

    if not image_dir.is_dir():
        raise SystemExit(f"image_dir not found: {image_dir}")

    if args.fresh and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = workspace / f"run_sfm_hloc_{ts}.log"
    setup_logging(log_path)
    log = logging.getLogger("run_sfm_hloc")

    image_names = collect_images(image_dir)
    image_list_path = workspace / "image_list.txt"
    image_list_path.write_text("\n".join(image_names) + "\n")

    log.info("=" * 64)
    log.info("CG-UFM HLoc SfM")
    log.info("  image_dir : %s", image_dir)
    log.info("  workspace : %s", workspace)
    log.info("  num_images: %d", len(image_names))
    log.info("  feature   : %s", args.feature)
    log.info("  matcher   : %s", args.matcher)
    log.info("  log       : %s", log_path)
    log.info("=" * 64)

    # Imports deferred so that --help works without hloc installed.
    try:
        from hloc import (
            extract_features,
            match_dense,
            match_features,
            pairs_from_exhaustive,
            reconstruction,
        )
    except ImportError as e:
        raise SystemExit(
            f"hloc import failed ({e}). hloc is GitHub-only, not on PyPI. Install with:\n"
            "    git clone --recursive https://github.com/cvg/Hierarchical-Localization.git ~/tools/hloc\n"
            "    pip install -e ~/tools/hloc\n"
            "    pip install pycolmap\n"
            "The --recursive flag matters: SuperGlue ships as a git submodule "
            "(LoFTR weights are pulled at runtime, but SuperGlue is needed if "
            "you ever switch --matcher back to a sparse one)."
        )

    is_dense = args.matcher in match_dense.confs                # LoFTR-family path
    if is_dense:
        matcher_conf = copy.deepcopy(match_dense.confs[args.matcher])
        feature_conf = None
        log.info("[detector-free] using dense matcher: %s", args.matcher)
    else:
        feature_conf = copy.deepcopy(extract_features.confs[args.feature])
        matcher_conf = copy.deepcopy(match_features.confs[args.matcher])

    if args.dense_features:
        if is_dense:
            log.warning("--dense-features is a SuperPoint knob and is ignored for dense matchers")
        else:
            # More keypoints, low NMS, no confidence threshold. Output name is
            # bumped so caches don't clash with default-config runs.
            feature_conf["model"]["max_keypoints"]      = args.max_keypoints
            feature_conf["model"]["nms_radius"]         = 1
            feature_conf["model"]["keypoint_threshold"] = 0.0
            feature_conf["preprocessing"]["resize_max"] = args.resize_max
            feature_conf["output"] = f"feats-superpoint-dense-n{args.max_keypoints}-r{args.resize_max}"
            # Both SuperGlue and LightGlue take match_threshold; lowering it keeps
            # weak correspondences that COLMAP's RANSAC can still vet.
            matcher_conf["model"]["match_threshold"] = args.match_threshold
            matcher_conf["output"] = f"{matcher_conf['output']}-thr{args.match_threshold:g}"
            log.info("[dense-features] SuperPoint: max_kp=%d nms=1 thr=0.0 resize=%d",
                     args.max_keypoints, args.resize_max)
            log.info("[dense-features] %s: match_threshold=%g", args.matcher, args.match_threshold)

    sfm_pairs = workspace / "pairs-exhaustive.txt"
    # pycolmap.Reconstruction writes cameras/images/points3D.bin directly into
    # this dir (not into a numbered subdir like the COLMAP CLI does).
    sfm_dir = workspace / "sparse"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if is_dense:
        # Detector-free path: pairs first, then a single pass extracts pseudo-
        # keypoints and matches them densely (LoFTR / RoMa / etc.).
        log.info("[1/3] Building exhaustive image pairs")
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)

        log.info("[2/3] Dense matching with %s (this also generates the keypoints)", args.matcher)
        feature_path, match_path = match_dense.main(
            matcher_conf,
            sfm_pairs,
            image_dir,
            export_dir=workspace,
            max_kps=args.max_keypoints,
        )
    else:
        log.info("[1/4] Extracting %s features", args.feature)
        feature_path = extract_features.main(
            feature_conf,
            image_dir,
            workspace,
            image_list=image_names,
        )

        log.info("[2/4] Building exhaustive image pairs")
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)

        log.info("[3/4] Matching with %s", args.matcher)
        match_path = match_features.main(
            matcher_conf,
            sfm_pairs,
            feature_conf["output"],
            workspace,
        )

    last_step = "[3/3]" if is_dense else "[4/4]"
    log.info("%s Triangulation + incremental reconstruction (pycolmap)", last_step)
    if args.relax:
        log.info("  [relax] using lenient mapper/triangulation thresholds")
        for k, v in RELAXED_MAPPER_OPTIONS.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    log.info("    %-12s.%-28s = %s", k, sk, sv)
            else:
                log.info("    %-41s = %s", k, v)
    model = reconstruction.main(
        sfm_dir,
        image_dir,
        sfm_pairs,
        feature_path,
        match_path,
        image_list=image_names,
        mapper_options=RELAXED_MAPPER_OPTIONS if args.relax else None,
    )

    if model is None or model.num_reg_images() == 0:
        log.error("Reconstruction produced no registered images.")
        log.error("Even %s could not handle these images.", args.matcher)
        log.error("Try --relax, or inspect features.h5 / matches.h5 -- the scattering may be too strong.")
        return 2

    ply_path = sfm_dir / "points3D.ply"
    n_pts = export_ply(model, ply_path)

    log.info("=" * 64)
    log.info("Summary")
    log.info("  registered images : %d / %d", model.num_reg_images(), len(image_names))
    log.info("  3D points         : %d", n_pts)
    log.info("  mean track length : %.2f", model.compute_mean_track_length())
    log.info("  mean reproj error : %.3f px", model.compute_mean_reprojection_error())
    log.info("  sparse PLY        : %s   (open in CloudCompare)", ply_path)
    log.info("  elapsed           : %.1fs", time.time() - t0)
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
