"""Known-pose triangulation for CG-UFM Blender datasets.

Bypasses COLMAP's incremental SfM mapper. Reads GT camera intrinsics and
75 c2w extrinsics from `transforms.json`, builds a `pycolmap.Reconstruction`
skeleton with the poses fixed, runs LoFTR matching via hloc, then calls
`hloc.triangulation.main` (which uses `pycolmap.triangulate_points`) to
produce 3D points under the fixed poses. No bundle adjustment on poses.

Use this when:
    - Your renders come from Blender with a `transforms.json` next to them.
    - The standard incremental mapper (run_sfm_hloc.py) is producing a
      mangled cloud where the structure is no longer recognisable.

Output layout matches scripts/run_sfm_hloc.py so downstream code does not
change:

    <workspace>/
        image_list.txt
        run_sfm_known_pose_<ts>.log
        skeleton/{cameras,frames,images,points3D,rigs}.bin
        features.h5
        matches.h5
        pairs-exhaustive.txt
        sparse/{cameras,frames,images,points3D,rigs}.bin
        sparse/points3D.ply

Usage:
    python scripts/run_sfm_known_pose.py <image_dir> <workspace> [--fresh]
    # image_dir must contain transforms.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Blender/OpenGL camera convention: +X right, +Y up, -Z forward (look at -Z).
# COLMAP camera convention:          +X right, +Y down, +Z forward.
# Conversion (camera-side axis flip): Y, Z negated.
#
# IMPORTANT: in our dataset's transforms.json, `transform_matrix` is the
# Blender-OpenGL **w2c** (despite the NeRF-style key name suggesting c2w).
# Verified empirically by projecting the .obj vertices: under "treat as w2c
# and apply the camera-axis flip" all 75 frames have all vertices in front of
# the camera, and ~25% land inside each 1280x720 image -- consistent with the
# scene geometry. Treating it as c2w (with or without flip) puts >95% of
# vertices behind the camera or off-screen.
BLENDER_TO_COLMAP_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_dir", type=Path, help="folder with images and transforms.json")
    p.add_argument("workspace", type=Path)
    p.add_argument("--fresh", action="store_true", help="wipe workspace before running")
    p.add_argument(
        "--matcher", default="loftr",
        help="hloc matcher config name (default: loftr). Sparse alternatives: superglue, "
             "superpoint+lightglue.",
    )
    p.add_argument(
        "--feature", default="superpoint_aachen",
        help="hloc feature config name; only used if --matcher is a sparse matcher",
    )
    p.add_argument(
        "--apply-distortion", action="store_true",
        help="use FULL_OPENCV camera with k1/k2/k3/p1/p2 from transforms.json. "
             "Default is PINHOLE because synthetic_is_pinhole=True for our Blender renders.",
    )
    p.add_argument(
        "--max-keypoints", type=int, default=2048,
        help="max_kps cap passed into hloc.match_dense (default 2048)",
    )
    p.add_argument(
        "--resize-max", type=int, default=720,
        help="longest image side fed to the matcher (default 1024). 720 ~halves matching "
             "time at the cost of slightly fewer matches per pair; 512 quarters it.",
    )
    p.add_argument(
        "--num-matched", type=int, default=20,
        help="if > 0, use hloc.pairs_from_poses to pick top-K most-covisible pairs per image "
             "instead of exhaustive (2775 pairs for 75 imgs). 20 is a sweet spot, drops pair "
             "count ~3.7x. Requires the skeleton to be built first (always is, here).",
    )
    p.add_argument(
        "--relax", action="store_true",
        help="loosen triangulation thresholds AND skip hloc geometric verification. With GT "
             "poses, the essential-matrix verification mostly throws away signal, and tighter "
             "tri thresholds were tuned for noisy poses. Strongly recommended for known-pose "
             "triangulation -- typically 5-10x more 3D points.",
    )
    return p.parse_args()


# Triangulation thresholds for fixed-pose mode. Stock COLMAP defaults are tuned
# for noisy incremental SfM; with GT poses we want to keep more tracks.
RELAXED_TRI_OPTIONS: dict = {
    "triangulation": {
        "min_angle":                 0.5,    # default 1.5  (deg) -- keep narrow-baseline pairs
        "create_max_angle_error":    4.0,    # default 2.0
        "continue_max_angle_error":  4.0,    # default 2.0
        "merge_max_reproj_error":    8.0,    # default 4.0
        "complete_max_reproj_error": 8.0,    # default 4.0
        "ignore_two_view_tracks":    False,  # default True -- KEEP 2-view tracks
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


def load_blender_transforms(json_path: Path) -> dict:
    """Read transforms.json. Returns dict with intrinsics + per-frame {name, w2c_blender}."""
    if not json_path.is_file():
        raise SystemExit(f"transforms.json not found at {json_path}")
    raw = json.loads(json_path.read_text())

    intr = raw["camera_intrinsics"]
    width, height = intr["resolution"]
    out = {
        "width":  int(width),
        "height": int(height),
        "fx":     float(intr["fx"]),
        "fy":     float(intr["fy"]),
        "cx":     float(intr["cx"]),
        "cy":     float(intr["cy"]),
        "synthetic_is_pinhole": bool(intr.get("synthetic_is_pinhole", False)),
        "distortion": intr.get("distortion") or {},
        "frames": [],
    }
    for f in raw["frames"]:
        M = np.asarray(f["transform_matrix"], dtype=np.float64)  # (4, 4) -- OpenGL w2c
        if M.shape != (4, 4):
            raise SystemExit(f"frame {f.get('file_path')} has bad transform_matrix shape {M.shape}")
        out["frames"].append({
            "name":         Path(f["file_path"]).name,   # tolerate path prefixes
            "w2c_blender":  M,
        })
    return out


def blender_w2c_to_colmap_w2c(w2c_blender: np.ndarray) -> np.ndarray:
    """Blender(OpenGL) w2c (4,4) -> COLMAP w2c (3,4). Flip Y and Z on the camera side."""
    return (BLENDER_TO_COLMAP_CAM @ w2c_blender)[:3, :]   # (3, 4)


def build_reconstruction_skeleton(
    transforms: dict,
    image_names: list[str],
    apply_distortion: bool,
    log: logging.Logger,
) -> "pycolmap.Reconstruction":
    """Construct a Reconstruction with one camera + N posed images, no 3D points."""
    import pycolmap

    if apply_distortion:
        d = transforms["distortion"]
        # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        params = [
            transforms["fx"], transforms["fy"], transforms["cx"], transforms["cy"],
            float(d.get("k1", 0.0)), float(d.get("k2", 0.0)),
            float(d.get("p1", 0.0)), float(d.get("p2", 0.0)),
            float(d.get("k3", 0.0)), 0.0, 0.0, 0.0,
        ]
        camera = pycolmap.Camera(
            model="FULL_OPENCV", width=transforms["width"], height=transforms["height"], params=params,
        )
        log.info("[skeleton] camera model: FULL_OPENCV   k1=%.3f k2=%.3f k3=%.3f p1=%.4f p2=%.4f",
                 *params[4:9])
    else:
        params = [transforms["fx"], transforms["fy"], transforms["cx"], transforms["cy"]]
        camera = pycolmap.Camera(
            model="PINHOLE", width=transforms["width"], height=transforms["height"], params=params,
        )
        log.info("[skeleton] camera model: PINHOLE   fx=%.2f fy=%.2f cx=%.2f cy=%.2f", *params)
    camera.camera_id = 1

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(camera)

    name_set = set(image_names)
    posed = 0
    for i, frame in enumerate(transforms["frames"], start=1):
        if frame["name"] not in name_set:
            log.warning("[skeleton] transforms.json references %s but no such image; skipped",
                        frame["name"])
            continue
        w2c = blender_w2c_to_colmap_w2c(frame["w2c_blender"])   # (3, 4)
        cam_from_world = pycolmap.Rigid3d(w2c)
        img = pycolmap.Image(name=frame["name"], camera_id=1, image_id=i)
        rec.add_image_with_trivial_frame(img, cam_from_world)
        posed += 1

    missing = name_set - {f["name"] for f in transforms["frames"]}
    if missing:
        log.warning("[skeleton] %d images have no pose in transforms.json: %s",
                    len(missing), sorted(missing)[:5])

    log.info("[skeleton] cameras=%d  rigs=%d  posed_images=%d", rec.num_cameras(), rec.num_rigs(), posed)
    if posed == 0:
        raise SystemExit("Skeleton has zero posed images. Check that frame file_paths match images on disk.")
    return rec


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
    log_path = workspace / f"run_sfm_known_pose_{ts}.log"
    setup_logging(log_path)
    log = logging.getLogger("run_sfm_known_pose")

    image_names = collect_images(image_dir)
    image_list_path = workspace / "image_list.txt"
    image_list_path.write_text("\n".join(image_names) + "\n")

    transforms_path = image_dir / "transforms.json"
    transforms = load_blender_transforms(transforms_path)

    log.info("=" * 64)
    log.info("CG-UFM Known-Pose Triangulation")
    log.info("  image_dir : %s", image_dir)
    log.info("  workspace : %s", workspace)
    log.info("  transforms: %s", transforms_path)
    log.info("  num_images: %d   num_frames: %d", len(image_names), len(transforms["frames"]))
    log.info("  matcher   : %s   resize_max=%d", args.matcher, args.resize_max)
    log.info("  distortion: %s   (synthetic_is_pinhole=%s)",
             "ON" if args.apply_distortion else "OFF (PINHOLE)",
             transforms["synthetic_is_pinhole"])
    log.info("  pairs     : %s",
             "exhaustive" if args.num_matched <= 0 else f"pose-covisibility top-{args.num_matched}")
    log.info("  relax     : %s   (loosened thresholds, geom-verif %s)",
             "ON" if args.relax else "OFF",
             "skipped" if args.relax else "enabled")
    log.info("  log       : %s", log_path)
    log.info("=" * 64)

    if not args.apply_distortion and not transforms["synthetic_is_pinhole"]:
        log.warning("[skeleton] synthetic_is_pinhole is False but --apply-distortion is OFF; "
                    "consider passing --apply-distortion if PLY looks bowed at image edges")

    # Imports deferred so that --help works without hloc installed.
    try:
        from hloc import (
            extract_features,
            match_dense,
            match_features,
            pairs_from_exhaustive,
            pairs_from_poses,
            triangulation,
        )
    except ImportError as e:
        raise SystemExit(
            f"hloc import failed ({e}). Install with:\n"
            "    git clone --recursive https://github.com/cvg/Hierarchical-Localization.git ~/tools/hloc\n"
            "    pip install -e ~/tools/hloc\n"
            "    pip install pycolmap"
        )

    is_dense = args.matcher in match_dense.confs                # LoFTR-family path
    if is_dense:
        matcher_conf = copy.deepcopy(match_dense.confs[args.matcher])
        # Override the matcher's preprocessing.resize_max to take effect on LoFTR input size.
        matcher_conf.setdefault("preprocessing", {})["resize_max"] = args.resize_max
        # Bump output name when resize differs from default so caches don't collide.
        if args.resize_max != 1024:
            matcher_conf["output"] = f"{matcher_conf['output']}-r{args.resize_max}"
        feature_conf = None
        log.info("[detector-free] using dense matcher: %s   resize_max=%d", args.matcher, args.resize_max)
    else:
        feature_conf = copy.deepcopy(extract_features.confs[args.feature])
        matcher_conf = copy.deepcopy(match_features.confs[args.matcher])

    skeleton_dir = workspace / "skeleton"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    rec = build_reconstruction_skeleton(transforms, image_names, args.apply_distortion, log)
    rec.write(str(skeleton_dir))
    log.info("[skeleton] written to %s", skeleton_dir)

    if args.num_matched > 0:
        sfm_pairs = workspace / f"pairs-poses-top{args.num_matched}.txt"
    else:
        sfm_pairs = workspace / "pairs-exhaustive.txt"
    sfm_dir = workspace / "sparse"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def build_pairs() -> None:
        """Write `sfm_pairs` as either exhaustive or pose-based covisibility top-K."""
        if args.num_matched > 0:
            log.info("Building pose-covisibility pairs (top-%d per image)", args.num_matched)
            pairs_from_poses.main(skeleton_dir, sfm_pairs, args.num_matched)
        else:
            n_pairs = len(image_names) * (len(image_names) - 1) // 2
            log.info("Building exhaustive image pairs (%d images -> %d pairs)",
                     len(image_names), n_pairs)
            pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)

    if is_dense:
        log.info("[1/3] %s", "pose-covisibility pairs" if args.num_matched > 0 else "exhaustive pairs")
        build_pairs()

        log.info("[2/3] Dense matching with %s", args.matcher)
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
            feature_conf, image_dir, workspace, image_list=image_names,
        )

        log.info("[2/4] %s", "pose-covisibility pairs" if args.num_matched > 0 else "exhaustive pairs")
        build_pairs()

        log.info("[3/4] Matching with %s", args.matcher)
        match_path = match_features.main(
            matcher_conf, sfm_pairs, feature_conf["output"], workspace,
        )

    last_step = "[3/3]" if is_dense else "[4/4]"
    log.info("%s Triangulation under fixed poses (pycolmap.triangulate_points)", last_step)
    if args.relax:
        log.info("  [relax] looser triangulation thresholds + geom-verif skipped")
        for sk, sv in RELAXED_TRI_OPTIONS["triangulation"].items():
            log.info("    triangulation.%-28s = %s", sk, sv)
    model = triangulation.main(
        sfm_dir,
        skeleton_dir,
        image_dir,
        sfm_pairs,
        feature_path,
        match_path,
        skip_geometric_verification=args.relax,
        mapper_options=RELAXED_TRI_OPTIONS if args.relax else None,
        verbose=False,
    )

    if model is None or model.num_points3D() == 0:
        log.error("Triangulation produced zero 3D points.")
        log.error("With fixed poses, this almost always means matches couldn't be back-projected:")
        log.error("  1. Coordinate convention mismatch -- check blender_w2c_to_colmap_w2c output")
        log.error("  2. Wrong intrinsics (e.g. cx/cy swapped, or distortion mis-applied)")
        log.error("  3. Image pixel coords vs camera coords convention drift")
        log.error("  4. Try --relax to skip geometric verification and loosen tri thresholds")
        return 2

    ply_path = sfm_dir / "points3D.ply"
    n_pts = export_ply(model, ply_path)

    log.info("=" * 64)
    log.info("Summary")
    log.info("  registered images : %d / %d  (all fixed-pose)", model.num_reg_images(), len(image_names))
    log.info("  3D points         : %d", n_pts)
    log.info("  mean track length : %.2f", model.compute_mean_track_length())
    log.info("  mean reproj error : %.3f px", model.compute_mean_reprojection_error())
    log.info("  sparse PLY        : %s   (open in CloudCompare)", ply_path)
    log.info("  elapsed           : %.1fs", time.time() - t0)
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
