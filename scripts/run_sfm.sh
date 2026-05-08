#!/usr/bin/env bash
# COLMAP SfM + MVS pipeline for CG-UFM underwater datasets.
#
# Usage:
#   bash scripts/run_sfm.sh <image_dir> <workspace_dir> [--sparse-only] [--gpu 0]
#
# Example:
#   bash scripts/run_sfm.sh datasets/raw/pipe_01/images datasets/sfm/pipe_01
#
# Outputs (under <workspace_dir>):
#   database.db
#   sparse/0/{cameras,images,points3D}.bin   # sparse SfM result
#   sparse/0/points3D.ply                    # sparse cloud (always exported)
#   dense/fused.ply                          # dense fused cloud (skipped if --sparse-only)
#
# The sparse cloud is usually noisier and more "broken" -- closer to the
# x_0 that CG-UFM is designed to repair. The dense cloud is denser but
# can over-smooth thin structures. Try both.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <image_dir> <workspace_dir> [--sparse-only] [--gpu <id>]" >&2
    exit 1
fi

IMAGE_DIR="$(realpath "$1")"
WORKSPACE="$(realpath -m "$2")"
shift 2

SPARSE_ONLY=0
GPU_INDEX=0
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sparse-only) SPARSE_ONLY=1; shift ;;
        --gpu)         GPU_INDEX="$2"; shift 2 ;;
        --fresh)       FRESH=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Image dir not found: $IMAGE_DIR" >&2
    exit 1
fi

mkdir -p "$WORKSPACE/sparse"
DB="$WORKSPACE/database.db"

# Build a whitelist of image files so non-image stuff (.obj/.mtl/.json...)
# in the same folder is never fed to COLMAP. Path is relative to IMAGE_DIR
# because COLMAP's --image_list_path expects relative names.
IMAGE_LIST="$WORKSPACE/image_list.txt"
( cd "$IMAGE_DIR" && find . -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
       -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.bmp' \) \
    | sed 's|^\./||' | sort ) > "$IMAGE_LIST"
NUM_IMAGES=$(wc -l < "$IMAGE_LIST")

if [[ $FRESH -eq 1 && -f "$DB" ]]; then
    echo "[--fresh] Removing stale database: $DB"
    rm -f "$DB"
    rm -rf "$WORKSPACE/sparse"
    mkdir -p "$WORKSPACE/sparse"
fi

# Detect a stale database from a previous run on a different image set.
# Symptom you saw: "SKIP: Features for image already extracted" because
# COLMAP keys cache by (name, mtime, size) inside this DB file.
if [[ -f "$DB" ]]; then
    DB_NUM=$(sqlite3 "$DB" 'SELECT COUNT(*) FROM images;' 2>/dev/null || echo 0)
    if [[ "$DB_NUM" != "$NUM_IMAGES" ]]; then
        echo "WARNING: database.db has $DB_NUM images but image_list has $NUM_IMAGES."
        echo "         Re-run with --fresh to wipe the workspace, otherwise stale"
        echo "         features from a previous run will be reused."
        exit 3
    fi
fi

LOG_FILE="$WORKSPACE/run_sfm_$(date +%Y%m%d_%H%M%S).log"
# Mirror everything (stdout + stderr) to both terminal and the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "CG-UFM SfM run"
echo "  image_dir : $IMAGE_DIR"
echo "  workspace : $WORKSPACE"
echo "  gpu       : $GPU_INDEX"
echo "  sparse_only: $SPARSE_ONLY"
echo "  fresh     : $FRESH"
echo "  num_images: $NUM_IMAGES (whitelist: $IMAGE_LIST)"
echo "  started   : $(date -Iseconds)"
echo "  log       : $LOG_FILE"
echo "================================================================"

START_TS=$(date +%s)

echo "[1/6] Feature extraction (SIFT, GPU $GPU_INDEX)"
# single_camera=1: 75 frames from one virtual rig share intrinsics, which
# stabilises BA on textureless underwater frames.
colmap feature_extractor \
    --image_list_path "$IMAGE_LIST" \
    --database_path "$DB" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index "$GPU_INDEX" \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

echo "[2/6] Exhaustive matching (75 views -> ~2.8k pairs, fine)"
colmap exhaustive_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index "$GPU_INDEX" \
    --SiftMatching.guided_matching 1

echo "[3/6] Sparse reconstruction (incremental mapper)"
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGE_DIR" \
    --output_path "$WORKSPACE/sparse" \
    --Mapper.ba_refine_principal_point 1

# Pick model 0 (the largest) as the canonical sparse model.
SPARSE_MODEL="$WORKSPACE/sparse/0"
if [[ ! -f "$SPARSE_MODEL/points3D.bin" ]]; then
    echo "Sparse reconstruction failed -- no model produced." >&2
    echo "Underwater scattering likely killed too many features." >&2
    echo "Try HLoc + LoFTR (scripts/run_sfm_hloc.py) instead." >&2
    exit 2
fi

echo "[4/6] Export sparse cloud as PLY"
colmap model_converter \
    --input_path "$SPARSE_MODEL" \
    --output_path "$SPARSE_MODEL/points3D.ply" \
    --output_type PLY

print_summary() {
    echo
    echo "================================================================"
    echo "Summary"
    echo "================================================================"
    colmap model_analyzer --path "$SPARSE_MODEL" 2>&1 | sed 's/^/  /'
    if [[ -f "$SPARSE_MODEL/points3D.ply" ]]; then
        local n_sparse
        n_sparse=$(grep -m1 '^element vertex' "$SPARSE_MODEL/points3D.ply" | awk '{print $3}')
        echo "  sparse PLY points : ${n_sparse:-?}  ($SPARSE_MODEL/points3D.ply)"
    fi
    if [[ -f "$WORKSPACE/dense/fused.ply" ]]; then
        local n_dense
        n_dense=$(grep -m1 '^element vertex' "$WORKSPACE/dense/fused.ply" | awk '{print $3}')
        echo "  dense  PLY points : ${n_dense:-?}  ($WORKSPACE/dense/fused.ply)"
    fi
    local elapsed=$(( $(date +%s) - START_TS ))
    echo "  elapsed           : ${elapsed}s"
    echo "  finished          : $(date -Iseconds)"
    echo "================================================================"
}

echo "Sparse cloud: $SPARSE_MODEL/points3D.ply"

if [[ $SPARSE_ONLY -eq 1 ]]; then
    print_summary
    echo "Done (sparse only)."
    exit 0
fi

mkdir -p "$WORKSPACE/dense"

echo "[5/6] Image undistortion -> dense workspace"
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_MODEL" \
    --output_path "$WORKSPACE/dense" \
    --output_type COLMAP \
    --max_image_size 2000

echo "[6/6] PatchMatch stereo + fusion"
colmap patch_match_stereo \
    --workspace_path "$WORKSPACE/dense" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.gpu_index "$GPU_INDEX"

colmap stereo_fusion \
    --workspace_path "$WORKSPACE/dense" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$WORKSPACE/dense/fused.ply"

echo "Dense cloud: $WORKSPACE/dense/fused.ply"
print_summary
echo "Done."
