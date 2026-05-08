# CG-UFM Data Pipeline (`scripts/`)

This document describes how raw assets become `(x_0, x_1)` training pairs for the
CG-UFM flow-matching network. Run all commands from the project root.

```
Blender (.obj + 50–75 rendered views)
   │
   ├── (A)  Multi-view images ─► SfM ──► residual point cloud  x_0
   │                                     (datasets/sfm/<scene>/sparse/0/points3D.ply)
   │
   └── (B)  CAD mesh           ─► PDS + FPS ──► dense GT point cloud  x_1
                                                (datasets/gt/<scene>.pt)

   (A) + (B) ──► data/dataset.py ──► batched (x_0, x_1) tensors ──► train.py
```

`x_0` is the broken/noisy SfM output the network must repair.
`x_1` is the clean CAD-derived target the flow ends at.

---

## 0. Environment

```bash
conda activate corl
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
# COLMAP (apt build comes with CUDA support):
sudo apt install colmap
# Optional, for the learned-feature SfM path:
pip install hloc pycolmap
```

---

## 1. Input layout

Each scene is a single folder produced by Blender:

```
dataset_test/input/<scene_name>/
├── <scene_name>.obj          # CAD mesh (the GT source)
├── <scene_name>.mtl          # material, optional
├── transforms.json           # GT camera intrinsics/extrinsics from Blender
├── view_0000.png
├── view_0001.png
└── ...                       # 50–75 underwater-degraded multi-view renders
```

Non-image files (`.obj`, `.mtl`, `.json`) are filtered out automatically by both
SfM scripts via an `image_list.txt` whitelist.

---

## 2. Get `x_0` — SfM on the rendered views

You have two interchangeable paths. Output layout is identical in both cases:

```
<workspace>/
├── image_list.txt
├── run_sfm[_hloc]_<timestamp>.log
├── database.db                              # COLMAP only
└── sparse/0/
    ├── cameras.bin / images.bin / points3D.bin
    └── points3D.ply                         # ◄── this is x_0
```

### 2a. SIFT path (`scripts/run_sfm.sh`)

Classical COLMAP. Fast, deterministic, weak under heavy underwater scattering.

```bash
bash scripts/run_sfm.sh \
    dataset_test/input/Elbow_30deg_000 \
    dataset_test/result \
    --sparse-only \
    --fresh           # wipe stale database when re-running on new images
```

Flags:
- `--sparse-only` skips `patch_match_stereo` + fusion. Recommended for CG-UFM
  because the sparse cloud already exhibits the breakage we want to repair.
- `--fresh` is required whenever the input image set changes; otherwise COLMAP
  reuses cached SIFT features keyed on `(name, mtime, size)`.
- `--gpu 0` selects the CUDA device (default 0). Without CUDA, `--sparse-only`
  is mandatory: the dense step will fail.

### 2b. LoFTR (detector-free dense) path (`scripts/run_sfm_hloc.py`)

Recommended for the actual paper experiments. LoFTR (CVPR 2021) is a
detector-free dense matcher: it produces correspondences directly between
image pairs without first running a keypoint detector, so it survives the
low-texture and scattering-degraded regions where SuperPoint/SIFT under-sample.
This is the path used to generate `x_0` for CG-UFM.

```bash
python scripts/run_sfm_hloc.py \
    dataset_test/input/Elbow_30deg_000 \
    dataset_test/result_hloc \
    --fresh
```

Useful flags:
- `--matcher superglue` to fall back to SuperPoint + SuperGlue (sparse path).
- `--matcher superpoint+lightglue` for SuperPoint + LightGlue (ICCV 2023).
- `--feature disk` to swap SuperPoint for DISK features (CVPR 2020); only
  applies on the sparse path — ignored when `--matcher` is a dense matcher.
- `--relax` lowers COLMAP triangulation thresholds so weakly-matched scenes
  still produce a (noisier) sparse cloud — ideal for the broken `x_0` we want.

### 2c. Known-pose triangulation (`scripts/run_sfm_known_pose.py`)

**Recommended whenever `transforms.json` is available** (e.g., all our Blender
renders). Bypasses COLMAP's incremental SfM mapper entirely: reads GT camera
intrinsics + 75 c2w extrinsics from `transforms.json`, builds a posed-but-empty
`pycolmap.Reconstruction`, runs LoFTR matching, then calls
`pycolmap.triangulate_points` with the poses **frozen**. No bundle adjustment
on poses, so the structure cannot get scrambled by failed image registrations
or aggressive outlier filtering — all 75 frames contribute by construction.

```bash
python scripts/run_sfm_known_pose.py \
    dataset_test/input/Elbow_30deg_000 \
    dataset_test/result_known_pose \
    --fresh
```

Useful flags:
- `--relax` — **strongly recommended for this script**. Loosens triangulation
  thresholds (`min_angle 1.5°→0.5°`, reproj-error caps doubled, two-view
  tracks kept) AND skips hloc's essential-matrix geometric verification.
  Justification: with GT poses, the verification mostly throws away signal
  rather than catching errors, and stock thresholds were tuned for noisy
  incremental SfM. Typical effect: 5–10× more 3D points.
- `--num-matched K` (default 0 = exhaustive) — switch to
  `hloc.pairs_from_poses` and keep only the K most-covisible pairs per image.
  K=20 cuts pair count from 2775 → ~750 (3.7× speedup); K=15 → ~560 (5×).
- `--resize-max N` (default 1024) — longest image side fed to LoFTR.
  N=720 halves the per-pair matching cost; N=512 quarters it. Combine with
  `--num-matched 20` to bring an 18-min run under 3 min on a 4070 Laptop.
- `--apply-distortion` — switch the camera from PINHOLE to FULL_OPENCV using
  `k1/k2/k3/p1/p2` from `transforms.json`. Default OFF because our renders
  have `synthetic_is_pinhole: True`; turn on if the cloud looks "bowed" near
  image edges.
- `--matcher superglue` / `--matcher superpoint+lightglue` — sparse matchers
  also work here, identical CLI to §2b.
- `--max-keypoints 2048` — passed straight into `hloc.match_dense`. Bump for
  denser clouds at the cost of GPU memory and matching time.

Recommended starting point for a fresh scene:

```bash
python scripts/run_sfm_known_pose.py \
    dataset_test/input/Reducer_01 \
    dataset_test/result_known_pose \
    --fresh --relax --num-matched 20 --resize-max 720
```

Output layout is **identical** to §2b (`<workspace>/sparse/points3D.ply`),
so downstream code (`data/dataset.py`, alignment, etc.) does not branch.
The intermediate posed skeleton lives at `<workspace>/skeleton/` for
debugging.

When this is the wrong choice: real-data scenes without GT poses (no
`transforms.json`). Use §2b there, then run §4's alignment script.

### 2d. Inspecting the result

The summary block at the end of every log file reports:
```
registered images : 71 / 75
3D points         : 8423
mean track length : 4.2
mean reproj error : 1.21 px
sparse PLY        : .../sparse/0/points3D.ply
```

For visual inspection:
```bash
python -c "import open3d as o3d; \
    o3d.visualization.draw_geometries([o3d.io.read_point_cloud(\
    'dataset_test/result/sparse/0/points3D.ply')])"
```
(or drag the `.ply` into MeshLab / CloudCompare)

---

## 3. Get `x_1` — Poisson-Disk sampling from CAD

```bash
# single mesh
python scripts/cad_to_gt.py \
    dataset_test/input/Elbow_30deg_000/Elbow_30deg_000.obj \
    datasets/gt/

# whole tree of scenes
python scripts/cad_to_gt.py dataset_test/input/ datasets/gt/ --recursive
```

Key options:
- `-n 4096` — number of GT points (matches `generate_dummy_data.py` default).
- `--oversample 2.0` — PDS samples `2N` points then FPS down to `N`. PDS alone
  occasionally returns fewer points than requested; the FPS pass guarantees
  exact `N` and improves uniformity.
- `--normalize aabb` (default) — center to mean and scale by max abs coordinate
  so the cloud fits in `[-1, 1]^3`. Use `sphere` for unit-ball, `none` to keep
  Blender units. Same option must be applied (independently) to `x_0` so the two
  clouds live in comparable coordinates.
- `--max-faces 200000` — decimate dense meshes before PDS. PDS cost is
  proportional to face count, not point count.

Output schema (`datasets/gt/<scene>.pt`):
```python
{
    "gt_points":  (N, 3) float32,           # surface samples
    "gt_normals": (N, 3) float32,           # outward unit normals
    "mesh_name":  str,
    "normalize":  {"center": (3,), "scale": float} | None,
}
```

The `gt_points` key is intentionally identical to the field used by
`generate_dummy_data.py`, so `data/dataset.py` does not need to branch on
"real vs synthetic" data.

---

## 4. Aligning `x_0` and `x_1`

SfM and CAD live in different coordinate frames. Two options, in order of
preference:

**Path A (Blender renders): use §2c.** `run_sfm_known_pose.py` triangulates
under the GT poses from `transforms.json`, so the resulting cloud is born in
the CAD frame. **No alignment step needed.**

**Path B (anything else): `scripts/align_sfm_to_cad.py`.** Solves a 7-DoF
Sim(3) (Umeyama) between SfM camera centers and `transforms.json` camera
centers, then applies it to the entire reconstruction (points + images).
Useful for an existing `run_sfm_hloc.py` workspace, or for future real-data
scenes where you ran a separate calibration that produced a transforms-style
JSON.

```bash
python scripts/align_sfm_to_cad.py \
    dataset_test/result_hloc \
    dataset_test/input/Elbow_30deg_000/transforms.json
# writes <ws>/sparse_aligned/{cameras,images,points3D}.bin + points3D.ply
```

The log reports an alignment **RMSE** in Blender world units. A healthy fit
has RMSE well below `object_bounding_radius` (≈0.55 for our elbows). If the
RMSE is larger than the bounding radius, the SfM is so broken that even
Sim(3) cannot rescue it — go back to §2c.

If neither `transforms.json` nor a known-pose path is available (truly
uncalibrated real data), fall back to ICP
(`open3d.pipelines.registration.registration_icp`) after independent AABB
normalisation of both clouds.

---

## 5. Filtering background out of `x_0`

Known-pose SfM is honest about *where* the points are, but LoFTR will
happily latch onto the textured pool floor / wall behind the object and
triangulate them too. Those points pollute the mass-killing gradient signal
of the UFM survival head, so they need to go before training -- but only the
**obviously far** ones. Near-object ghost mass should stay so the survival
head has training signal for what it's actually meant to learn.

```bash
python scripts/filter_sfm_background.py dataset_test/result_known_pose
# writes <workspace>/sparse_filtered/points3D.ply  (this is x_0)
#        <workspace>/sparse_filtered/points3D_colored.ply  (green=kept, red=dropped)
#        <workspace>/sparse_filtered/filter_stats.json
```

Knobs:
- `--radius-factor 2.0` (default) — keep points within `K * object_bounding_radius`
  of `object_center` (both read from `transforms.json`). Tighten to 1.5 for
  a cleaner cloud at the cost of mass-head training signal; loosen to 2.5
  to give the survival head more obvious "kill me" examples.
- `--scene-dir PATH` — only needed if the workspace's most recent run log
  cannot be parsed automatically.
- `--min-points 512` — warns if the kept count drops below the dataset.py
  x_0 target (which would force `random sampling-with-replacement` and
  duplicate points in training).

What this does **NOT** do, deliberately: density clustering, surface
projection, mass redistribution. UFM's dual-head ODE is supposed to learn
all of that. This script just trims the most obvious signal-drowning bulk.

---

## 6. Compose the training-ready `.pt`

Stitches a filtered SfM cloud (`x_0`) and a CAD-PDS cloud (`x_1`) into a
single `.pt` whose schema matches `generate_dummy_data.py` and
`data/dataset.py`. Most importantly: re-applies the CAD's AABB normalize
affine to the SfM cloud so `x_0` and `x_1` stay aligned post-normalization.

```bash
python scripts/ply_to_pt.py \
    dataset_test/result_known_pose \
    datasets/gt/Elbow_30deg_004.pt \
    datasets/training/Elbow_30deg_004.pt
```

Output schema:
```python
{
    "noisy_points": (M, 3) float32,   # filtered SfM cloud (post-normalize)
    "features":     (M, 6) float32,   # per-point feature vector (default: random noise)
    "gt_points":    (K, 3) float32,   # CAD-sampled cloud (post-normalize)
}
```

Knobs:
- `--features {random, zeros, rgb_pad6}` — default `random` matches the
  dummy generator (so the model that's been trained on dummy still sees
  the same feature distribution). `rgb_pad6` swaps in PLY colors padded
  with zeros for later experiments where you want photometric signal in
  the consensus head.
- `--use-raw` — read `sparse/points3D.ply` instead of `sparse_filtered/`.
  Skips background filtering; use only for debugging.
- `--seed 0` — RNG for `--features random`; frozen at .pt write time, so
  this only affects which exact noise lives in the file.

`data/dataset.py` will FPS-down/upsample to 512 / 4096 at load time, so
native point counts in the `.pt` can vary scene-to-scene as long as both
are above target (the script warns when they aren't).

For batch generation, just bash-loop:
```bash
for scene in datasets/sim_dataset/elbow/Elbow_30deg_*; do
    name=$(basename "$scene")
    python scripts/run_sfm_known_pose.py "$scene" "outputs/sfm/$name" --fresh
    python scripts/filter_sfm_background.py "outputs/sfm/$name"
    python scripts/cad_to_gt.py "$scene/$name.obj" datasets/gt/
    python scripts/ply_to_pt.py "outputs/sfm/$name" \
        "datasets/gt/$name.pt" "datasets/training/$name.pt"
done
```

---

## 7. Training & inference

```bash
python train.py                     # logs to W&B, checkpoints under outputs/
python inference.py                 # exports input/output .ply pairs for viewing
```

Both scripts read from `datasets/dummy_dataset/` by default. Point them at
`datasets/training/` (the `ply_to_pt.py` output dir) for real data.

---

## 8. Quick reference

| Script                          | Purpose                                       | Output                                           |
| ---                             | ---                                           | ---                                              |
| `generate_dummy_data.py`        | Synthetic broken cylinders                    | `datasets/dummy_dataset/patch_*.pt`              |
| `scripts/run_sfm.sh`            | COLMAP SIFT SfM                               | `<ws>/sparse/0/points3D.ply` + log               |
| `scripts/run_sfm_hloc.py`       | LoFTR detector-free SfM (incremental mapper)  | `<ws>/sparse/0/points3D.ply` + log               |
| `scripts/run_sfm_known_pose.py` | Known-pose triangulation (Blender)            | `<ws>/sparse/points3D.ply` + log                 |
| `scripts/align_sfm_to_cad.py`   | Sim(3) align SfM workspace to CAD frame       | `<ws>/sparse_aligned/points3D.ply` + log         |
| `scripts/filter_sfm_background.py` | Sphere-clip background out of x_0          | `<ws>/sparse_filtered/points3D.ply` + log        |
| `scripts/cad_to_gt.py`          | Poisson Disk sample CAD mesh                  | `<out>/<mesh>.pt` + log                          |
| `scripts/ply_to_pt.py`          | Merge filtered SfM + CAD GT into training .pt | `<out>/<scene>.pt` matching dataset.py schema    |

---

## 9. TODO

- [x] `scripts/ply_to_pt.py` — convert SfM `points3D.ply` into the dataset .pt
      schema, applying CAD's normalize affine to keep x_0/x_1 aligned.
- [x] `scripts/align_sfm_to_cad.py` — Sim(3) alignment using `transforms.json`,
      so `x_0` and `x_1` share a frame for training. (Plus `run_sfm_known_pose.py`
      for the Blender path that skips alignment entirely.)
- [ ] Replace the CPU FPS in `cad_to_gt.py` with a CUDA implementation once N > 16k.
- [ ] Real per-point features in `ply_to_pt.py` (track length, reproj error,
      multi-view consistency) instead of the current random/RGB placeholder.
