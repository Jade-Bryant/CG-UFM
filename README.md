# CG-UFM: Consensus-Guided Unbalanced Flow Matching for Underwater 3D Reconstruction

This is the PyTorch implementation of our paper targeting CoRL 2026. 

CG-UFM is a point cloud optimization network designed for underwater fine-framed structures. The method uses Unbalanced Optimal Transport (UOT) combined with Continuous Normalizing Flows (CNF) to deform noisy Structure-from-Motion (SfM) point clouds into clean, measurement-grade models.

## Installation

We recommend using Conda to manage your environment. The project requires Python `>=3.10`.

```bash
conda create -n corl python=3.10
conda activate corl
```

### 1. Install PyTorch

To ensure compatibility with your hardware (especially CUDA versions), please install **PyTorch** and its related packages manually **before** installing the rest of the project dependencies. 

We recommend PyTorch `>= 2.0.0` for optimal performance.

**For CUDA 11.8:**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

*(You can find the installation command for your specific setup on the [official PyTorch website](https://pytorch.org/get-started/locally/).)*

### 2. Install Other Dependencies

Once PyTorch is installed, you can install the rest of the required packages and the `corl` package in editable mode:

```bash
pip install -e .
```

### 3. Install torchdiffeq (ODE Solver)

The ODE solver depends on PyTorch. We recommend version `>= 0.2.3`:

```bash
pip install torchdiffeq>=0.2.3
```

## Directory Structure

- `core/`: Core mathematical modules (Flow Matching Loss, ODE Solver, Nadaraya-Watson Aggregation).
- `data/`: Data loading pipelines and transforms (`dataset.py` does FPS to 512/4096 at load time).
- `models/`: Network architectures (Backbones, Modules, and the main CUFM Net).
- `metrics/`: `E_topo` evaluation (skeleton extraction + caliper measurements).
- `scripts/`: Data preparation pipeline — SfM, CAD sampling, alignment, filtering, `.pt` assembly.
- `generate_dummy_data.py`: Synthetic broken-cylinder generator for sanity training.
- `train.py`: Main training script.
- `inference.py`: Inference pipeline script.

## Getting Started

### Quick path with synthetic data

```bash
python generate_dummy_data.py        # writes datasets/dummy_dataset/patch_*.pt
python train.py                       # reads ./datasets/dummy_dataset by default
```

### Real-data pipeline

End-to-end recipe is documented in [`scripts/README.md`](scripts/README.md).
Per-scene short version (Blender renders + `transforms.json` + `.obj`):

```bash
SCENE=dataset_test/input/Elbow_30deg_004
NAME=$(basename "$SCENE")
WS=outputs/sfm/$NAME

python scripts/run_sfm_known_pose.py "$SCENE" "$WS" --fresh    # known-pose triangulation (LoFTR)
python scripts/filter_sfm_background.py "$WS"                   # mild sphere clip of far background
python scripts/cad_to_gt.py "$SCENE/$NAME.obj" datasets/gt/    # Poisson-disk sample CAD -> x_1
python scripts/ply_to_pt.py "$WS" "datasets/gt/$NAME.pt" \
    "datasets/training/$NAME.pt"                                # merge x_0 + x_1 into one .pt
```

Then point `train.py:82` at `./datasets/training` and train.

### Important caveats (read before regenerating data)

- **`transforms.json` convention**: in our Blender exporter, `transform_matrix`
  is OpenGL **w2c**, not c2w (despite the NeRF-style field name). The known-pose
  script handles this internally; if you swap exporters, re-run the projection
  sanity sweep before trusting the convention.
- **LoFTR is texture-hungry**: smooth/uniform pipe surfaces cause LoFTR to
  latch onto the background instead. The renderer must give surfaces a
  texture/normal-map (brushed metal, biofouling, etc.). If you see a SfM
  cloud where >90% of points sit on the floor/wall, this is the cause —
  the filter cannot recover signal that was never matched.
- **Coordinate alignment is brittle**: the data generator must export `.obj`
  AFTER applying any per-scene world transforms used during rendering, OR
  store that transform in `transforms.json` for downstream to apply. A
  silent mismatch makes the network learn a useless global SE(3) instead of
  shape repair (see `scripts/README.md` §4 for verification).
