# CG-UFM: Consensus-Guided Unbalanced Flow Matching for Underwater 3D Reconstruction

This is the PyTorch implementation of our paper targeting CoRL 2026. 

CG-UFM is a point cloud optimization network designed for underwater fine-framed structures. The method uses Unbalanced Optimal Transport (UOT) combined with Continuous Normalizing Flows (CNF) to deform noisy Structure-from-Motion (SfM) point clouds into clean, measurement-grade models.

## Installation

We recommend using Conda to manage your environment. The project requires Python `>=3.10`.

```bash
conda create -n cufm python=3.10
conda activate cufm
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

- `CG_UFM/config/`: Configuration files and hyperparameters.
- `CG_UFM/core/`: Core mathematical modules (Flow Matching Loss, ODE Solver, Nadaraya-Watson Aggregation).
- `CG_UFM/data/`: Data loading pipelines and transforms.
- `CG_UFM/models/`: Network architectures (Backbones, Modules, and the main CUFM Net).
- `train.py`: Main training script.
- `inference.py`: Inference pipeline script.

## Getting Started

*(Add instructions for preparing the dataset and running `train.py` here...)*
