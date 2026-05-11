"""End-to-end CG-UFM pipeline: SfM + PDS + assemble + train + infer + benchmark.

Reads scenes from `sim_dataset/{elbow,reducer,t_joint,y_joint}/<scene>/`,
processes every scene through known-pose SfM, background filter, CAD PDS,
and `.pt` assembly. A deterministic per-category 9:1 stratified split decides
which scenes feed `train.py`-equivalent training and which become the held-out
test set. After training, the best checkpoint is run on the test set,
predictions are exported as `.ply`, and `benchmark.py` produces the final
metric table.

Output layout under --output-root (default: outputs/)
    sfm/<category>/<scene>/                  # SfM workspaces
    gt/<scene>.pt                            # PDS GT (cad_to_gt.py output)
    assembled/<scene>.pt                     # ply_to_pt.py merged tensors
    training/<scene>.pt -> ../assembled/...  # symlinks for train split
    test_input/<scene>.pt -> ../assembled/...# symlinks for test split
    weights/{best,latest}_model.pth          # checkpoints
    results/Ours/<scene>.ply                 # predicted clouds
    test_gt/<scene>.ply                      # GT clouds for benchmark
    benchmark/{evaluation_results.csv,evaluation_table.tex}
    split.json                               # scene -> {train,test} record
    pipeline.log                             # full run log

Usage examples
    # full run (all 120 scenes, ~hours)
    python scripts/run_full_pipeline.py

    # quick smoke test (4 scenes total, train ~5min)
    python scripts/run_full_pipeline.py --limit 1 --epochs 5

    # only re-run training and downstream stages with cached SfM/PDS
    python scripts/run_full_pipeline.py --skip-existing --stages train infer benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CATEGORIES = ["elbow", "reducer", "straight_pipe", "t_joint", "y_joint"]
ALL_STAGES = ["sfm", "gt", "assemble", "train", "infer", "benchmark"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--input-root", type=Path, default=_ROOT / "sim_dataset")
    p.add_argument("--output-root", type=Path, default=_ROOT / "outputs")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--limit", type=int, default=0,
                   help="Per-category scene cap (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip per-scene stages whose output already exists.")
    p.add_argument("--overwrite", action="store_true",
                   help="Force regeneration of outputs that already exist "
                        "(GT, assembled .pt, etc.). Opposite of --skip-existing.")
    p.add_argument("--reuse-split", action="store_true",
                   help="If split.json already exists, reuse its train/test "
                        "assignment instead of regenerating from --seed. "
                        "Useful when adding a new category without disturbing "
                        "existing splits, or to keep checkpoint comparability.")
    p.add_argument("--clean-results", action="store_true",
                   help="Force-delete outputs/results/Ours/, outputs/test_gt/, "
                        "and split-mismatched outputs/assembled/*.pt before running. "
                        "Use this when reusing an output dir across different "
                        "splits to prevent benchmark.py from picking up stale "
                        "predictions / GT clouds from a previous run.")
    p.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)

    # SfM speed knobs (forwarded to run_sfm_known_pose.py)
    p.add_argument("--num-matched", type=int, default=20)
    p.add_argument("--resize-max", type=int, default=720)
    p.add_argument("--no-relax", action="store_true",
                   help="Drop --relax (default uses it; recommended for known-pose).")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-every", type=int, default=20,
                   help="Run validation every N epochs (CD_norm + F-Score@2%% on "
                        "a held-out subset of train). 0 disables validation. "
                        "Validation is what selects best_model.pth — without it "
                        "we'd pick the ckpt with the lowest training loss, which "
                        "doesn't track geometric quality (a model can have low "
                        "BCE loss while predicting badly-shrunken clouds).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of training set held out for validation.")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cuda', 'cuda:0', 'cpu', ...")
    p.add_argument("--wandb", choices=["online", "offline", "disabled"],
                   default="offline")

    # Inference / benchmark
    # ode-step: 0.05 = 20 Euler steps over t∈[0,1]. 0.1 was undersolving the
    # flow — the velocity field had to traverse the full normalised AABB in
    # 10 steps which is too coarse for thin-tube structures.
    p.add_argument("--ode-step", type=float, default=0.05)
    p.add_argument("--gt-diameter", type=float, default=0.1,
                   help="Physical pipe diameter for E_caliper (post-normalize). "
                        "Fallback when per-scene diameter is missing from the .pt.")
    p.add_argument("--survival-threshold", type=float, default=0.3,
                   help="sigmoid(alpha) threshold for keeping a predicted point. "
                        "Default 0.3, deliberately below the sigmoid mid-point: "
                        "with lambda_surv tuned down (vel-dominant loss), α tends "
                        "to be cautious and a strict 0.5 over-prunes the cloud, "
                        "producing visually-shrunken predictions. 0.3 keeps "
                        "borderline-confident points without re-introducing the "
                        "0.0-threshold bug. Per-scene kept/total counts are "
                        "logged at inference so you can tell when α is drifting.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for benchmark.py.")

    # Multi-GPU / parallel preprocessing
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated physical GPU IDs to use, e.g. '0,2,5'. "
                        "Overrides --num-gpus. Default: use all available GPUs.")
    p.add_argument("--num-gpus", type=int, default=-1,
                   help="Number of GPUs (ignored when --gpus is set). "
                        "-1 = auto-detect via torch.cuda.device_count(). "
                        "1 keeps the legacy single-GPU path; >=2 enables DDP training "
                        "and per-scene SfM parallelism.")
    p.add_argument("--sfm-workers", type=int, default=-1,
                   help="Parallel SfM scenes. -1 = match --num-gpus (one scene per GPU).")
    p.add_argument("--preproc-cpu-workers", type=int, default=-1,
                   help="Parallel CPU workers for GT/filter/assemble subprocess fan-out. "
                        "-1 = min(8, os.cpu_count()).")
    p.add_argument("--ddp-master-port", type=str, default="29500",
                   help="MASTER_PORT for torch.distributed rendezvous (loopback only).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


_LOG_LOCK = threading.Lock()


def run_subprocess(cmd: list[str], log: logging.Logger, cwd: Path = _ROOT,
                   env: dict | None = None, label: str = "") -> bool:
    """Run a subprocess and stream stderr/stdout into the pipeline log.

    Thread-safe: log writes are serialized by a module lock so parallel
    workers don't interleave their lines.
    """
    label_prefix = f"[{label}] " if label else ""
    cmd_str = " ".join(str(c) for c in cmd)
    with _LOG_LOCK:
        log.info("%s$ %s", label_prefix, cmd_str)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    with _LOG_LOCK:
        if proc.returncode != 0:
            log.error("%sFAILED (%.1fs): %s", label_prefix, elapsed, cmd_str)
            if proc.stdout:
                log.error("--- stdout ---\n%s", proc.stdout.strip())
            if proc.stderr:
                log.error("--- stderr ---\n%s", proc.stderr.strip())
            return False
        log.info("%sok (%.1fs)", label_prefix, elapsed)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Parallel execution helpers
# ─────────────────────────────────────────────────────────────────────────────
def resolve_gpu_ids(gpus_str: str | None, num_gpus_spec: int) -> list[int]:
    """Resolve --gpus / --num-gpus into a concrete list of physical GPU IDs.

    --gpus '0,2,5'  →  [0, 2, 5]          (explicit)
    --num-gpus 2    →  [0, 1]              (first N)
    --num-gpus -1   →  [0 .. device_count) (auto-detect)
    """
    if gpus_str is not None:
        return [int(g) for g in gpus_str.split(",")]
    try:
        import torch
        n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_available = 0
    if num_gpus_spec >= 0:
        n = min(num_gpus_spec, max(n_available, 1))
    else:
        n = max(1, n_available)
    return list(range(n))


def resolve_cpu_workers(spec: int) -> int:
    if spec > 0:
        return spec
    return min(8, os.cpu_count() or 1)


def run_parallel_subprocess(
    tasks: list[dict],
    num_workers: int,
    log: logging.Logger,
    gpu_pool: list[int] | None = None,
    stage_name: str = "stage",
) -> list[bool]:
    """Run a batch of subprocess tasks concurrently.

    Each task is a dict: {"cmd": list[str], "label": str, "skip": bool (optional)}.
    Skipped tasks short-circuit to True and are logged once.

    If `gpu_pool` is given, workers acquire a GPU index from a shared queue
    and inject `CUDA_VISIBLE_DEVICES` into the subprocess env, then return
    the index when done. This pins each subprocess to one physical GPU.
    """
    if not tasks:
        return []

    num_workers = max(1, min(num_workers, len(tasks)))
    gpu_q: queue.Queue[int] | None = None
    if gpu_pool:
        gpu_q = queue.Queue()
        for g in gpu_pool:
            gpu_q.put(g)

    results: list[bool] = [False] * len(tasks)

    def worker(i: int, task: dict) -> tuple[int, bool]:
        if task.get("skip"):
            with _LOG_LOCK:
                log.info("[%s %d/%d] %s cached", stage_name, i + 1, len(tasks), task["label"])
            return i, True

        env = None
        gpu_idx: int | None = None
        if gpu_q is not None:
            gpu_idx = gpu_q.get()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        try:
            label = f"{stage_name} {i + 1}/{len(tasks)} {task['label']}"
            if gpu_idx is not None:
                label += f" gpu={gpu_idx}"
            ok = run_subprocess(task["cmd"], log, env=env, label=label)
            return i, ok
        finally:
            if gpu_q is not None and gpu_idx is not None:
                gpu_q.put(gpu_idx)

    log.info("══ %s: %d task(s), %d worker(s)%s ══",
             stage_name, len(tasks), num_workers,
             f", gpu_pool={gpu_pool}" if gpu_pool else "")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(worker, i, t) for i, t in enumerate(tasks)]
        for fut in as_completed(futures):
            idx, ok = fut.result()
            results[idx] = ok

    n_ok = sum(results)
    n_fail = len(results) - n_ok
    if n_fail:
        log.warning("%s: %d/%d failed", stage_name, n_fail, len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scene discovery + split
# ─────────────────────────────────────────────────────────────────────────────
def discover_scenes(input_root: Path, categories: list[str], limit: int,
                    log: logging.Logger) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for cat in categories:
        cat_dir = input_root / cat
        if not cat_dir.is_dir():
            log.warning("category '%s' not found under %s — skipping", cat, input_root)
            continue
        scenes = sorted(p for p in cat_dir.iterdir()
                        if p.is_dir() and (p / "transforms.json").exists())
        if limit > 0:
            scenes = scenes[:limit]
        out[cat] = scenes
        log.info("category %-10s -> %d scene(s)", cat, len(scenes))
    return out


def stratified_split(scenes_by_cat: dict[str, list[Path]], seed: int,
                     log: logging.Logger) -> tuple[list[Path], list[Path]]:
    """Per-category 9:1, deterministic. Each category contributes max(1, n/10) test."""
    rng = random.Random(seed)
    train, test = [], []
    for cat, scenes in scenes_by_cat.items():
        if not scenes:
            continue
        shuffled = scenes[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * 0.1))
        cat_test = shuffled[:n_test]
        cat_train = shuffled[n_test:]
        log.info("split %-10s : %d train / %d test", cat, len(cat_train), len(cat_test))
        train.extend(cat_train)
        test.extend(cat_test)
    return train, test


def write_split_record(out: Path, train: list[Path], test: list[Path]) -> None:
    record = {
        "train": [{"name": p.name, "category": p.parent.name, "path": str(p)} for p in train],
        "test":  [{"name": p.name, "category": p.parent.name, "path": str(p)} for p in test],
    }
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_split_record(path: Path) -> tuple[list[Path], list[Path]]:
    """Load train/test scene Paths from an existing split.json."""
    record = json.loads(path.read_text(encoding="utf-8"))
    train = [Path(e["path"]) for e in record.get("train", [])]
    test  = [Path(e["path"]) for e in record.get("test",  [])]
    return train, test


def clean_stale_outputs(output_root: Path,
                        train_scenes: list[Path],
                        test_scenes: list[Path],
                        log: logging.Logger,
                        force: bool = False) -> None:
    """Drop outputs that don't match the current split.

    benchmark.py pairs prediction and GT clouds by file stem, so any .ply left
    over from a previous split (or a previous run with --limit) would silently
    contaminate the metrics. Same hazard for outputs/assembled/<scene>.pt — a
    stale .pt with mismatched normalization can derail training.

    Allowed-stem sets (per directory):
      • results/Ours/*.ply, test_gt/*.ply  →  only current TEST stems are valid.
        Anything else (including stems that belong to the current train set —
        those are residue from an earlier split where the scene was a test)
        gets removed.
      • assembled/*.pt  →  any stem in train ∪ test is valid (we use it for
        both sides), so we only delete stems outside the discovered scenes.

    If `force` is True every file under those three dirs is wiped.
    """
    test_stems = {p.name for p in test_scenes}
    all_stems  = test_stems | {p.name for p in train_scenes}

    targets = [
        ("predictions", output_root / "results" / "Ours", "*.ply", test_stems),
        ("test_gt",     output_root / "test_gt",          "*.ply", test_stems),
        ("assembled",   output_root / "assembled",        "*.pt",  all_stems),
    ]

    total_removed = 0
    for label, d, pattern, allowed in targets:
        if not d.exists():
            continue
        removed_here = 0
        for f in d.glob(pattern):
            if force or f.stem not in allowed:
                f.unlink()
                removed_here += 1
        if removed_here:
            log.info("cleaned %d stale %s file(s) under %s",
                     removed_here, label, d)
            total_removed += removed_here

    if total_removed == 0:
        log.info("clean_stale_outputs: nothing to clean (split matches disk)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: SfM (known-pose) + background filter
# ─────────────────────────────────────────────────────────────────────────────
def stage_sfm(scenes: list[Path], output_root: Path, args: argparse.Namespace,
              log: logging.Logger, gpu_ids: list[int]) -> list[Path]:
    """Run known-pose SfM + filter for every scene, scenes fanned across GPUs.

    Two passes:
      1. SfM (LoFTR/hloc, GPU-bound) — parallel with one scene per GPU,
         CUDA_VISIBLE_DEVICES pinned per subprocess.
      2. Background filter (CPU-only) — parallel CPU.
    Returns list of workspaces (in scene order).
    """
    log.info("══ Stage SfM: %d scene(s), gpus=%s ══", len(scenes), gpu_ids)
    workspaces: list[Path] = []
    sfm_tasks: list[dict] = []
    filter_tasks: list[dict] = []

    for scene in scenes:
        cat = scene.parent.name
        ws = output_root / "sfm" / cat / scene.name
        workspaces.append(ws)

        sparse_ply = ws / "sparse" / "points3D.ply"
        filtered_ply = ws / "sparse_filtered" / "points3D.ply"

        cmd = [sys.executable, "scripts/run_sfm_known_pose.py", str(scene), str(ws), "--fresh"]
        if not args.no_relax:
            cmd.append("--relax")
        if args.num_matched > 0:
            cmd.extend(["--num-matched", str(args.num_matched)])
        if args.resize_max > 0:
            cmd.extend(["--resize-max", str(args.resize_max)])
        sfm_tasks.append({
            "cmd": cmd,
            "label": f"SfM {scene.name}",
            "skip": args.skip_existing and not args.overwrite and sparse_ply.exists(),
            "_ws": ws,
            "_scene": scene,
            "_filtered_ply": filtered_ply,
        })

    sfm_workers = args.sfm_workers if args.sfm_workers > 0 else len(gpu_ids)
    gpu_pool = gpu_ids if len(gpu_ids) > 0 else None
    sfm_results = run_parallel_subprocess(
        sfm_tasks, num_workers=sfm_workers, log=log,
        gpu_pool=gpu_pool, stage_name="SfM",
    )

    # Build filter tasks only for scenes whose SfM stage produced (or already had) a sparse ply.
    cpu_workers = resolve_cpu_workers(args.preproc_cpu_workers)
    for ok, task in zip(sfm_results, sfm_tasks):
        ws = task["_ws"]
        scene = task["_scene"]
        if not ok:
            log.error("SfM failed for %s — skipping its filter pass", scene.name)
            continue
        if not (ws / "sparse" / "points3D.ply").exists():
            log.error("SfM ok but no sparse/points3D.ply for %s — skipping filter", scene.name)
            continue
        filter_tasks.append({
            "cmd": [sys.executable, "scripts/filter_sfm_background.py",
                    str(ws), "--scene-dir", str(scene)],
            "label": f"filter {scene.name}",
            "skip": args.skip_existing and not args.overwrite and task["_filtered_ply"].exists(),
        })

    run_parallel_subprocess(
        filter_tasks, num_workers=cpu_workers, log=log,
        gpu_pool=None, stage_name="filter",
    )
    return workspaces


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: PDS GT
# ─────────────────────────────────────────────────────────────────────────────
def stage_gt(scenes: list[Path], output_root: Path, args: argparse.Namespace,
             log: logging.Logger) -> None:
    log.info("══ Stage GT (PDS): %d scene(s) ══", len(scenes))
    gt_dir = output_root / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for scene in scenes:
        obj = scene / f"{scene.name}.obj"
        if not obj.exists():
            log.warning("no .obj for %s — skip", scene.name)
            continue
        out_pt = gt_dir / f"{scene.name}.pt"
        cmd = [sys.executable, "scripts/cad_to_gt.py", str(obj), str(gt_dir)]
        if args.overwrite:
            cmd.append("--overwrite")
        should_skip = args.skip_existing and not args.overwrite and out_pt.exists()
        tasks.append({
            "cmd": cmd,
            "label": f"PDS {scene.name}",
            "skip": should_skip,
        })
    cpu_workers = resolve_cpu_workers(args.preproc_cpu_workers)
    run_parallel_subprocess(tasks, num_workers=cpu_workers, log=log,
                            gpu_pool=None, stage_name="GT")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Assemble .pt + write split symlinks
# ─────────────────────────────────────────────────────────────────────────────
def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def stage_assemble(train_scenes: list[Path], test_scenes: list[Path],
                   output_root: Path, args: argparse.Namespace,
                   log: logging.Logger) -> tuple[Path, Path]:
    log.info("══ Stage Assemble (.pt) ══")
    assembled = output_root / "assembled"
    train_dir = output_root / "training"
    test_dir = output_root / "test_input"
    for d in (assembled, train_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_scenes = list(train_scenes) + list(test_scenes)
    tasks: list[dict] = []
    for scene in all_scenes:
        cat = scene.parent.name
        ws = output_root / "sfm" / cat / scene.name
        gt_pt = output_root / "gt" / f"{scene.name}.pt"
        out_pt = assembled / f"{scene.name}.pt"

        if not gt_pt.exists():
            log.warning("%s missing GT .pt — skip assemble", scene.name)
            continue
        if not (ws / "sparse_filtered" / "points3D.ply").exists() and \
           not (ws / "sparse" / "points3D.ply").exists():
            log.warning("%s missing SfM ply — skip assemble", scene.name)
            continue

        tasks.append({
            "cmd": [sys.executable, "scripts/ply_to_pt.py",
                    str(ws), str(gt_pt), str(out_pt)],
            "label": f"assemble {scene.name}",
            "skip": args.skip_existing and not args.overwrite and out_pt.exists(),
        })
    cpu_workers = resolve_cpu_workers(args.preproc_cpu_workers)
    run_parallel_subprocess(tasks, num_workers=cpu_workers, log=log,
                            gpu_pool=None, stage_name="assemble")

    # write split symlinks
    for d in (train_dir, test_dir):
        for f in d.glob("*.pt"):
            f.unlink()
    for s in train_scenes:
        src = (assembled / f"{s.name}.pt").resolve()
        if src.exists():
            _link(src, train_dir / f"{s.name}.pt")
    for s in test_scenes:
        src = (assembled / f"{s.name}.pt").resolve()
        if src.exists():
            _link(src, test_dir / f"{s.name}.pt")

    n_train = len(list(train_dir.glob("*.pt")))
    n_test = len(list(test_dir.glob("*.pt")))
    log.info("split symlinks ready: %d train / %d test", n_train, n_test)
    return train_dir, test_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Training — single-GPU path + DDP launcher
# ─────────────────────────────────────────────────────────────────────────────
def stage_train(train_dir: Path, output_root: Path, args: argparse.Namespace,
                log: logging.Logger, gpu_ids: list[int]) -> Path:
    """Launcher: dispatch to single-GPU body or spawn DDP workers."""
    weights_dir = output_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best_model.pth"

    if len(gpu_ids) <= 1:
        device_str = f"cuda:{gpu_ids[0]}" if gpu_ids else args.device
        device = resolve_device(device_str)
        log.info("══ Stage Train (single, device=%s) ══", device)
        return _train_single(train_dir, output_root, args, log, device, best_path)

    num_gpus = len(gpu_ids)
    log.info("══ Stage Train (DDP gpus=%s, batch=%d/gpu, effective=%d) ══",
             gpu_ids, args.batch_size, args.batch_size * num_gpus)

    import torch.multiprocessing as mp
    log_path = str(output_root / "pipeline.log")
    mp.spawn(
        _train_worker_ddp,
        args=(num_gpus, str(train_dir), str(output_root), args,
              log_path, args.ddp_master_port, gpu_ids),
        nprocs=num_gpus,
        join=True,
    )
    return best_path


def _train_single(train_dir: Path, output_root: Path, args: argparse.Namespace,
                  log: logging.Logger, device: "torch.device",
                  best_path: Path) -> Path:
    import torch
    from torch.utils.data import DataLoader, Subset

    from data.dataset import UnderwaterPatchDataset
    from models.cufm_net import CG_UFM_Network
    from core.flow_matching import FlowMatchingLoss

    log.info("data=%s epochs=%d bs=%d lr=%.0e device=%s",
             train_dir, args.epochs, args.batch_size, args.lr, device)
    latest_path = best_path.parent / "latest_model.pth"

    use_wandb = args.wandb != "disabled"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore
            os.environ["WANDB_MODE"] = args.wandb
            _wandb.init(project="CG-UFM", mode=args.wandb,
                        config={"learning_rate": args.lr, "epochs": args.epochs,
                                "batch_size": args.batch_size,
                                "data_dir": str(train_dir)})
            wandb = _wandb
        except Exception as e:
            log.warning("wandb init failed (%s) — continuing without it", e)
            wandb = None

    dataset = UnderwaterPatchDataset(data_dir=str(train_dir))
    if len(dataset) == 0:
        log.error("training dataset is empty (%s)", train_dir)
        return best_path

    # Train/val split (deterministic by seed=0). See DDP path for rationale.
    val_indices: list[int] = []
    train_indices: list[int] = list(range(len(dataset)))
    if args.val_every > 0 and args.val_frac > 0:
        rng_split = np.random.default_rng(0)
        n_val = max(1, int(round(len(dataset) * args.val_frac)))
        all_idx = np.arange(len(dataset))
        rng_split.shuffle(all_idx)
        val_indices = sorted(all_idx[:n_val].tolist())
        train_indices = sorted(all_idx[n_val:].tolist())
        log.info("validation split: %d train / %d val (every %d epochs)",
                 len(train_indices), len(val_indices), args.val_every)
    train_subset = Subset(dataset, train_indices)
    val_pool = [dataset[i] for i in val_indices]

    loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)

    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64,
                           backbone_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FlowMatchingLoss(lambda_vel=2.0, lambda_surv=1.0,
                                 lambda_ot=0.1).to(device)

    # See DDP path for rationale on best_fscore + best_loss fallback.
    best_fscore = -float("inf")
    best_loss = float("inf")
    metric_keys = ("loss_total", "loss_vel", "loss_surv", "loss_transport",
                   "survivor_ratio", "alpha_mean")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        metric_acc = {k: 0.0 for k in metric_keys}
        for batch in loader:
            optimizer.zero_grad()
            x_raw = batch["noisy_points"].to(device)
            features = batch["features"].to(device)
            x_gt = batch["gt_points"].to(device)

            c_i = model.consensus_mlp(features)
            x_0, c_dense = model.densifier(x_raw, c_i)
            B, M, _ = x_0.shape
            t = torch.rand((B, 1), device=device)

            with torch.no_grad():
                matched_x_gt, _, _, _ = criterion.compute_ot_assignment(x_0, x_gt)

            t_exp = t.unsqueeze(-1).expand(-1, M, 3)
            x_t = (1 - t_exp) * x_0 + t_exp * matched_x_gt
            v_pred, alpha_pred = model(x_t, t, c_dense)

            loss, metrics = criterion(
                x_0, x_gt, v_pred, alpha_pred, t,
                matched_x_gt_precomputed=matched_x_gt,
            )
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            for k in metric_keys:
                if k in metrics:
                    metric_acc[k] += float(metrics[k])

        scheduler.step()
        avg = running / max(1, n_batches)
        lr_now = scheduler.get_last_lr()[0]
        log.info("epoch %3d/%d  loss=%.4f  lr=%.2e", epoch + 1, args.epochs, avg, lr_now)
        epoch_metrics = {k: metric_acc[k] / max(1, n_batches) for k in metric_keys}

        val_metrics = None
        should_validate = (
            args.val_every > 0
            and val_pool
            and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs)
        )
        if should_validate:
            val_metrics = _run_validation(
                model, val_pool, device,
                ode_step=args.ode_step,
                survival_threshold=args.survival_threshold,
            )
            log.info("epoch %3d  val: CD_norm=%.4f  F@2%%=%.4f",
                     epoch + 1, val_metrics["cd_norm"],
                     val_metrics["fscore_2pct"])

        if wandb is not None:
            log_payload = {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train/avg_loss": avg,
                **{f"train/{k}": v for k, v in epoch_metrics.items()},
            }
            if val_metrics is not None:
                log_payload["val/cd_norm"] = val_metrics["cd_norm"]
                log_payload["val/fscore_2pct"] = val_metrics["fscore_2pct"]
            wandb.log(log_payload, step=epoch + 1)

        torch.save(model.state_dict(), latest_path)
        improved = False
        if val_metrics is not None and np.isfinite(val_metrics["fscore_2pct"]):
            if val_metrics["fscore_2pct"] > best_fscore:
                best_fscore = val_metrics["fscore_2pct"]
                improved = True
        elif not np.isfinite(best_fscore) or best_fscore == -float("inf"):
            if avg < best_loss:
                best_loss = avg
                improved = True
        if improved:
            torch.save(model.state_dict(), best_path)

    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    log.info("training done. best F@2%%=%.4f (loss-fallback=%.4f) -> %s",
             best_fscore if best_fscore > -float("inf") else float("nan"),
             best_loss, best_path)
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper (used by both single-GPU and DDP training paths)
# ─────────────────────────────────────────────────────────────────────────────
def _run_validation(model, val_subset, device, ode_step: float,
                    survival_threshold: float):
    """Single-pass geometric validation on a held-out training subset.

    Returns:
        dict with keys "cd_norm" and "fscore_2pct" (means over scenes).
        NaN if the subset is empty or every scene errored out.

    Why this exists: training loss (vel-MSE + surv-BCE + transport-cost) can
    descend cleanly while pred geometry quietly drifts — e.g. the model can
    drive BCE low by over-pruning α, which manifests as a shrunken pred bbox
    that loss doesn't see. CD_norm and F-Score@2% are scale-aware geometric
    metrics that catch this directly. We use F-Score@2% (not 1%) as the
    primary signal because 1% gave 4% raw values in the v1 run and is hard
    to read; 2% lands in [0.1, 0.6] for our scale, which is monotonic and
    interpretable.

    The validation pass runs the full ODE-integrate → survival-mask →
    de-normalize pipeline (mirrors stage_infer), so the metric reflects what
    benchmark.py will actually see. DDP-safe: caller is responsible for
    invoking this on rank 0 only and broadcasting results if needed.
    """
    import numpy as _np
    import torch  # imported here because the helper sits at module scope
    from core.ode_solver import ODESolver
    from scipy.spatial import KDTree as _KDTree

    if len(val_subset) == 0:
        return {"cd_norm": float("nan"), "fscore_2pct": float("nan")}

    model.eval()
    solver = ODESolver(method="euler", step_size=ode_step)
    cd_norms = []
    fscores = []

    with torch.no_grad():
        for item in val_subset:
            x_raw = item["noisy_points"].unsqueeze(0).to(device)
            features = item["features"].unsqueeze(0).to(device)
            x_gt = item["gt_points"]                          # CPU tensor, (M, 3)

            c_i = model.consensus_mlp(features)
            x_0, c_dense = model.densifier(x_raw, c_i)
            x_1, alpha_1 = solver.integrate(model, x_0, c_dense)

            survival_mask = (
                torch.sigmoid(alpha_1.squeeze(0).squeeze(-1)) > survival_threshold
            )
            x_1_np = x_1.squeeze(0).detach().cpu().numpy().astype(_np.float64)
            pts = x_1_np[survival_mask.detach().cpu().numpy()]
            if pts.shape[0] < 0.10 * x_1_np.shape[0]:
                # Mirror stage_infer's fallback so validation numbers track
                # what the published metric would see.
                pts = x_1_np

            # Normalize both to a unit sphere (GT-defined) and compute CD_norm
            # and F-Score@2% in that frame. The 2% threshold uses the GT
            # bbox diagonal as a length scale, matching benchmark.py.
            gt_np = x_gt.detach().cpu().numpy().astype(_np.float64)
            if gt_np.size == 0 or pts.size == 0:
                continue
            aabb_min = gt_np.min(0)
            aabb_max = gt_np.max(0)
            center = (aabb_min + aabb_max) / 2.0
            diag = float(_np.linalg.norm(aabb_max - aabb_min))
            scale = max(diag / 2.0, 1e-9)
            pred_n = (pts - center) / scale
            gt_n = (gt_np - center) / scale

            tree_pred = _KDTree(pred_n)
            tree_gt = _KDTree(gt_n)
            d_p2g, _ = tree_gt.query(pred_n)
            d_g2p, _ = tree_pred.query(gt_n)
            cd_norm = 0.5 * (d_p2g.mean() + d_g2p.mean())

            # F-Score@2% uses absolute distance in the GT-original frame so
            # τ has physical meaning; both directions of nearest-neighbour
            # are computed unnormalized.
            tau = 0.02 * diag
            tree_pred_orig = _KDTree(pts)
            tree_gt_orig = _KDTree(gt_np)
            d_p2g_o, _ = tree_gt_orig.query(pts)
            d_g2p_o, _ = tree_pred_orig.query(gt_np)
            precision = float((d_p2g_o < tau).mean())
            recall = float((d_g2p_o < tau).mean())
            fscore = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            cd_norms.append(float(cd_norm))
            fscores.append(fscore)

    model.train()
    if not cd_norms:
        return {"cd_norm": float("nan"), "fscore_2pct": float("nan")}
    return {
        "cd_norm": float(_np.mean(cd_norms)),
        "fscore_2pct": float(_np.mean(fscores)),
    }


def _train_worker_ddp(rank: int, world_size: int, train_dir_str: str,
                      output_root_str: str, args: argparse.Namespace,
                      log_path: str, master_port: str,
                      gpu_ids: list[int] | None = None) -> None:
    """Per-rank DDP worker spawned by torch.multiprocessing.spawn.

    Initializes process group, builds DDP-wrapped model, runs training loop.
    Only rank 0 logs, saves checkpoints, and (optionally) writes wandb.
    Saves `model.module.state_dict()` so checkpoints stay drop-in compatible
    with the single-GPU path used by stage_infer.
    """
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from data.dataset import UnderwaterPatchDataset
    from models.cufm_net import CG_UFM_Network
    from core.flow_matching import FlowMatchingLoss

    train_dir = Path(train_dir_str)
    output_root = Path(output_root_str)
    is_main = rank == 0

    # Map logical rank → physical GPU ID
    physical_gpu = gpu_ids[rank] if gpu_ids else rank

    log = logging.getLogger(f"pipeline.rank{rank}")
    log.handlers.clear()
    log.setLevel(logging.INFO if is_main else logging.WARNING)
    log.propagate = False
    if is_main:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(physical_gpu)
    device = torch.device(f"cuda:{physical_gpu}")

    if is_main:
        log.info("DDP rank=%d/%d device=%s data=%s epochs=%d bs/gpu=%d lr=%.0e",
                 rank, world_size, device, train_dir, args.epochs,
                 args.batch_size, args.lr)

    use_wandb = is_main and args.wandb != "disabled"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore
            os.environ["WANDB_MODE"] = args.wandb
            _wandb.init(project="CG-UFM", mode=args.wandb,
                        config={"learning_rate": args.lr, "epochs": args.epochs,
                                "batch_size": args.batch_size,
                                "world_size": world_size,
                                "data_dir": str(train_dir)})
            wandb = _wandb
        except Exception as e:
            log.warning("wandb init failed (%s) — continuing without it", e)
            wandb = None

    weights_dir = output_root / "weights"
    if is_main:
        weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best_model.pth"
    latest_path = weights_dir / "latest_model.pth"

    dataset = UnderwaterPatchDataset(data_dir=str(train_dir))
    if len(dataset) == 0:
        if is_main:
            log.error("training dataset is empty (%s)", train_dir)
        dist.destroy_process_group()
        return

    # Train/val split. We hold out args.val_frac of the dataset (deterministic
    # by seed=0 so the split is identical across DDP ranks — every rank skips
    # the same indices). Validation runs on rank 0 only and reads the held-out
    # samples directly from the underlying dataset (no DistributedSampler over
    # val — DDP isn't useful for a ~15-sample geometric eval).
    val_indices: list[int] = []
    train_indices: list[int] = list(range(len(dataset)))
    if args.val_every > 0 and args.val_frac > 0:
        rng_split = np.random.default_rng(0)
        n_val = max(1, int(round(len(dataset) * args.val_frac)))
        all_idx = np.arange(len(dataset))
        rng_split.shuffle(all_idx)
        val_indices = sorted(all_idx[:n_val].tolist())
        train_indices = sorted(all_idx[n_val:].tolist())
        if is_main:
            log.info("validation split: %d train / %d val (every %d epochs)",
                     len(train_indices), len(val_indices), args.val_every)
    train_subset = Subset(dataset, train_indices)

    sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=False)
    loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=2, pin_memory=True)

    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64,
                           backbone_dim=256).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[physical_gpu], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FlowMatchingLoss(lambda_vel=2.0, lambda_surv=1.0,
                                 lambda_ot=0.1).to(device)

    # best_fscore (higher is better) replaces best_loss for ckpt selection.
    # We keep best_loss alongside it as a fallback when validation is disabled
    # or hasn't run yet (early epochs).
    best_fscore = -float("inf")
    best_loss = float("inf")
    val_pool = [dataset[i] for i in val_indices] if (is_main and val_indices) else []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n_batches = 0
        # Per-epoch accumulators for the sub-loss metrics returned by
        # FlowMatchingLoss. We aggregate over the local rank's batches, then
        # all-reduce at epoch end so wandb sees one entry per epoch (not one
        # per batch × world_size, which floods the offline run dir and is
        # never read while wandb mode is "offline").
        metric_keys = ("loss_total", "loss_vel", "loss_surv", "loss_transport",
                       "survivor_ratio", "alpha_mean")
        metric_acc = {k: 0.0 for k in metric_keys}

        for batch in loader:
            optimizer.zero_grad()
            x_raw = batch["noisy_points"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            x_gt = batch["gt_points"].to(device, non_blocking=True)

            inner = model.module
            c_i = inner.consensus_mlp(features)
            x_0, c_dense = inner.densifier(x_raw, c_i)
            B, M, _ = x_0.shape
            t = torch.rand((B, 1), device=device)

            with torch.no_grad():
                matched_x_gt, _, _, _ = criterion.compute_ot_assignment(x_0, x_gt)

            t_exp = t.unsqueeze(-1).expand(-1, M, 3)
            x_t = (1 - t_exp) * x_0 + t_exp * matched_x_gt
            v_pred, alpha_pred = model(x_t, t, c_dense)

            loss, metrics = criterion(
                x_0, x_gt, v_pred, alpha_pred, t,
                matched_x_gt_precomputed=matched_x_gt,
            )
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            for k in metric_keys:
                if k in metrics:
                    metric_acc[k] += float(metrics[k])

        scheduler.step()
        local_avg = running / max(1, n_batches)
        avg_t = torch.tensor([local_avg], device=device)
        dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
        avg = avg_t.item() / world_size
        lr_now = scheduler.get_last_lr()[0]

        # All-reduce the per-rank metric averages so wandb (rank 0) logs the
        # true cross-GPU mean per epoch. Build a stacked tensor for a single
        # all-reduce call instead of one round-trip per metric.
        local_means = torch.tensor(
            [metric_acc[k] / max(1, n_batches) for k in metric_keys],
            device=device, dtype=torch.float32,
        )
        dist.all_reduce(local_means, op=dist.ReduceOp.SUM)
        global_means = (local_means / world_size).tolist()
        epoch_metrics = dict(zip(metric_keys, global_means))

        if is_main:
            log.info("epoch %3d/%d  loss=%.4f  lr=%.2e",
                     epoch + 1, args.epochs, avg, lr_now)

            # Periodic geometric validation. Only on rank 0 — the other ranks
            # are blocked on the upcoming barrier (well, the next set_epoch
            # really, but the cost is negligible vs an epoch of training).
            val_metrics = None
            should_validate = (
                args.val_every > 0
                and val_pool
                and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs)
            )
            if should_validate:
                val_metrics = _run_validation(
                    model.module, val_pool, device,
                    ode_step=args.ode_step,
                    survival_threshold=args.survival_threshold,
                )
                log.info("epoch %3d  val: CD_norm=%.4f  F@2%%=%.4f",
                         epoch + 1, val_metrics["cd_norm"],
                         val_metrics["fscore_2pct"])

            if wandb is not None:
                log_payload = {
                    "epoch": epoch + 1,
                    "lr": lr_now,
                    "train/avg_loss": avg,
                    **{f"train/{k}": v for k, v in epoch_metrics.items()},
                }
                if val_metrics is not None:
                    log_payload["val/cd_norm"] = val_metrics["cd_norm"]
                    log_payload["val/fscore_2pct"] = val_metrics["fscore_2pct"]
                wandb.log(log_payload, step=epoch + 1)

            torch.save(model.module.state_dict(), latest_path)
            # Ckpt selection: prefer F-Score@2% (higher = better) when
            # validation has produced a finite value; otherwise fall back to
            # the loss-based criterion so we still write *something* in early
            # epochs. This is the fix for the previous run, where best_loss
            # picked a ckpt with low BCE but visually-shrunken predictions.
            improved = False
            if val_metrics is not None and np.isfinite(val_metrics["fscore_2pct"]):
                if val_metrics["fscore_2pct"] > best_fscore:
                    best_fscore = val_metrics["fscore_2pct"]
                    improved = True
            elif not np.isfinite(best_fscore) or best_fscore == -float("inf"):
                # No validation yet — keep the loss-based fallback.
                if avg < best_loss:
                    best_loss = avg
                    improved = True
            if improved:
                torch.save(model.module.state_dict(), best_path)

    dist.barrier()
    if is_main:
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        log.info("DDP training done. best F@2%%=%.4f (loss-fallback=%.4f) -> %s",
                 best_fscore if best_fscore > -float("inf") else float("nan"),
                 best_loss, best_path)
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Test-set inference + .ply export
# ─────────────────────────────────────────────────────────────────────────────
def stage_infer(test_dir: Path, output_root: Path, ckpt_path: Path,
                args: argparse.Namespace, log: logging.Logger,
                device: "torch.device") -> tuple[Path, Path]:
    import torch
    import open3d as o3d

    from data.dataset import UnderwaterPatchDataset
    from models.cufm_net import CG_UFM_Network
    from core.ode_solver import ODESolver

    log.info("══ Stage Infer: ckpt=%s ode_step=%.3f ══", ckpt_path, args.ode_step)
    pred_dir = output_root / "results" / "Ours"
    gt_dir = output_root / "test_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Fix 1: clear any stale .ply from a previous run / different split.
    # benchmark.py pairs by file stem, so leftover files would silently
    # poison the metrics with samples that no longer belong to this split.
    n_stale_pred = 0
    for f in pred_dir.glob("*.ply"):
        f.unlink()
        n_stale_pred += 1
    n_stale_gt = 0
    for f in gt_dir.glob("*.ply"):
        f.unlink()
        n_stale_gt += 1
    if n_stale_pred or n_stale_gt:
        log.info("cleared %d stale pred + %d stale GT .ply from prior run",
                 n_stale_pred, n_stale_gt)

    if not ckpt_path.exists():
        log.error("checkpoint not found: %s — skipping infer", ckpt_path)
        return pred_dir, gt_dir

    dataset = UnderwaterPatchDataset(data_dir=str(test_dir))
    if len(dataset) == 0:
        log.error("test dataset empty (%s) — skipping infer", test_dir)
        return pred_dir, gt_dir

    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64,
                           backbone_dim=256).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        log.error(
            "checkpoint key mismatch — likely a pre-FiLM PointNet2 weight. "
            "The backbone is now PointTransformer+FiLM (background.md §3.1 "
            "Gap 1/2); retrain rather than try to map old keys. ckpt=%s",
            ckpt_path)
        raise RuntimeError(
            f"Incompatible checkpoint at {ckpt_path}. Retrain after the "
            "PointTransformer+FiLM migration."
        ) from e
    model.eval()

    solver = ODESolver(method="euler", step_size=args.ode_step)

    for i, scene_path in enumerate(sorted(test_dir.glob("*.pt")), 1):
        sample = torch.load(str(scene_path.resolve()), weights_only=True)
        # dataset.__getitem__ resamples — call it via dataset for consistency
        idx = next(k for k, fp in enumerate(dataset.file_list)
                   if Path(fp).name == scene_path.name)
        item = dataset[idx]
        x_raw = item["noisy_points"].unsqueeze(0).to(device)
        features = item["features"].unsqueeze(0).to(device)
        # GT for export must come from the ORIGINAL (un-normalized) frame:
        # benchmark.py compares against this, and the on-disk .ply GTs from
        # the previous run were also in the original frame. If we wrote the
        # normalized GT here we'd silently change scale between runs.
        x_gt_full = sample["gt_points"]

        # Pull the per-sample affine the dataset applied. We need it to map
        # the model's normalized output back into the original CAD frame.
        # Older .pt files (or normalize_per_sample=False) won't have these
        # keys in `item` — fall back to identity in that case.
        norm_center = item.get("normalize_center", None)
        norm_scale = item.get("normalize_scale", None)
        if norm_center is not None and norm_scale is not None:
            norm_center_np = norm_center.detach().cpu().numpy().astype(np.float64).reshape(1, 3)
            norm_scale_f = float(norm_scale.detach().cpu().item())
        else:
            norm_center_np = np.zeros((1, 3), dtype=np.float64)
            norm_scale_f = 1.0

        with torch.no_grad():
            c_i = model.consensus_mlp(features)
            x_0, c_dense = model.densifier(x_raw, c_i)
            x_1, alpha_1 = solver.integrate(model, x_0, c_dense)

        survival_mask = torch.sigmoid(alpha_1.squeeze(0).squeeze(-1)) > args.survival_threshold
        x_1_np = x_1.squeeze(0).detach().cpu().numpy().astype(np.float64)
        pts = x_1_np[survival_mask.detach().cpu().numpy()]
        # Track how many points α actually kept BEFORE the fallback rewrites
        # `pts` to the full x_1. We log this every scene (not just on the
        # <10% catastrophic case) so a drift in α confidence is visible
        # without waiting for the fallback warning. Also useful when tuning
        # --survival-threshold.
        kept_before_fallback = int(pts.shape[0])
        alpha_mean_scene = float(torch.sigmoid(alpha_1).mean().item())

        # Fallback: if the survival head wiped out everything (or almost
        # everything — common with an undertrained or mis-thresholded α),
        # benchmark would see a tiny cloud and report meaningless metrics.
        # Keep all x_1 points so downstream CD/F-Score at least measures
        # the velocity field. The log line below makes the regression
        # visible: a healthy run keeps roughly the same number of points
        # as x_1 (4096), a bad α-head triggers the fallback.
        min_keep_frac = 0.10
        if pts.shape[0] < min_keep_frac * x_1_np.shape[0]:
            log.warning("[%d] %s — survival kept only %d/%d points (< %.0f%%); "
                        "falling back to raw x_1",
                        i, scene_path.stem, pts.shape[0], x_1_np.shape[0],
                        min_keep_frac * 100)
            pts = x_1_np

        # De-normalize back to the original CAD frame so pred and GT live
        # in the same coordinate system before benchmark.py reads them.
        pts = pts * norm_scale_f + norm_center_np

        # write predicted ply
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(pred_dir / f"{scene_path.stem}.ply"), pcd)

        # write GT ply (full-resolution gt_points from .pt, before FPS resample)
        gt_arr = np.asarray(x_gt_full, dtype=np.float64)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_arr)
        o3d.io.write_point_cloud(str(gt_dir / f"{scene_path.stem}.ply"), gt_pcd)

        # Sidecar: per-scene physical diameter. benchmark.py prefers this
        # over the global --gt-diameter fallback. Without it, E_caliper
        # measures "predicted diameter vs an arbitrary CLI constant" and
        # nothing in the metric tracks the actual scene geometry.
        if "gt_diameter" in sample:
            np.save(gt_dir / f"{scene_path.stem}_diameter.npy",
                    np.float32(sample["gt_diameter"]))
        log.info("[%d] %s  pred=%d  gt=%d  survival_kept=%d/%d  α_mean=%.3f",
                 i, scene_path.stem, pts.shape[0], gt_arr.shape[0],
                 kept_before_fallback, x_1_np.shape[0], alpha_mean_scene)

    return pred_dir, gt_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def stage_benchmark(gt_dir: Path, results_dir: Path, output_root: Path,
                    args: argparse.Namespace, log: logging.Logger) -> None:
    log.info("══ Stage Benchmark: gt=%s results=%s ══", gt_dir, results_dir)
    bench_out = output_root / "benchmark"
    cmd = [
        sys.executable, "benchmark.py",
        "--gt-dir", str(gt_dir),
        "--results-dir", str(results_dir),
        "--output-dir", str(bench_out),
        "--gt-diameter", str(args.gt_diameter),
        "--workers", str(args.workers),
        "--methods", "Ours",
    ]
    run_subprocess(cmd, log)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device(spec: str):
    import torch
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_root / "pipeline.log")
    gpu_ids = resolve_gpu_ids(args.gpus, args.num_gpus)
    log.info("CG-UFM full pipeline | input=%s output=%s stages=%s gpus=%s",
             args.input_root, args.output_root, args.stages, gpu_ids)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. discover scenes + decide split (always — needed by every stage)
    scenes_by_cat = discover_scenes(args.input_root, args.categories,
                                    args.limit, log)
    if not any(scenes_by_cat.values()):
        log.error("no scenes discovered — aborting")
        return 1

    split_path = args.output_root / "split.json"
    if args.reuse_split and split_path.exists():
        train_scenes, test_scenes = load_split_record(split_path)
        log.info("reusing existing split.json (%d train / %d test)",
                 len(train_scenes), len(test_scenes))
    else:
        train_scenes, test_scenes = stratified_split(scenes_by_cat, args.seed, log)
        write_split_record(split_path, train_scenes, test_scenes)
    all_scenes = train_scenes + test_scenes
    log.info("total: %d scenes (%d train / %d test)",
             len(all_scenes), len(train_scenes), len(test_scenes))

    # Fix 9: drop stale outputs whose stem is no longer in this split.
    # Always runs (cheap no-op when disk matches split). --clean-results
    # forces a full wipe of results/Ours, test_gt, assembled — useful when
    # you suspect mixed runs or want a guaranteed-clean benchmark.
    clean_stale_outputs(args.output_root, train_scenes, test_scenes,
                        log, force=args.clean_results)

    # 2. preprocessing stages
    if "sfm" in args.stages:
        stage_sfm(all_scenes, args.output_root, args, log, gpu_ids=gpu_ids)
    if "gt" in args.stages:
        stage_gt(all_scenes, args.output_root, args, log)

    train_dir = args.output_root / "training"
    test_dir = args.output_root / "test_input"
    if "assemble" in args.stages:
        train_dir, test_dir = stage_assemble(train_scenes, test_scenes,
                                             args.output_root, args, log)

    # 3. training + inference + benchmark (need torch only here, lazy import)
    device = None
    ckpt = args.output_root / "weights" / "best_model.pth"
    if "train" in args.stages:
        ckpt = stage_train(train_dir, args.output_root, args, log,
                           gpu_ids=gpu_ids)

    pred_dir = args.output_root / "results" / "Ours"
    gt_ply_dir = args.output_root / "test_gt"
    if "infer" in args.stages:
        # Inference runs on one GPU — use the first from the pool
        device_str = f"cuda:{gpu_ids[0]}" if gpu_ids else args.device
        device = resolve_device(device_str)
        pred_dir, gt_ply_dir = stage_infer(test_dir, args.output_root, ckpt,
                                           args, log, device)

    if "benchmark" in args.stages:
        stage_benchmark(gt_ply_dir, args.output_root / "results",
                        args.output_root, args, log)

    log.info("== pipeline complete ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
