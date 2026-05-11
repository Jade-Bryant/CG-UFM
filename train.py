"""
CG-UFM training entry point.

Single-GPU usage:
    python train.py

Multi-GPU usage (DDP via torchrun):
    torchrun --nproc_per_node=4 train.py

Finetune from a checkpoint:
    python train.py --resume weights/best_model.pth --data-dir ./outputs_real/training --lr 1e-5 --epochs 30

DDP is detected automatically via the LOCAL_RANK env var. Only rank 0 does
wandb logging and checkpoint writes.
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from data.dataset import UnderwaterPatchDataset
from models.cufm_net import CG_UFM_Network
from core.flow_matching import FlowMatchingLoss


def parse_args():
    p = argparse.ArgumentParser(description="CG-UFM training / finetuning")
    p.add_argument("--data-dir", type=str, default="./datasets/dummy_dataset",
                   help="Directory containing .pt training patches")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pth) to resume / finetune from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="./weights")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────

def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def setup_distributed():
    """Initialize NCCL process group from torchrun env vars."""
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def unwrap(model):
    """Return the underlying CG_UFM_Network whether or not it is DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model


# ─────────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, criterion, device, is_main: bool):
    model.train()
    total_loss = 0.0
    inner = unwrap(model)

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        x_raw = batch['noisy_points'].to(device, non_blocking=True)
        features = batch['features'].to(device, non_blocking=True)
        x_gt = batch['gt_points'].to(device, non_blocking=True)

        c_i = inner.consensus_mlp(features)
        x_0, c_dense = inner.densifier(x_raw, c_i)
        B, M, _ = x_0.shape
        t = torch.rand((B, 1), device=device)

        # No-grad UOT pre-pass for x_t construction.
        with torch.no_grad():
            matched_x_gt, _, _, _ = criterion.compute_ot_assignment(x_0, x_gt)

        t_exp = t.unsqueeze(-1).expand(-1, M, 3)
        x_t = (1 - t_exp) * x_0 + t_exp * matched_x_gt

        # Main forward goes through DDP wrapper so gradients are synced on backward.
        v_pred, alpha_pred = model(x_t, t, c_dense)

        loss, metrics = criterion(
            x_0, x_gt, v_pred, alpha_pred, t,
            matched_x_gt_precomputed=matched_x_gt,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if is_main:
            wandb.log(metrics)
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

    return total_loss / max(1, len(dataloader))


# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    save_dir = args.save_dir

    # ─── DDP init ─────────────────────────────────────────────────
    if is_distributed():
        rank, world = setup_distributed()
        device = torch.device(f"cuda:{rank}")
        is_main = (rank == 0)
        print(f"[rank {rank}/{world}] DDP active on {device}")
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        print(f"single-process training on {device}")

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(project="CG-UFM",
                   config={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                           "world_size": world})

    # ─── Data ─────────────────────────────────────────────────────
    dataset = UnderwaterPatchDataset(data_dir=args.data_dir)
    if is_distributed():
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=True, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=2, pin_memory=True)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0)

    # ─── Model ────────────────────────────────────────────────────
    model = CG_UFM_Network(feature_dim=6, c_dim=64, time_emb_dim=64,
                           backbone_dim=256).to(device)
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        if is_main:
            print(f"Resumed from {args.resume}")
    if is_distributed():
        # FiLM, ConsensusBranch, both heads always run on every forward —
        # find_unused_parameters=False is safe.
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    criterion = FlowMatchingLoss(lambda_vel=1.0, lambda_surv=2.0,
                                 lambda_ot=0.1).to(device)

    if is_main:
        print(f"Starting training: epochs={epochs} world_size={world}")
    best_loss = float('inf')

    for epoch in range(epochs):
        if is_distributed():
            sampler.set_epoch(epoch)  # required so DDP shuffles change each epoch

        if is_main:
            print(f"--- Epoch {epoch+1}/{epochs} ---")

        local_avg = train_epoch(model, dataloader, optimizer, criterion,
                                device, is_main)

        # All-reduce mean loss across ranks for logging consistency
        if is_distributed():
            t_ = torch.tensor([local_avg], device=device)
            dist.all_reduce(t_, op=dist.ReduceOp.SUM)
            avg_loss = (t_.item() / world)
        else:
            avg_loss = local_avg

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if is_main:
            print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f} | lr={current_lr:.2e}")
            wandb.log({"epoch": epoch+1, "avg_loss": avg_loss, "lr": current_lr})

            # Save the unwrapped state dict (drop-in compatible with single-GPU)
            sd = unwrap(model).state_dict()
            torch.save(sd, os.path.join(save_dir, "latest_model.pth"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(sd, os.path.join(save_dir, "best_model.pth"))
                print("🌟 New best model saved!")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
