#!/usr/bin/env python3
"""
scripts/train_act.py

Standalone ACT training — reads HDF5 demos directly, trains an ACT policy,
saves checkpoints. No LeRobot dependency needed for training.

Usage:
  cd ~/projects/humanoid_vla
  python3 scripts/train_act.py                               # defaults: 300 epochs
  python3 scripts/train_act.py --epochs 1000 --batch-size 16 # longer training
  python3 scripts/train_act.py --resume data/checkpoints/latest.pt

Training on RTX 4050 (6GB VRAM):
  - ACT model: ~6M trainable params, ~1.5 GB VRAM at batch_size=32
  - 300 epochs ≈ 40 min, 1000 epochs ≈ 2.2 hours
  - Loss should drop below 0.001 for good convergence
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy, DemoDataset, TASK_LABELS


def count_params(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, epoch, loss, path, config):
    """Save training checkpoint with model config for standalone loading."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy on HDF5 demos")
    parser.add_argument("--demos", default="data/demos",
                        help="Directory with episode_NNNN.hdf5 files")
    parser.add_argument("--output", default="data/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (300 ≈ 40 min on RTX 4050)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (reduce to 16 if OOM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Action chunk length (predict N steps ahead)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=10,
                        help="Print loss every N epochs")
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable image augmentation")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    dataset = DemoDataset(args.demos, chunk_size=args.chunk_size,
                          augment=not args.no_augment)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,        # data is preloaded in RAM, no need for workers
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──
    config = {
        'state_dim': 58,
        'action_dim': 29,
        'chunk_size': args.chunk_size,
        'hidden_dim': args.hidden_dim,
        'nhead': 4,
        'num_layers': args.num_layers,
        'num_tasks': len(TASK_LABELS),
        'task_labels': TASK_LABELS,
    }

    model = ACTPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_tasks=config['num_tasks'],
    ).to(args.device)

    total_params, trainable_params = count_params(model)

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0

    # ── Resume ──
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}, loss was {ckpt['loss']:.6f}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"ACT Training")
    print(f"{'='*60}")
    print(f"  Dataset:    {len(dataset)} samples from {args.demos}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device}")
    print(f"  Model:      {total_params/1e6:.1f}M total, "
          f"{trainable_params/1e6:.1f}M trainable")
    print(f"  Output:     {output_dir.absolute()}")
    batches_per_epoch = len(loader)
    est_time = args.epochs * batches_per_epoch * 0.06 / 60  # ~60ms/batch
    print(f"  Batches/ep: {batches_per_epoch}")
    print(f"  Est. time:  ~{est_time:.0f} min")
    print(f"{'='*60}\n")

    # ── Training loop ──
    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, states, task_ids, action_chunks in loader:
            images = images.to(args.device)
            states = states.to(args.device)
            task_ids = task_ids.to(args.device, dtype=torch.long)
            action_chunks = action_chunks.to(args.device)

            # Forward
            pred = model(images, states, task_ids)    # (B, chunk, 29)
            loss = F.mse_loss(pred, action_chunks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Logging
        if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs} — "
                  f"loss: {avg_loss:.6f} — lr: {lr_now:.2e} — "
                  f"{elapsed:.0f}s elapsed")

        # Save checkpoint
        if (epoch > 0 and epoch % args.save_freq == 0) or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"checkpoint_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path, config)

            # Also save as 'latest.pt'
            latest_path = output_dir / "latest.pt"
            save_checkpoint(model, optimizer, epoch, avg_loss, latest_path, config)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "best.pt"
                save_checkpoint(model, optimizer, epoch, avg_loss, best_path, config)

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoints: {output_dir.absolute()}")
    print(f"\nNext step: evaluate the model")
    print(f"  MUJOCO_GL=egl python3 scripts/evaluate.py "
          f"--checkpoint {output_dir}/best.pt")


if __name__ == "__main__":
    main()
