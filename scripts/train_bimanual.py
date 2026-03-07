#!/usr/bin/env python3
"""
scripts/train_bimanual.py

Train a bimanual ACT model on physics-based demo data.
Separate from the single-arm model — different dimensions and single task.

Key differences from train_act.py:
  - state_dim = 28  (14 joint pos + 14 joint vel, both arms)
  - action_dim = 14 (7 left arm + 7 right arm targets)
  - Single task (no multi-task embedding needed)
  - Loads bimanual HDF5 format (14-dim obs/actions)

Usage:
  cd ~/projects/humanoid_vla
  python3 scripts/train_bimanual.py                     # defaults: 300 epochs
  python3 scripts/train_bimanual.py --epochs 500        # longer
  python3 scripts/train_bimanual.py --resume data/bimanual_checkpoints/latest.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvt

try:
    import h5py
except ImportError:
    sys.exit("h5py required: pip3 install --break-system-packages h5py")

try:
    import cv2
except ImportError:
    sys.exit("opencv required: pip3 install --break-system-packages opencv-python")

sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy

# ImageNet normalization
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# ────────────────────────────────────────────────────────
# Bimanual Dataset
# ────────────────────────────────────────────────────────

class BimanualDemoDataset(Dataset):
    """Loads bimanual HDF5 demo episodes for ACT training.

    Format per episode:
      obs/joint_positions:  (T, 14) — left arm (7) + right arm (7)
      obs/joint_velocities: (T, 14)
      obs/camera_frames:    (T, 480, 640, 3)
      action:               (T, 14) — joint position targets

    Each sample returns:
      image:        (3, 224, 224) float32 (ImageNet-normalized)
      state:        (28,) float32 (14 pos + 14 vel)
      task_id:      int (always 0 — single task)
      action_chunk: (chunk_size, 14) float32
    """

    def __init__(self, demos_dir: str, chunk_size: int = 20, augment: bool = False,
                 filter_success: bool = False):
        self.chunk_size = chunk_size
        self.augment = augment
        if augment:
            self.color_jitter = tvt.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            self.gaussian_blur = tvt.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        ep_files = sorted(
            f for f in os.listdir(demos_dir) if f.endswith('.hdf5'))
        if not ep_files:
            raise FileNotFoundError(f"No HDF5 files in {demos_dir}")

        self.episodes = []
        self.samples = []

        print(f"Loading {len(ep_files)} bimanual episodes from {demos_dir} ...")

        skipped = 0
        for ep_idx, fname in enumerate(ep_files):
            path = os.path.join(demos_dir, fname)
            with h5py.File(path, 'r') as f:
                # Skip failed episodes if filtering is enabled
                if filter_success and not bool(f.attrs.get('success', True)):
                    skipped += 1
                    continue
                positions = f['obs/joint_positions'][:]       # (T, 14)
                velocities = f['obs/joint_velocities'][:]     # (T, 14)
                actions = f['action'][:]                      # (T, 14)
                images_raw = f['obs/camera_frames'][:]        # (T, H, W, 3)

            T = len(positions)
            assert positions.shape[1] == 14, f"Expected 14-dim obs, got {positions.shape[1]}"
            assert actions.shape[1] == 14, f"Expected 14-dim action, got {actions.shape[1]}"

            # Resize images to 224×224
            images = np.empty((T, 224, 224, 3), dtype=np.uint8)
            for i in range(T):
                images[i] = cv2.resize(images_raw[i], (224, 224),
                                       interpolation=cv2.INTER_AREA)

            self.episodes.append({
                'positions': positions.astype(np.float32),
                'velocities': velocities.astype(np.float32),
                'actions': actions.astype(np.float32),
                'images': images,
                'length': T,
            })

            actual_idx = len(self.episodes) - 1
            for t in range(T):
                self.samples.append((actual_idx, t))

        if skipped > 0:
            print(f"  Skipped {skipped} failed episodes")
        total_frames = sum(ep['length'] for ep in self.episodes)
        print(f"  {len(self.episodes)} episodes, {len(self.samples)} samples, "
              f"avg {total_frames / len(self.episodes):.0f} frames/ep")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        T = ep['length']

        # Image: uint8 (224,224,3) → float32 (3,224,224), ImageNet-normalized
        img_uint8 = ep['images'][t]  # (224, 224, 3) uint8

        if self.augment:
            # Convert to tensor [0,1] for torchvision transforms
            img_t = torch.from_numpy(img_uint8.transpose(2, 0, 1).copy()).float() / 255.0
            img_t = self.color_jitter(img_t)
            if random.random() < 0.3:
                img_t = self.gaussian_blur(img_t)
            # Random resized crop: simulate small camera shifts
            if random.random() < 0.5:
                i, j, h, w = tvt.RandomResizedCrop.get_params(
                    img_t, scale=(0.85, 1.0), ratio=(0.95, 1.05))
                img_t = tvt.functional.resized_crop(img_t, i, j, h, w, [224, 224])
            # ImageNet normalize
            img = (img_t.numpy() - _IMG_MEAN) / _IMG_STD
            img = torch.from_numpy(img)
        else:
            img = img_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
            img = (img - _IMG_MEAN) / _IMG_STD
            img = torch.from_numpy(img)

        # State: 14 pos + 14 vel = 28
        state = np.concatenate([ep['positions'][t], ep['velocities'][t]])
        state = torch.from_numpy(state)

        # Task ID: always 0 (single task)
        task_id = 0

        # Action chunk with edge padding
        end = min(t + self.chunk_size, T)
        chunk = ep['actions'][t:end].copy()
        if len(chunk) < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad])
        chunk = torch.from_numpy(chunk)

        return img, state, task_id, chunk


# ────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, epoch, loss, path, config):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train bimanual ACT model on physics-based demos")
    parser.add_argument("--demos", default="data/bimanual_demos")
    parser.add_argument("--output", default="data/bimanual_checkpoints")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable image augmentation")
    parser.add_argument("--filter-success", action="store_true",
                        help="Skip episodes marked as failed (success=False in HDF5 attrs)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    dataset = BimanualDemoDataset(args.demos, chunk_size=args.chunk_size,
                                  augment=not args.no_augment,
                                  filter_success=args.filter_success)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──
    # state_dim=28 (14 pos + 14 vel), action_dim=14, single task
    config = {
        'state_dim':  28,
        'action_dim': 14,
        'chunk_size': args.chunk_size,
        'hidden_dim': args.hidden_dim,
        'nhead':      4,
        'num_layers': args.num_layers,
        'num_tasks':  1,
        'task_labels': ["pick up the green box with both hands"],
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
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    start_epoch = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}, loss was {ckpt['loss']:.6f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Bimanual ACT Training")
    print(f"{'='*60}")
    print(f"  Task:       pick up the green box with both hands")
    print(f"  Dataset:    {len(dataset)} samples from {args.demos}")
    print(f"  State dim:  28 (14 pos + 14 vel)")
    print(f"  Action dim: 14 (7 left + 7 right)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device}")
    print(f"  Model:      {total_params/1e6:.1f}M total, "
          f"{trainable_params/1e6:.1f}M trainable")
    print(f"  Output:     {output_dir.absolute()}")
    batches_per_epoch = len(loader)
    print(f"  Batches/ep: {batches_per_epoch}")
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

            pred = model(images, states, task_ids)    # (B, chunk, 14)
            loss = F.mse_loss(pred, action_chunks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs} — "
                  f"loss: {avg_loss:.6f} — lr: {lr_now:.2e} — "
                  f"{elapsed:.0f}s elapsed")

        if (epoch > 0 and epoch % args.save_freq == 0) or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"checkpoint_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path, config)

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
    print(f"\nNext: evaluate the bimanual model")
    print(f"  MUJOCO_GL=egl python3 scripts/evaluate_bimanual.py "
          f"--checkpoint {output_dir}/best.pt")


if __name__ == "__main__":
    main()
