"""Unified, reproducible ACT trainer for single-arm and bimanual data.

Replaces ``scripts/train_act.py`` and ``scripts/train_bimanual.py``. What the
legacy scripts were missing (see docs/CODEBASE_REVIEW.md §2.5) is built in:

- global seeding (recorded in the checkpoint),
- episode-level train/val split with ``best.pt`` selected on VALIDATION loss,
- state/action normalization from training-split statistics,
- L1 loss by default (matches the ACT paper; ``--loss mse`` for the old behavior),
- mixed precision on CUDA + multi-worker data loading,
- optional Weights & Biases logging (``--wandb``),
- optional text conditioning from instruction paraphrases (``--conditioning text``).

Usage (dims and task vocabulary are inferred from the data):

  # single-arm
  python -m humanoid_vla.train --demos data/demos --output data/checkpoints_v2

  # bimanual, success-filtered, text-conditioned
  python -m humanoid_vla.train --demos data/bimanual_demos_phase_f2 \\
      --output data/bimanual_checkpoints_v2 --filter-success --conditioning text
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from humanoid_vla.data import ChunkDataset, EpisodeStore
from humanoid_vla.loading import save_checkpoint
from humanoid_vla.models.act import ACTPolicy
from humanoid_vla.seeding import seed_everything, worker_init_fn


@dataclass
class TrainConfig:
    demos: str = "data/demos"
    output: str = "data/checkpoints_v2"
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    chunk_size: int = 20
    hidden_dim: int = 256
    nhead: int = 4
    num_layers: int = 4
    loss: str = "l1"  # "l1" | "mse"
    conditioning: str = "task_id"  # "task_id" | "text"
    text_model: str = "openai/clip-vit-base-patch32"
    vision_tokens: str = "spatial"  # "spatial" | "pooled"
    val_frac: float = 0.1
    seed: int = 42
    deterministic: bool = False
    augment: bool = True
    filter_success: bool = False
    num_workers: int = 4
    amp: bool = True  # mixed precision (CUDA only)
    grad_clip: float = 1.0
    log_freq: int = 5
    save_freq: int = 50
    resume: str = ""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    wandb: bool = False
    wandb_project: str = "humanoid_vla"
    wandb_run: str = ""
    pretrained_backbone: bool = True


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f_name, f_val in asdict(cfg).items():
        flag = "--" + f_name.replace("_", "-")
        if isinstance(f_val, bool):
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=f_name, action="store_true")
            group.add_argument(
                "--no-" + f_name.replace("_", "-"), dest=f_name, action="store_false"
            )
            parser.set_defaults(**{f_name: f_val})
        else:
            parser.add_argument(flag, type=type(f_val), default=f_val)
    args = parser.parse_args(argv)
    return TrainConfig(**vars(args))


class InstructionSampler:
    """Maps batch task-ids to sampled paraphrase embeddings (text mode)."""

    def __init__(self, bank: dict, task_labels: list[str], device: str, seed: int):
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.train_embs = [
            torch.from_numpy(np.asarray(bank[label]["train"])).float().to(device)
            for label in task_labels
        ]
        self.canonical = torch.stack(
            [
                torch.from_numpy(np.asarray(bank[label]["canonical"])).float()
                for label in task_labels
            ]
        ).to(device)

    def sample(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Random train-paraphrase embedding per sample (training)."""
        out = []
        for tid in task_ids.tolist():
            embs = self.train_embs[tid]
            idx = int(torch.randint(len(embs), (1,), generator=self.gen))
            out.append(embs[idx])
        return torch.stack(out)

    def canonical_batch(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Deterministic canonical embedding per sample (validation)."""
        return self.canonical[task_ids]


def _run_epoch(
    model: ACTPolicy,
    loader: DataLoader,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    sampler: InstructionSampler | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = F.l1_loss if cfg.loss == "l1" else F.mse_loss
    device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"
    use_amp = cfg.amp and device_type == "cuda"
    total, n = 0.0, 0

    with torch.set_grad_enabled(training):
        for images, states, task_ids, chunks in loader:
            images = images.to(cfg.device, non_blocking=True)
            states = states.to(cfg.device, non_blocking=True)
            task_ids = task_ids.to(cfg.device, dtype=torch.long, non_blocking=True)
            chunks = chunks.to(cfg.device, non_blocking=True)

            if sampler is not None:
                task_in = (
                    sampler.sample(task_ids) if training else sampler.canonical_batch(task_ids)
                )
            else:
                task_in = task_ids

            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred = model(images, states, task_in)
                target = model.normalize_actions(chunks)
                loss = loss_fn(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

            total += loss.item()
            n += 1
    return total / max(n, 1)


def train(cfg: TrainConfig) -> Path:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    output_dir = Path(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    store = EpisodeStore(cfg.demos, filter_success=cfg.filter_success)
    train_idx, val_idx = store.split(cfg.val_frac, seed=cfg.seed)
    stats = store.norm_stats(train_idx)
    train_ds = ChunkDataset(store, train_idx, cfg.chunk_size, augment=cfg.augment)
    val_ds = ChunkDataset(store, val_idx, cfg.chunk_size, augment=False)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── Model ──
    model_cfg = {
        "state_dim": store.state_dim,
        "action_dim": store.action_dim,
        "chunk_size": cfg.chunk_size,
        "hidden_dim": cfg.hidden_dim,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "num_tasks": len(store.task_labels),
        "conditioning": cfg.conditioning,
        "vision_tokens": cfg.vision_tokens,
    }
    model = ACTPolicy(pretrained_backbone=cfg.pretrained_backbone, **model_cfg).to(cfg.device)
    model.set_norm_stats(stats)
    total_params, trainable_params = model.count_params()

    # ── Language (text mode): embed paraphrases once, store in checkpoint ──
    instruction_bank = None
    sampler = None
    if cfg.conditioning == "text":
        from humanoid_vla.models.text_encoder import TextEncoder, build_instruction_bank

        encoder = TextEncoder(cfg.text_model, device=cfg.device)
        instruction_bank = build_instruction_bank(store.task_labels, encoder)
        sampler = InstructionSampler(instruction_bank, store.task_labels, cfg.device, seed=cfg.seed)
        del encoder  # free the text tower; embeddings are cached

    # ── Optimizer / scheduler / AMP ──
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.amp and cfg.device.startswith("cuda") else None

    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    full_config = {"model": model_cfg, "train": asdict(cfg), "norm_stats": stats.to_dict()}

    # ── wandb (optional) ──
    wb = None
    if cfg.wandb:
        try:
            import wandb as wb  # type: ignore[no-redef]

            wb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run or None,
                config=full_config,
            )
        except ImportError:
            print("wandb not installed (pip install -e .[track]) — continuing without")
            wb = None

    print(
        f"\n{'=' * 64}\n"
        f"humanoid_vla trainer\n"
        f"  demos:      {cfg.demos} — {len(train_ds)} train / {len(val_ds)} val samples\n"
        f"  episodes:   {len(train_idx)} train / {len(val_idx)} val (stratified)\n"
        f"  tasks:      {store.task_labels}\n"
        f"  model:      {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable\n"
        f"  vision:     {cfg.vision_tokens} · conditioning: {cfg.conditioning}\n"
        f"  loss: {cfg.loss} · amp: {scaler is not None} · seed: {cfg.seed} "
        f"· device: {cfg.device}\n"
        f"{'=' * 64}\n"
    )

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = _run_epoch(model, train_loader, cfg, optimizer, scaler, sampler)
        val_loss = _run_epoch(model, val_loader, cfg, None, None, sampler)
        scheduler.step()

        if wb is not None:
            wb.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
        if epoch % cfg.log_freq == 0 or epoch == cfg.epochs - 1:
            print(
                f"  epoch {epoch:4d}/{cfg.epochs} — train {train_loss:.5f} — "
                f"val {val_loss:.5f} — lr {scheduler.get_last_lr()[0]:.2e} — "
                f"{time.time() - t0:.0f}s"
            )

        def _save(path: Path, epoch=epoch, train_loss=train_loss, val_loss=val_loss) -> None:
            save_checkpoint(
                str(path),
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                full_config,
                store.task_labels,
                instruction_bank,
            )

        if val_loss < best_val:
            best_val = val_loss
            _save(output_dir / "best.pt")
        if (epoch > 0 and epoch % cfg.save_freq == 0) or epoch == cfg.epochs - 1:
            _save(output_dir / "latest.pt")

    print(
        f"\nDone in {(time.time() - t0) / 60:.1f} min — best val loss {best_val:.5f}\n"
        f"Checkpoints: {output_dir.resolve()}\n"
        f"Next: MUJOCO_GL=egl python3 scripts/evaluate.py "
        f"--checkpoint {output_dir}/best.pt --episodes 50"
    )
    if wb is not None:
        wb.finish()
    return output_dir / "best.pt"


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
