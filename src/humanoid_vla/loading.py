"""Checkpoint saving/loading with legacy compatibility.

New checkpoints (``format: "humanoid_vla/v2"``) are fully self-contained:
model config, normalization statistics, task labels, and — for text-conditioned
models — precomputed instruction embeddings, so evaluation needs neither
``transformers`` nor network access.

Legacy checkpoints written by ``scripts/train_act.py`` / ``train_bimanual.py``
(pooled vision token, integer task-id, no normalization) also load: their
config keys are translated and identity norm stats are used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from humanoid_vla.models.act import ACTPolicy

CHECKPOINT_FORMAT = "humanoid_vla/v2"

# Keys allowed to be absent when loading a legacy state dict into v2.
_NORM_BUFFER_KEYS = {"state_mean", "state_std", "action_mean", "action_std"}


def save_checkpoint(
    path: str,
    model: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    task_labels: list[str],
    instruction_bank: dict | None = None,
) -> None:
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "task_labels": task_labels,
            "instruction_bank": instruction_bank,
        },
        path,
    )


class Policy:
    """Loaded policy + everything needed to run it.

    ``predict(image, state, task_id)`` works for both conditioning modes: in
    text mode the task id is resolved to a stored instruction embedding
    (``instruction_split`` selects canonical / train / heldout paraphrases).
    """

    def __init__(
        self,
        model: ACTPolicy,
        task_labels: list[str],
        device: str,
        instruction_bank: dict | None = None,
    ):
        self.model = model
        self.task_labels = task_labels
        self.device = device
        self.instruction_bank = instruction_bank
        self.chunk_size = model.chunk_size
        self.action_dim = model.action_dim

    def _task_input(self, task_id: int, instruction_split: str, rng: np.random.Generator | None):
        if self.model.conditioning == "task_id":
            return task_id
        label = self.task_labels[task_id]
        bank = self.instruction_bank[label]
        if instruction_split == "canonical":
            return bank["canonical"]
        embs = bank[instruction_split]
        idx = int(rng.integers(len(embs))) if rng is not None else 0
        return embs[idx]

    def predict(
        self,
        image: np.ndarray,
        state: np.ndarray,
        task_id: int = 0,
        instruction_split: str = "canonical",
        rng: np.random.Generator | None = None,
        device: str | None = None,  # tolerated for legacy call sites; self.device wins
    ) -> np.ndarray:
        """Returns a denormalized (chunk_size, action_dim) action chunk."""
        task = self._task_input(task_id, instruction_split, rng)
        return self.model.predict(image, state, task, device=self.device)


def load_policy(checkpoint_path: str, device: str = "cuda") -> tuple[Policy, dict]:
    """Load a v2 or legacy checkpoint → (:class:`Policy`, config dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: dict[str, Any] = dict(ckpt["config"])

    if ckpt.get("format") == CHECKPOINT_FORMAT:
        task_labels = ckpt["task_labels"]
        model_cfg = config["model"]
    else:
        # Legacy translation: scripts/train_act.py-style flat config.
        task_labels = config.get("task_labels", ["pick up the green box with both hands"])
        model_cfg = {
            "state_dim": config["state_dim"],
            "action_dim": config["action_dim"],
            "chunk_size": config["chunk_size"],
            "hidden_dim": config["hidden_dim"],
            "nhead": config.get("nhead", 4),
            "num_layers": config["num_layers"],
            "num_tasks": config["num_tasks"],
            "conditioning": "task_id",
            "vision_tokens": "pooled",
        }
        config = {"model": model_cfg, "legacy": True}

    model = ACTPolicy(pretrained_backbone=False, **model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Only the norm buffers may be missing (legacy checkpoints → identity stats).
    bad_missing = [k for k in missing if k not in _NORM_BUFFER_KEYS]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}")
    model.eval()

    policy = Policy(
        model,
        task_labels=task_labels,
        device=device,
        instruction_bank=ckpt.get("instruction_bank"),
    )
    epoch = ckpt.get("epoch")
    val_loss = ckpt.get("val_loss", ckpt.get("loss"))
    print(
        f"Loaded {checkpoint_path} (epoch {epoch}, val_loss "
        f"{val_loss if val_loss is None else f'{val_loss:.6f}'}, "
        f"conditioning={model.conditioning}, vision={model.vision_tokens})"
    )
    return policy, config
