"""Unified HDF5 demo dataset for single-arm and bimanual training.

Replaces the two ~95 %-identical dataset classes in ``scripts/act_model.py``
and ``scripts/train_bimanual.py``. Dimensions (29-d single-arm vs 14-d
bimanual) are inferred per file; the task vocabulary is discovered from the
``task_description`` attributes.

Key differences from the legacy pipeline:

- **Episode-level train/val split** (stratified by task) — checkpoint selection
  is done on validation loss, never training loss.
- **Normalization statistics** are computed on the training split only.
- Augmentation runs on tensors end-to-end (no numpy round-trips) and is safe
  with ``num_workers > 0``.

Episode schema (written by the generators in ``scripts/``):
  obs/joint_positions  (T, D)  float32
  obs/joint_velocities (T, D)  float32
  obs/camera_frames    (T, 480, 640, 3) uint8
  action               (T, D)  float32
  attrs: task_description (str), success (bool, bimanual only), ...
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from humanoid_vla.constants import CHUNK_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from humanoid_vla.normalization import NormStats, compute_norm_stats


class EpisodeStore:
    """Loads all episodes from a directory into RAM (images resized to 224²).

    ~9 k frames ≈ 1.4 GB as uint8 — fine at current scale. The LeRobot v3
    migration (Upgrade Plan Phase 5) replaces this with memory-mapped streaming.
    """

    def __init__(self, demos_dir: str, filter_success: bool = False):
        import cv2
        import h5py

        ep_files = sorted(f for f in os.listdir(demos_dir) if f.endswith(".hdf5"))
        if not ep_files:
            raise FileNotFoundError(f"No HDF5 files in {demos_dir}")

        self.episodes: list[dict] = []
        self.task_labels: list[str] = []
        skipped = 0

        for fname in ep_files:
            path = os.path.join(demos_dir, fname)
            with h5py.File(path, "r") as f:
                if filter_success and not bool(f.attrs.get("success", True)):
                    skipped += 1
                    continue
                positions = f["obs/joint_positions"][:].astype(np.float32)
                velocities = f["obs/joint_velocities"][:].astype(np.float32)
                actions = f["action"][:].astype(np.float32)
                images_raw = f["obs/camera_frames"][:]
                task = str(f.attrs["task_description"])

            if task not in self.task_labels:
                self.task_labels.append(task)

            T = len(positions)
            images = np.empty((T, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for i in range(T):
                images[i] = cv2.resize(
                    images_raw[i], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
                )

            self.episodes.append(
                {
                    "positions": positions,
                    "velocities": velocities,
                    "actions": actions,
                    "images": images,
                    "task": task,
                    "length": T,
                }
            )

        if not self.episodes:
            raise ValueError(f"All episodes filtered out in {demos_dir}")

        self.task_labels.sort()
        self.state_dim = 2 * self.episodes[0]["positions"].shape[1]
        self.action_dim = self.episodes[0]["actions"].shape[1]
        for ep in self.episodes:
            if 2 * ep["positions"].shape[1] != self.state_dim:
                raise ValueError("Mixed state dimensions across episodes")

        print(
            f"Loaded {len(self.episodes)} episodes "
            f"({skipped} filtered), state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, tasks={self.task_labels}"
        )

    def task_id(self, label: str) -> int:
        return self.task_labels.index(label)

    def split(self, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
        """Stratified-by-task episode split → (train_indices, val_indices).

        Guarantees ≥1 validation episode per task whenever a task has ≥2
        episodes (so validation loss covers the full task set).
        """
        rng = np.random.default_rng(seed)
        by_task: dict[str, list[int]] = defaultdict(list)
        for idx, ep in enumerate(self.episodes):
            by_task[ep["task"]].append(idx)

        train_idx: list[int] = []
        val_idx: list[int] = []
        for label in sorted(by_task):
            idxs = np.array(by_task[label])
            rng.shuffle(idxs)
            n_val = max(1, round(len(idxs) * val_frac)) if len(idxs) > 1 else 0
            val_idx.extend(idxs[:n_val].tolist())
            train_idx.extend(idxs[n_val:].tolist())
        return sorted(train_idx), sorted(val_idx)

    def norm_stats(self, episode_indices: list[int]) -> NormStats:
        """Compute state/action statistics over the given (training) episodes."""
        eps = [self.episodes[i] for i in episode_indices]
        states = (np.concatenate([ep["positions"], ep["velocities"]], axis=1) for ep in eps)
        actions = (ep["actions"] for ep in eps)
        return compute_norm_stats(states, actions)


class ChunkDataset(Dataset):
    """Per-frame samples with action chunks over a subset of episodes.

    Returns ``(image, state, task_id, action_chunk)``:
      image:        (3, 224, 224) float32, ImageNet-normalized (+ augmentation)
      state:        (state_dim,) float32 RAW (the model normalizes internally)
      task_id:      int
      action_chunk: (chunk_size, action_dim) float32 RAW
    """

    def __init__(
        self,
        store: EpisodeStore,
        episode_indices: list[int],
        chunk_size: int = CHUNK_SIZE,
        augment: bool = False,
    ):
        from torchvision import transforms as tvt

        self.store = store
        self.chunk_size = chunk_size
        self.augment = augment
        self.samples = [
            (ep_idx, t)
            for ep_idx in episode_indices
            for t in range(store.episodes[ep_idx]["length"])
        ]
        if not self.samples:
            raise ValueError("Empty dataset split")

        self._normalize = tvt.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if augment:
            self._augment_tf = tvt.Compose(
                [
                    tvt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                    tvt.RandomApply([tvt.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
                    tvt.RandomApply(
                        [
                            tvt.RandomResizedCrop(
                                IMAGE_SIZE,
                                scale=(0.85, 1.0),
                                ratio=(0.95, 1.05),
                                antialias=True,
                            )
                        ],
                        p=0.5,
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ep_idx, t = self.samples[idx]
        ep = self.store.episodes[ep_idx]
        T = ep["length"]

        img = torch.from_numpy(ep["images"][t].transpose(2, 0, 1).copy()).float() / 255.0
        if self.augment:
            img = self._augment_tf(img)
        img = self._normalize(img)

        state = torch.from_numpy(np.concatenate([ep["positions"][t], ep["velocities"][t]]))

        end = min(t + self.chunk_size, T)
        chunk = ep["actions"][t:end]
        if len(chunk) < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad])
        chunk = torch.from_numpy(chunk.copy())

        return img, state, self.store.task_id(ep["task"]), chunk
