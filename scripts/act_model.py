#!/usr/bin/env python3
"""
scripts/act_model.py

ACT (Action Chunking with Transformers) policy for language-conditioned
manipulation of a Unitree G1 humanoid right arm.

Architecture:
  - ResNet18 image encoder (frozen except last block)
  - State MLP encoder (29 joint positions + 29 joint velocities = 58 dims)
  - Task embedding (learned per-task vector, enabling language conditioning)
  - Transformer decoder with chunk_size query tokens
  - Linear action head → predicted action chunk (chunk_size × 29)

The model predicts a *chunk* of future actions (not just one), then at inference
time only the first action is executed. This is the "action chunking" idea from
the ACT paper — it provides temporal coherence and reduces compounding errors.

References:
  - ACT: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware", RSS 2023 (https://arxiv.org/abs/2304.13705)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as tvt
from torchvision.models import resnet18, ResNet18_Weights

try:
    import h5py
except ImportError:
    raise SystemExit("h5py required: pip3 install --break-system-packages h5py")

try:
    import cv2
except ImportError:
    raise SystemExit("opencv required: pip3 install --break-system-packages opencv-python")


# ────────────────────────────────────────────────────────
# Task label registry
# ────────────────────────────────────────────────────────

TASK_LABELS = [
    "reach the red cube",
    "grasp the red cube",
    "pick up the red cube",
    "place the red cube on the blue plate",
]


def task_to_id(task_str: str) -> int:
    """Map task description string to integer ID."""
    try:
        return TASK_LABELS.index(task_str)
    except ValueError:
        raise ValueError(f"Unknown task: '{task_str}'. Known: {TASK_LABELS}")


# ────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────

# ImageNet normalization constants
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class DemoDataset(Dataset):
    """Loads HDF5 demo episodes for ACT training.

    All data is preloaded into RAM at init time for fast training.
    Images are resized to 224×224 at load time (~515 MB for 60 episodes).

    Each sample returns:
      image:        (3, 224, 224) float32 tensor (ImageNet-normalized)
      state:        (58,) float32 tensor (29 pos + 29 vel)
      task_id:      int
      action_chunk: (chunk_size, 29) float32 tensor
    """

    def __init__(self, demos_dir: str, chunk_size: int = 20, augment: bool = False):
        self.chunk_size = chunk_size
        self.augment = augment
        if augment:
            self.color_jitter = tvt.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            self.gaussian_blur = tvt.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        ep_files = sorted(
            f for f in os.listdir(demos_dir)
            if f.endswith('.hdf5')
        )
        if not ep_files:
            raise FileNotFoundError(f"No HDF5 files in {demos_dir}")

        self.episodes = []
        self.samples = []   # (ep_idx, frame_idx)

        print(f"Loading {len(ep_files)} episodes from {demos_dir} ...")

        for ep_idx, fname in enumerate(ep_files):
            path = os.path.join(demos_dir, fname)
            with h5py.File(path, 'r') as f:
                positions = f['obs/joint_positions'][:]       # (T, 29)
                velocities = f['obs/joint_velocities'][:]     # (T, 29)
                actions = f['action'][:]                      # (T, 29)
                images_raw = f['obs/camera_frames'][:]        # (T, 480, 640, 3)
                task = str(f.attrs['task_description'])

            T = len(positions)
            tid = task_to_id(task)

            # Resize images to 224×224 immediately (saves RAM vs keeping 640×480)
            images = np.empty((T, 224, 224, 3), dtype=np.uint8)
            for i in range(T):
                images[i] = cv2.resize(images_raw[i], (224, 224),
                                       interpolation=cv2.INTER_AREA)

            self.episodes.append({
                'positions': positions.astype(np.float32),
                'velocities': velocities.astype(np.float32),
                'actions': actions.astype(np.float32),
                'images': images,
                'task_id': tid,
                'length': T,
            })

            for t in range(T):
                self.samples.append((ep_idx, t))

        n_tasks = len(set(ep['task_id'] for ep in self.episodes))
        print(f"  {len(self.episodes)} episodes, {len(self.samples)} samples, "
              f"{n_tasks} tasks")

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

        # State: 29 pos + 29 vel = 58
        state = np.concatenate([ep['positions'][t], ep['velocities'][t]])
        state = torch.from_numpy(state)

        # Task ID
        task_id = ep['task_id']

        # Action chunk: actions[t:t+chunk_size], pad with last action
        end = min(t + self.chunk_size, T)
        chunk = ep['actions'][t:end].copy()
        if len(chunk) < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad])
        chunk = torch.from_numpy(chunk)

        return img, state, task_id, chunk


# ────────────────────────────────────────────────────────
# ACT Policy Model
# ────────────────────────────────────────────────────────

class ACTPolicy(nn.Module):
    """Action Chunking with Transformers — simplified, deterministic version.

    Unlike the original ACT paper which uses a CVAE (variational autoencoder),
    this version is deterministic (no latent variable z). This works well for
    scripted expert demonstrations that have low variance — there's only one
    "correct" action for each state, so we don't need the multimodal capacity
    of a CVAE.

    Architecture:
      Encode image → 1 token (512→256 via ResNet18 + proj)
      Encode state → 1 token (58→256 via MLP)
      Encode task  → 1 token (embedding lookup)
      Memory = [img_tok, state_tok, task_tok] — 3 tokens

      Queries = chunk_size learnable embeddings
      Transformer decoder: queries attend to memory → chunk_size output tokens
      Linear head: each output → action_dim

    Total params: ~6M trainable (ResNet18 layer4 + decoder + heads)
    VRAM usage: ~1.5 GB at batch_size=32
    """

    def __init__(self, state_dim: int = 58, action_dim: int = 29,
                 chunk_size: int = 20, hidden_dim: int = 256,
                 nhead: int = 4, num_layers: int = 4, num_tasks: int = 8):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # ── Image encoder: ResNet18 → avgpool → project ──
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.img_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.img_proj = nn.Linear(512, hidden_dim)

        # Freeze early layers — only fine-tune layer4 (saves memory + prevents overfitting)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.img_encoder[7].parameters():   # layer4
            param.requires_grad = True

        # ── State encoder ──
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Task encoder (learned embedding per task) ──
        self.task_embed = nn.Embedding(num_tasks, hidden_dim)

        # ── Transformer decoder ──
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ── Action output head ──
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, image, state, task_id):
        """
        Args:
            image:   (B, 3, 224, 224) float32
            state:   (B, 58) float32
            task_id: (B,) long

        Returns:
            actions: (B, chunk_size, action_dim) float32
        """
        B = image.shape[0]

        # Encode inputs → 1 token each
        img_feat = self.img_encoder(image).flatten(1)          # (B, 512)
        img_tok = self.img_proj(img_feat).unsqueeze(1)         # (B, 1, D)
        state_tok = self.state_proj(state).unsqueeze(1)        # (B, 1, D)
        task_tok = self.task_embed(task_id).unsqueeze(1)       # (B, 1, D)

        # Memory: 3 context tokens
        memory = torch.cat([img_tok, state_tok, task_tok], dim=1)  # (B, 3, D)

        # Queries: one per future action timestep
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode: queries attend to memory → action features
        out = self.decoder(queries, memory)         # (B, chunk, D)
        actions = self.action_head(out)             # (B, chunk, action_dim)
        return actions

    @torch.no_grad()
    def predict(self, image: np.ndarray, state: np.ndarray,
                task_id: int, device: str = 'cuda') -> np.ndarray:
        """Single-step inference for evaluation.

        Args:
            image:   (H, W, 3) uint8 RGB
            state:   (58,) float32
            task_id: int

        Returns:
            actions: (chunk_size, 29) float32 numpy
        """
        self.eval()

        # Preprocess image: resize → normalize → tensor
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = (img - _IMG_MEAN) / _IMG_STD
        img_t = torch.from_numpy(img).unsqueeze(0).to(device)

        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        task_t = torch.tensor([task_id], dtype=torch.long, device=device)

        actions = self(img_t, state_t, task_t)
        return actions[0].cpu().numpy()
