"""ACT policy v2 — action-chunking transformer with optional language conditioning.

Improvements over the legacy ``scripts/act_model.py`` implementation:

- **Spatial vision tokens** (``vision_tokens="spatial"``): the ResNet18 layer4
  feature map (7x7x512) is projected 1x1 to the model dim and fed to the
  decoder as 49 tokens with a learned 2D positional embedding — matching the
  ACT paper, instead of destroying all spatial structure with global avgpool.
  ``vision_tokens="pooled"`` reproduces the legacy single-token behavior and
  remains checkpoint-compatible with old checkpoints.
- **Text conditioning** (``conditioning="text"``): the task token is projected
  from a frozen text-encoder embedding (e.g. CLIP text, 512-d) instead of an
  integer task-id lookup, enabling paraphrase-robust instruction following.
  ``conditioning="task_id"`` keeps the legacy integer embedding as an ablation
  baseline.
- **Built-in normalization**: dataset state/action statistics are stored as
  buffers inside the module (and therefore inside every checkpoint). ``forward``
  consumes *raw* states and predicts *normalized* action chunks; ``predict``
  returns denormalized actions. Legacy checkpoints load with identity stats.

Parameter count (defaults, pooled + task_id): ~15.6M total, ~12.8M trainable
(ResNet18 backbone 11.2M of which layer4 8.4M is fine-tuned; decoder 4.2M).
The old "~6M trainable" docstring was wrong — see docs/CODEBASE_REVIEW.md §2.6.

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware", RSS 2023 (https://arxiv.org/abs/2304.13705).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from humanoid_vla.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from humanoid_vla.normalization import NormStats

_SPATIAL_TOKENS = 49  # ResNet18 layer4 output for 224x224 input: 7x7


class ACTPolicy(nn.Module):
    """Deterministic ACT: transformer decoder over vision/state/task tokens.

    Args:
        state_dim: proprioception dim (pos+vel; 58 single-arm, 28 bimanual).
        action_dim: joint-target dim (29 single-arm, 14 bimanual).
        chunk_size: number of future actions predicted per query.
        hidden_dim: transformer model dim.
        nhead / num_layers: decoder shape.
        num_tasks: task vocabulary size (``conditioning="task_id"`` only).
        conditioning: ``"task_id"`` (integer embedding) or ``"text"`` (frozen
            text-encoder embeddings passed to :meth:`forward`).
        text_dim: dimensionality of the text embeddings (text mode).
        vision_tokens: ``"pooled"`` (1 token, legacy) or ``"spatial"``
            (49 tokens + positional embedding, ACT-paper style).
        pretrained_backbone: load ImageNet weights (disable in unit tests).
    """

    def __init__(
        self,
        state_dim: int = 58,
        action_dim: int = 29,
        chunk_size: int = 20,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        num_tasks: int = 4,
        conditioning: str = "task_id",
        text_dim: int = 512,
        vision_tokens: str = "spatial",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if conditioning not in ("task_id", "text"):
            raise ValueError(f"conditioning must be 'task_id' or 'text', got {conditioning!r}")
        if vision_tokens not in ("pooled", "spatial"):
            raise ValueError(f"vision_tokens must be 'pooled' or 'spatial', got {vision_tokens!r}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.conditioning = conditioning
        self.vision_tokens = vision_tokens

        # ── Vision encoder: ResNet18 backbone ──
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        if vision_tokens == "pooled":
            # conv1..layer4 + avgpool (legacy layout; avgpool holds no params,
            # so backbone state-dict keys match the spatial variant).
            self.img_encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.img_proj = nn.Linear(512, hidden_dim)
        else:
            # conv1..layer4 only — keep the 7x7 feature map as 49 tokens.
            self.img_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.img_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
            self.img_pos_embed = nn.Parameter(torch.randn(1, _SPATIAL_TOKENS, hidden_dim) * 0.02)

        # Freeze everything except layer4 (index 7 in both layouts).
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.img_encoder[7].parameters():
            param.requires_grad = True

        # ── State encoder ──
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Task conditioning ──
        if conditioning == "task_id":
            self.task_embed = nn.Embedding(num_tasks, hidden_dim)
        else:
            self.task_proj = nn.Linear(text_dim, hidden_dim)

        # ── Transformer decoder over chunk_size learned queries ──
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # ── Normalization buffers (persisted in every checkpoint) ──
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

        # Image normalization as buffers too (avoids numpy round-trips).
        self.register_buffer(
            "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # ── normalization helpers ──

    def set_norm_stats(self, stats: NormStats) -> None:
        self.state_mean.copy_(torch.from_numpy(stats.state_mean))
        self.state_std.copy_(torch.from_numpy(stats.state_std))
        self.action_mean.copy_(torch.from_numpy(stats.action_mean))
        self.action_std.copy_(torch.from_numpy(stats.action_std))

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize raw action targets for loss computation."""
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    # ── forward ──

    def _encode_task(self, task: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "task_id":
            if task.dtype not in (torch.int32, torch.int64):
                raise TypeError("task_id conditioning expects integer task ids")
            return self.task_embed(task)
        if task.dim() != 2:
            raise ValueError("text conditioning expects (B, text_dim) embeddings")
        return self.task_proj(task)

    def forward(self, image: torch.Tensor, state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Predict a normalized action chunk.

        Args:
            image: (B, 3, H, W) float32, already ImageNet-normalized.
            state: (B, state_dim) float32, RAW (normalized internally).
            task:  (B,) long task ids, or (B, text_dim) float text embeddings.

        Returns:
            (B, chunk_size, action_dim) normalized actions.
        """
        B = image.shape[0]
        state = (state - self.state_mean) / self.state_std

        feat = self.img_encoder(image)
        if self.vision_tokens == "pooled":
            img_toks = self.img_proj(feat.flatten(1)).unsqueeze(1)  # (B, 1, D)
        else:
            proj = self.img_proj(feat)  # (B, D, 7, 7)
            img_toks = proj.flatten(2).transpose(1, 2)  # (B, 49, D)
            img_toks = img_toks + self.img_pos_embed

        state_tok = self.state_proj(state).unsqueeze(1)
        task_tok = self._encode_task(task).unsqueeze(1)
        memory = torch.cat([img_toks, state_tok, task_tok], dim=1)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return self.action_head(out)

    # ── inference ──

    def preprocess_image(self, image: np.ndarray, device: torch.device) -> torch.Tensor:
        """(H, W, 3) uint8 RGB → (1, 3, 224, 224) normalized float tensor."""
        import cv2

        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(image.transpose(2, 0, 1).copy()).to(device).float() / 255.0
        return ((t.unsqueeze(0) - self.img_mean) / self.img_std).float()

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        state: np.ndarray,
        task: int | np.ndarray,
        device: str = "cuda",
    ) -> np.ndarray:
        """Single-observation inference returning a DENORMALIZED action chunk.

        Args:
            image: (H, W, 3) uint8 RGB.
            state: (state_dim,) float raw proprioception.
            task: int task id (task_id mode) or (text_dim,) embedding (text mode).

        Returns:
            (chunk_size, action_dim) float32 numpy actions in raw joint space.
        """
        self.eval()
        img_t = self.preprocess_image(image, device)
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        if self.conditioning == "task_id":
            task_t = torch.tensor([int(task)], dtype=torch.long, device=device)
        else:
            task_t = torch.from_numpy(np.asarray(task, dtype=np.float32)).unsqueeze(0).to(device)
        pred = self(img_t, state_t, task_t)
        return self.denormalize_actions(pred)[0].float().cpu().numpy()

    # ── introspection ──

    def count_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
