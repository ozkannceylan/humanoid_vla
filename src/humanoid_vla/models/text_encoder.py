"""Frozen CLIP text encoder for instruction conditioning.

Used at *training time* to embed instruction paraphrases; the resulting
embeddings for every task's train/held-out paraphrases are stored inside the
checkpoint (``instruction_bank``), so evaluation and the ROS node need neither
``transformers`` nor network access. Encoding novel free-form instructions at
inference time does require this module.

Requires the ``lang`` extra: ``pip install -e .[lang]``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEXT_MODEL = "openai/clip-vit-base-patch32"
TEXT_EMBED_DIM = 512


class TextEncoder:
    """Lazy wrapper around a frozen CLIP text tower (projected, L2-normalized)."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = "cpu"):
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        except ImportError as e:
            raise ImportError(
                "Text conditioning requires the 'lang' extra: pip install -e .[lang]"
            ) from e

        self.device = device
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of instructions → (N, 512) float32, L2-normalized."""
        import torch

        missing = [t for t in texts if t not in self._cache]
        if missing:
            tokens = self.tokenizer(missing, padding=True, truncation=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                out = self.model(**tokens).text_embeds  # (N, 512)
                out = out / out.norm(dim=-1, keepdim=True)
            for text, emb in zip(missing, out.cpu().numpy().astype(np.float32), strict=True):
                self._cache[text] = emb
        return np.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_instruction_bank(
    task_labels: list[str], encoder: TextEncoder
) -> dict[str, dict[str, np.ndarray | list]]:
    """Precompute embeddings for every task's paraphrases.

    Returns ``{label: {"canonical": (512,), "train": (N,512), "heldout": (M,512)}}``
    ready to be stored in a checkpoint.
    """
    from humanoid_vla.instructions import get_instructions

    bank: dict[str, dict] = {}
    for label in task_labels:
        train = get_instructions(label, "train")
        heldout = get_instructions(label, "heldout")
        bank[label] = {
            "canonical": encoder.encode_one(label),
            "train": encoder.encode(train),
            "heldout": encoder.encode(heldout),
            "train_texts": train,
            "heldout_texts": heldout,
        }
    return bank
