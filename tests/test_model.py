import numpy as np
import pytest
import torch

from humanoid_vla.models.act import ACTPolicy
from humanoid_vla.normalization import NormStats

B, STATE, ACT, CHUNK = 2, 28, 14, 20


def _model(**kw):
    defaults = dict(
        state_dim=STATE,
        action_dim=ACT,
        chunk_size=CHUNK,
        num_tasks=2,
        pretrained_backbone=False,
    )
    defaults.update(kw)
    return ACTPolicy(**defaults)


def _batch():
    img = torch.randn(B, 3, 224, 224)
    state = torch.randn(B, STATE)
    return img, state


@pytest.mark.parametrize("vision", ["pooled", "spatial"])
def test_forward_shapes_task_id(vision):
    model = _model(vision_tokens=vision)
    img, state = _batch()
    out = model(img, state, torch.tensor([0, 1]))
    assert out.shape == (B, CHUNK, ACT)


def test_forward_text_conditioning():
    model = _model(conditioning="text", text_dim=512)
    img, state = _batch()
    out = model(img, state, torch.randn(B, 512))
    assert out.shape == (B, CHUNK, ACT)


def test_text_mode_rejects_ids_and_vice_versa():
    img, state = _batch()
    with pytest.raises(ValueError):
        _model(conditioning="text")(img, state, torch.tensor([0, 1]))
    with pytest.raises(TypeError):
        _model()(img, state, torch.randn(B, 512))


def test_backbone_frozen_except_layer4():
    model = _model()
    frozen = [n for n, p in model.img_encoder.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in model.img_encoder.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("7.") for n in trainable)  # layer4 only
    assert frozen and not any(n.startswith("7.") for n in frozen)


def test_param_count_matches_documented_scale():
    # Single-arm defaults must match the documented ~15.6M / ~12.8M split
    # (docs/CODEBASE_REVIEW.md §2.6 fixed the old "~6M" claim).
    model = ACTPolicy(
        state_dim=58,
        action_dim=29,
        num_tasks=4,
        vision_tokens="pooled",
        pretrained_backbone=False,
    )
    total, trainable = model.count_params()
    assert 15.0e6 < total < 16.5e6
    assert 12.0e6 < trainable < 13.5e6


def test_normalization_applied_in_predict():
    model = _model()
    stats = NormStats(
        state_mean=np.zeros(STATE, dtype=np.float32),
        state_std=np.ones(STATE, dtype=np.float32),
        action_mean=np.full(ACT, 5.0, dtype=np.float32),
        action_std=np.full(ACT, 2.0, dtype=np.float32),
    )
    model.set_norm_stats(stats)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    state = np.zeros(STATE, dtype=np.float32)
    chunk = model.predict(image, state, 0, device="cpu")
    assert chunk.shape == (CHUNK, ACT)
    # Raw network outputs are ~N(0,1)-scale; denormalized outputs must be
    # pulled toward action_mean=5 — i.e. clearly not centered at 0.
    assert abs(chunk.mean() - 5.0) < 2.0


def test_normalize_denormalize_inverse():
    model = _model()
    stats = NormStats(
        state_mean=np.zeros(STATE, dtype=np.float32),
        state_std=np.ones(STATE, dtype=np.float32),
        action_mean=np.random.default_rng(0).normal(size=ACT).astype(np.float32),
        action_std=np.random.default_rng(1).uniform(0.5, 2, ACT).astype(np.float32),
    )
    model.set_norm_stats(stats)
    a = torch.randn(3, CHUNK, ACT)
    torch.testing.assert_close(
        model.denormalize_actions(model.normalize_actions(a)), a, rtol=1e-4, atol=1e-5
    )
