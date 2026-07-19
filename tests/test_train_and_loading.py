"""End-to-end: train 2 epochs on synthetic data, reload, run inference.

Also covers legacy-checkpoint loading (scripts/train_act.py format).
"""

import numpy as np
import pytest
import torch

from humanoid_vla.loading import load_policy
from humanoid_vla.train import TrainConfig, parse_args, train


@pytest.fixture
def trained(demo_dir, tmp_path):
    cfg = TrainConfig(
        demos=demo_dir,
        output=str(tmp_path / "ckpt"),
        epochs=2,
        batch_size=8,
        chunk_size=5,
        hidden_dim=32,
        nhead=2,
        num_layers=1,
        val_frac=0.25,
        num_workers=0,
        device="cpu",
        amp=False,
        augment=False,
        pretrained_backbone=False,
        log_freq=1,
        save_freq=1,
    )
    return train(cfg), cfg


def test_train_saves_best_and_latest(trained):
    best_path, _ = trained
    assert best_path.exists()
    assert (best_path.parent / "latest.pt").exists()


def test_checkpoint_is_self_contained(trained):
    best_path, _ = trained
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert ckpt["format"] == "humanoid_vla/v2"
    assert ckpt["val_loss"] is not None
    assert ckpt["config"]["train"]["seed"] == 42
    assert "norm_stats" in ckpt["config"]
    assert ckpt["task_labels"] == ["grasp the red cube", "pick up the red cube"]


def test_load_and_predict(trained):
    best_path, _ = trained
    policy, config = load_policy(str(best_path), device="cpu")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    state = np.zeros(28, dtype=np.float32)
    chunk = policy.predict(image, state, task_id=1)
    assert chunk.shape == (5, 14)
    assert np.isfinite(chunk).all()


def test_norm_stats_survive_reload(trained):
    best_path, _ = trained
    policy, _ = load_policy(str(best_path), device="cpu")
    # Stats came from data with nonzero mean → buffers must not be identity.
    assert policy.model.action_std.abs().sum() != policy.model.action_std.numel()


def test_legacy_checkpoint_loads(tmp_path):
    """A scripts/train_act.py-style checkpoint loads via the compat path."""
    from humanoid_vla.models.act import ACTPolicy

    legacy_model = ACTPolicy(
        state_dim=58,
        action_dim=29,
        chunk_size=20,
        hidden_dim=64,
        nhead=2,
        num_layers=1,
        num_tasks=4,
        conditioning="task_id",
        vision_tokens="pooled",
        pretrained_backbone=False,
    )
    state_dict = legacy_model.state_dict()
    # Legacy checkpoints predate the norm buffers.
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        state_dict.pop(key)
    legacy_ckpt = {
        "model_state_dict": state_dict,
        "epoch": 299,
        "loss": 9e-6,
        "config": {
            "state_dim": 58,
            "action_dim": 29,
            "chunk_size": 20,
            "hidden_dim": 64,
            "nhead": 2,
            "num_layers": 1,
            "num_tasks": 4,
            "task_labels": [
                "reach the red cube",
                "grasp the red cube",
                "pick up the red cube",
                "place the red cube on the blue plate",
            ],
        },
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy_ckpt, path)

    policy, config = load_policy(str(path), device="cpu")
    assert config.get("legacy") is True
    # Identity norm stats for legacy models.
    assert torch.equal(policy.model.action_std, torch.ones(29))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    chunk = policy.predict(image, np.zeros(58, dtype=np.float32), task_id=2)
    assert chunk.shape == (20, 29)


def test_parse_args_round_trip():
    cfg = parse_args(["--demos", "d", "--epochs", "7", "--no-augment", "--conditioning", "text"])
    assert cfg.demos == "d"
    assert cfg.epochs == 7
    assert cfg.augment is False
    assert cfg.conditioning == "text"
