"""State/action normalization from dataset statistics.

The legacy pipeline fed raw radians to the network and regressed raw radians
out — workable for O(1)-scale arm joints, but it breaks on any differently
scaled action space and diverges from how every modern policy stack
(LeRobot, ACT reference, Diffusion Policy) handles observations. Statistics
are computed on the *training* split only and stored in the checkpoint so
inference applies exactly the same transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MIN_STD = 1e-4  # floor to keep near-constant dims (e.g. frozen joints) stable


@dataclass
class NormStats:
    """Per-dimension mean/std for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self) -> None:
        self.state_std = np.maximum(self.state_std, _MIN_STD)
        self.action_std = np.maximum(self.action_std, _MIN_STD)

    # ── transforms ──

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean

    # ── (de)serialization ──

    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        return cls(
            state_mean=np.asarray(d["state_mean"], dtype=np.float32),
            state_std=np.asarray(d["state_std"], dtype=np.float32),
            action_mean=np.asarray(d["action_mean"], dtype=np.float32),
            action_std=np.asarray(d["action_std"], dtype=np.float32),
        )

    @classmethod
    def identity(cls, state_dim: int, action_dim: int) -> NormStats:
        """No-op stats, used when loading legacy (unnormalized) checkpoints."""
        return cls(
            state_mean=np.zeros(state_dim, dtype=np.float32),
            state_std=np.ones(state_dim, dtype=np.float32),
            action_mean=np.zeros(action_dim, dtype=np.float32),
            action_std=np.ones(action_dim, dtype=np.float32),
        )


@dataclass
class _RunningMoments:
    """Accumulates mean/std over arrays without holding them all in memory."""

    count: int = 0
    _sum: np.ndarray | None = field(default=None, repr=False)
    _sumsq: np.ndarray | None = field(default=None, repr=False)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        if self._sum is None:
            self._sum = np.zeros(x.shape[-1])
            self._sumsq = np.zeros(x.shape[-1])
        self.count += x.shape[0]
        self._sum += x.sum(axis=0)
        self._sumsq += (x**2).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("no data accumulated")
        mean = self._sum / self.count
        var = np.maximum(self._sumsq / self.count - mean**2, 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_norm_stats(states_iter, actions_iter) -> NormStats:
    """Compute :class:`NormStats` from iterables of (T, D) arrays."""
    sm = _RunningMoments()
    am = _RunningMoments()
    for s in states_iter:
        sm.update(s)
    for a in actions_iter:
        am.update(a)
    state_mean, state_std = sm.finalize()
    action_mean, action_std = am.finalize()
    return NormStats(state_mean, state_std, action_mean, action_std)
