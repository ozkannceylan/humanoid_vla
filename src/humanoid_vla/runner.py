"""The single temporal-ensembling implementation.

Previously this logic was copy-pasted across five files (``evaluate.py``,
``evaluate_bimanual.py``, ``live_demo.py``, ``live_bimanual.py``, and the ROS
task manager). All inference loops should construct a
:class:`TemporalEnsembler` instead.

Algorithm (ACT, Zhao et al. RSS 2023): the policy predicts a chunk of
``chunk_size`` future actions every ``chunk_exec`` steps; predictions from
overlapping chunks that cover the same timestep are blended with exponential
weights ``w_i = exp(-k * i)`` where ``i`` is the offset *within its chunk* —
so fresher predictions (small offset) dominate.
"""

from __future__ import annotations

import numpy as np

from humanoid_vla.constants import CHUNK_EXEC, ENSEMBLE_K


class TemporalEnsembler:
    """Blends overlapping action chunks with exponential decay weighting."""

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        max_steps: int,
        k: float = ENSEMBLE_K,
        chunk_exec: int = CHUNK_EXEC,
    ):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.k = k
        self.chunk_exec = chunk_exec
        total = max_steps + chunk_size
        self._action_sum = np.zeros((total, action_dim), dtype=np.float64)
        self._weight_sum = np.zeros(total, dtype=np.float64)
        # Precomputed within-chunk weights.
        self._weights = np.exp(-k * np.arange(chunk_size))

    def needs_replan(self, step: int) -> bool:
        """True when the policy should be queried for a fresh chunk."""
        return step % self.chunk_exec == 0

    def add_chunk(self, step: int, chunk: np.ndarray) -> None:
        """Register a freshly predicted chunk starting at ``step``."""
        chunk = np.asarray(chunk)
        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(f"chunk shape {chunk.shape} != ({self.chunk_size}, {self.action_dim})")
        sl = slice(step, step + self.chunk_size)
        self._action_sum[sl] += self._weights[:, None] * chunk
        self._weight_sum[sl] += self._weights

    def get_action(self, step: int) -> np.ndarray:
        """Blended action for ``step`` (zeros if nothing covers it yet)."""
        w = self._weight_sum[step]
        if w <= 0:
            return np.zeros(self.action_dim)
        return self._action_sum[step] / w

    def reset_from(self, step: int) -> None:
        """Discard all predictions for timesteps > ``step``.

        Used by hierarchical task decomposition when the task embedding
        switches mid-episode (approach -> lift): stale approach-phase chunks
        must not blend into the new phase.
        """
        self._action_sum[step + 1 :] = 0.0
        self._weight_sum[step + 1 :] = 0.0
