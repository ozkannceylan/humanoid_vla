"""v2/ik/grasp_strategies.py — Bimanual squeeze approach parameters.

Defines offset distances, timing variations, and compliance targets for the
bilateral squeeze approach used in bimanual box manipulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from v2.common import (
    PRE_APPROACH_OFFSET_Y,
    PRE_APPROACH_OFFSET_Z,
    APPROACH_OFFSET_Y,
    SQUEEZE_OFFSET_Y,
    LIFT_OFFSET_Z,
)


@dataclass(frozen=True)
class BimanualGraspParams:
    """Parameters for a bimanual squeeze trajectory."""
    # Lateral offsets (Y-axis, relative to box centre)
    pre_approach_offset_y: float = PRE_APPROACH_OFFSET_Y
    pre_approach_offset_z: float = PRE_APPROACH_OFFSET_Z
    approach_offset_y: float = APPROACH_OFFSET_Y
    squeeze_offset_y: float = SQUEEZE_OFFSET_Y
    lift_offset_z: float = LIFT_OFFSET_Z
    # Timing (frames at control Hz)
    frames_home_to_pre: int = 30
    frames_pre_to_approach: int = 20
    frames_approach_to_squeeze: int = 30
    frames_squeeze_hold: int = 10
    frames_squeeze_to_lift: int = 60
    frames_hold_top: int = 20


def sample_grasp_params(rng: np.random.Generator,
                        config: dict | None = None) -> BimanualGraspParams:
    """Sample randomised bimanual grasp parameters.

    Variations:
      - squeeze_offset_y: +-5mm around nominal (how deep palms penetrate)
      - lift_offset_z: +-3cm around nominal
      - timing: +-10% per phase
    """
    cfg = config or {}
    vary = cfg.get("timing_variation", 0.10)

    def _vary(base: int) -> int:
        return max(8, int(base * rng.uniform(1.0 - vary, 1.0 + vary)))

    return BimanualGraspParams(
        squeeze_offset_y=float(rng.uniform(0.035, 0.045)),
        lift_offset_z=float(rng.uniform(0.12, 0.18)),
        frames_home_to_pre=_vary(30),
        frames_pre_to_approach=_vary(20),
        frames_approach_to_squeeze=_vary(30),
        frames_squeeze_hold=_vary(10),
        frames_squeeze_to_lift=_vary(60),
        frames_hold_top=_vary(20),
    )
