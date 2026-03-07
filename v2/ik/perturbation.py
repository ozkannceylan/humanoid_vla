"""v2/ik/perturbation.py — Trajectory variation for bimanual episodes.

Kept minimal: the BimanualGraspParams dataclass and sample_grasp_params()
in grasp_strategies.py handle all per-episode variation for the bimanual
pipeline.  This module is retained for backward compatibility but simply
re-exports the relevant pieces.
"""
from __future__ import annotations

from v2.ik.grasp_strategies import BimanualGraspParams, sample_grasp_params

__all__ = ["BimanualGraspParams", "sample_grasp_params"]
