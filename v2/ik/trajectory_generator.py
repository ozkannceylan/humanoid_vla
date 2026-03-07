"""v2/ik/trajectory_generator.py — Bimanual 6-phase trajectory planner.

Phases:
  1. Home -> Pre-approach   (above + to sides of box)
  2. Pre-approach -> Approach (at box height, ~5cm out)
  3. Approach -> Squeeze     (palms press into box sides)
  4. Squeeze hold            (stabilise grasp)
  5. Squeeze -> Lift         (maintain squeeze, raise hands)
  6. Hold at top             (box held aloft)

IK is solved independently for each arm.  Trajectories are linear
interpolation in joint space (matching the proven generate_bimanual_demos.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import mujoco
import numpy as np

from v2.common import LEFT_ARM_CTRL, RIGHT_ARM_CTRL
from v2.env.table_env_v2 import TableEnvV2
from v2.ik.grasp_strategies import BimanualGraspParams, sample_grasp_params


@dataclass
class BimanualPlan:
    """Result of bimanual trajectory planning."""
    left_traj: np.ndarray     # (T, 7) left arm joint targets
    right_traj: np.ndarray    # (T, 7) right arm joint targets
    n_frames: int
    phase_ends: list[int]
    params: BimanualGraspParams
    success: bool
    failure_reason: str | None


def interpolate(q0: np.ndarray, q1: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear interpolation in joint space.  Returns (n_frames+1, D)."""
    t = np.linspace(0, 1, n_frames + 1).reshape(-1, 1)
    return q0 + t * (q1 - q0)


def plan_bimanual_trajectory(
    env: TableEnvV2,
    box_pos: np.ndarray,
    rng: np.random.Generator,
    params: BimanualGraspParams | None = None,
    q_home_L: np.ndarray | None = None,
    q_home_R: np.ndarray | None = None,
    config: dict | None = None,
) -> BimanualPlan:
    """Plan a bimanual squeeze-and-lift trajectory.

    Returns a BimanualPlan with left/right trajectories and success flag.
    """
    if params is None:
        params = sample_grasp_params(rng, config)

    box_pos = np.asarray(box_pos, dtype=np.float64)

    # Waypoint targets (world coordinates)
    pre_L = box_pos + np.array([0, params.pre_approach_offset_y, params.pre_approach_offset_z])
    pre_R = box_pos + np.array([0, -params.pre_approach_offset_y, params.pre_approach_offset_z])
    app_L = box_pos + np.array([0, params.approach_offset_y, 0])
    app_R = box_pos + np.array([0, -params.approach_offset_y, 0])
    sq_L  = box_pos + np.array([0, params.squeeze_offset_y, 0])
    sq_R  = box_pos + np.array([0, -params.squeeze_offset_y, 0])
    lft_L = box_pos + np.array([0, params.squeeze_offset_y, params.lift_offset_z])
    lft_R = box_pos + np.array([0, -params.squeeze_offset_y, params.lift_offset_z])

    # ---- Solve IK for left arm waypoints ----
    if q_home_L is None:
        q_home_L = np.zeros(7)
    env.data.qpos[env.left_arm_qpos_adr] = q_home_L
    mujoco.mj_forward(env.model, env.data)

    ik_results_L = []
    for name, target in [("pre_L", pre_L), ("app_L", app_L),
                          ("sq_L", sq_L), ("lft_L", lft_L)]:
        ok = env.solve_ik_left(target)
        if not ok:
            return BimanualPlan(
                left_traj=np.zeros((0, 7)), right_traj=np.zeros((0, 7)),
                n_frames=0, phase_ends=[], params=params,
                success=False, failure_reason=f"ik_failed:{name}")
        ik_results_L.append(env.left_arm_q.copy())
    q_pre_L, q_app_L, q_sq_L, q_lft_L = ik_results_L

    # ---- Solve IK for right arm waypoints ----
    if q_home_R is None:
        q_home_R = np.zeros(7)
    env.data.qpos[env.right_arm_qpos_adr] = q_home_R
    mujoco.mj_forward(env.model, env.data)

    ik_results_R = []
    for name, target in [("pre_R", pre_R), ("app_R", app_R),
                          ("sq_R", sq_R), ("lft_R", lft_R)]:
        ok = env.solve_ik_right(target)
        if not ok:
            return BimanualPlan(
                left_traj=np.zeros((0, 7)), right_traj=np.zeros((0, 7)),
                n_frames=0, phase_ends=[], params=params,
                success=False, failure_reason=f"ik_failed:{name}")
        ik_results_R.append(env.right_arm_q.copy())
    q_pre_R, q_app_R, q_sq_R, q_lft_R = ik_results_R

    # ---- Build trajectories via linear interpolation ----
    n1, n2, n3, n4, n5, n6 = (
        params.frames_home_to_pre, params.frames_pre_to_approach,
        params.frames_approach_to_squeeze, params.frames_squeeze_hold,
        params.frames_squeeze_to_lift, params.frames_hold_top,
    )

    segments_L = [
        interpolate(q_home_L, q_pre_L, n1),
        interpolate(q_pre_L,  q_app_L, n2),
        interpolate(q_app_L,  q_sq_L,  n3),
        interpolate(q_sq_L,   q_sq_L,  n4),   # hold
        interpolate(q_sq_L,   q_lft_L, n5),
        interpolate(q_lft_L,  q_lft_L, n6),   # hold at top
    ]
    segments_R = [
        interpolate(q_home_R, q_pre_R, n1),
        interpolate(q_pre_R,  q_app_R, n2),
        interpolate(q_app_R,  q_sq_R,  n3),
        interpolate(q_sq_R,   q_sq_R,  n4),
        interpolate(q_sq_R,   q_lft_R, n5),
        interpolate(q_lft_R,  q_lft_R, n6),
    ]

    # Concatenate (skip duplicate endpoints between segments)
    left_traj = np.concatenate(
        [seg[:-1] for seg in segments_L[:-1]] + [segments_L[-1]])
    right_traj = np.concatenate(
        [seg[:-1] for seg in segments_R[:-1]] + [segments_R[-1]])

    # Phase boundary frame indices
    phase_ends: list[int] = []
    total = 0
    for seg in segments_L:
        total += len(seg) - 1
        phase_ends.append(total)
    phase_ends[-1] = len(left_traj) - 1

    return BimanualPlan(
        left_traj=left_traj.astype(np.float32),
        right_traj=right_traj.astype(np.float32),
        n_frames=len(left_traj),
        phase_ends=phase_ends,
        params=params,
        success=True,
        failure_reason=None,
    )
