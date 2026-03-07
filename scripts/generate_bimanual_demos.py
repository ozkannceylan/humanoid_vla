#!/usr/bin/env python3
"""
scripts/generate_bimanual_demos.py

Physics-based scripted expert for bimanual box manipulation.
Generates demos of both hands squeezing a box from the sides and lifting it.

Key differences from generate_demos.py:
  - Uses mj_step (real contact forces, friction) instead of kinematic mode
  - Controls BOTH arms (14 DOF) instead of just the right arm (7 DOF)
  - Grasping is via friction only — no weld constraints
  - Actions are 14D (7 left + 7 right arm joint position targets)

Trajectory phases:
  1. Home → pre-approach (above + to sides of box)   [30 frames]
  2. Pre-approach → approach (at box height, 5cm out) [20 frames]
  3. Approach → squeeze (palm pads press box sides)   [30 frames]
  4. Squeeze hold (stabilize grasp)                    [10 frames]
  5. Squeeze → lift (maintain squeeze, raise hands)    [60 frames]
  6. Hold at top                                       [20 frames]
  Total: ~170 frames @ 30Hz ≈ 5.7 seconds

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/generate_bimanual_demos.py --episodes 30
"""

import os
import sys
import argparse
import datetime
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

try:
    import h5py
except ImportError:
    sys.exit("h5py required: pip3 install --break-system-packages h5py")

from physics_sim import (
    PhysicsSim, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, BOTH_ARMS_CTRL,
    CONTROL_HZ,
)
from domain_randomization import DomainRandomizer

# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────

TASK_LABEL = "pick up the green box with both hands"

# Nominal box position (matches g1_with_camera.xml)
BOX_POS_NOMINAL = np.array([0.3, 0.0, 0.875])

# Trajectory timing (frames at 30 Hz)
FRAMES_HOME_TO_PRE     = 30
FRAMES_PRE_TO_APPROACH = 20
FRAMES_APPROACH_TO_SQ  = 30
FRAMES_SQUEEZE_HOLD    = 10
FRAMES_SQUEEZE_TO_LIFT = 60
FRAMES_HOLD_TOP        = 20

# Target offsets relative to box center (Y = lateral, Z = vertical)
PRE_APPROACH_OFFSET_Y = 0.15   # 15cm to each side
PRE_APPROACH_OFFSET_Z = 0.12   # 12cm above box
APPROACH_OFFSET_Y     = 0.12   # 12cm to each side (5cm from surface)
SQUEEZE_OFFSET_Y      = 0.04   # 4cm from center → 3.5cm inside surface
LIFT_OFFSET_Z         = 0.15   # 15cm lift height

# Success thresholds
MIN_LIFT_CM = 3.0    # minimum lift in cm
MIN_FORCE_N = 2.0    # minimum contact force on each palm


# ────────────────────────────────────────────────────────
# Trajectory planning
# ────────────────────────────────────────────────────────

def interpolate(q0: np.ndarray, q1: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear interpolation in joint space. Returns (n_frames+1, D) array
    from q0 (at t=0) to q1 (at t=n_frames)."""
    t = np.linspace(0, 1, n_frames + 1).reshape(-1, 1)
    return q0 + t * (q1 - q0)


def plan_bimanual_trajectory(sim: PhysicsSim, box_pos: np.ndarray,
                             rng: np.random.Generator,
                             q_home_L: np.ndarray = None,
                             q_home_R: np.ndarray = None) -> dict | None:
    """Solve IK for all waypoints and build joint-space trajectories.

    Returns dict with:
      left_traj:  (T, 7) left arm joint targets
      right_traj: (T, 7) right arm joint targets
      phase_ends: list of frame indices where each phase ends
      n_frames:   total number of frames

    Returns None if any IK solution fails (unreachable target).
    """
    # Timing with slight random variation (±10%)
    def vary(base):
        return max(10, int(base * rng.uniform(0.9, 1.1)))

    n1 = vary(FRAMES_HOME_TO_PRE)
    n2 = vary(FRAMES_PRE_TO_APPROACH)
    n3 = vary(FRAMES_APPROACH_TO_SQ)
    n4 = vary(FRAMES_SQUEEZE_HOLD)
    n5 = vary(FRAMES_SQUEEZE_TO_LIFT)
    n6 = vary(FRAMES_HOLD_TOP)

    # Waypoint targets (world coordinates)
    pre_L = box_pos + np.array([0, PRE_APPROACH_OFFSET_Y, PRE_APPROACH_OFFSET_Z])
    pre_R = box_pos + np.array([0, -PRE_APPROACH_OFFSET_Y, PRE_APPROACH_OFFSET_Z])
    app_L = box_pos + np.array([0, APPROACH_OFFSET_Y, 0])
    app_R = box_pos + np.array([0, -APPROACH_OFFSET_Y, 0])
    sq_L  = box_pos + np.array([0, SQUEEZE_OFFSET_Y, 0])
    sq_R  = box_pos + np.array([0, -SQUEEZE_OFFSET_Y, 0])
    lft_L = box_pos + np.array([0, SQUEEZE_OFFSET_Y, LIFT_OFFSET_Z])
    lft_R = box_pos + np.array([0, -SQUEEZE_OFFSET_Y, LIFT_OFFSET_Z])

    # Solve IK sequentially (each uses previous as seed)
    # Left arm chain
    if q_home_L is None:
        q_home_L = np.zeros(7)
    sim.data.qpos[sim.left_arm_qpos_adr] = q_home_L
    mujoco.mj_forward(sim.model, sim.data)

    ik_targets_L = [("pre_L", pre_L), ("app_L", app_L),
                    ("sq_L", sq_L), ("lft_L", lft_L)]
    ik_results_L = []
    for name, target in ik_targets_L:
        ok = sim.solve_ik_left(target)
        if not ok:
            err = np.linalg.norm(target - sim.left_hand_pos)
            return None  # IK failed — skip this episode
        ik_results_L.append(sim.left_arm_q.copy())
    q_pre_L, q_app_L, q_sq_L, q_lft_L = ik_results_L

    # Right arm chain
    if q_home_R is None:
        q_home_R = np.zeros(7)
    sim.data.qpos[sim.right_arm_qpos_adr] = q_home_R
    mujoco.mj_forward(sim.model, sim.data)

    ik_targets_R = [("pre_R", pre_R), ("app_R", app_R),
                    ("sq_R", sq_R), ("lft_R", lft_R)]
    ik_results_R = []
    for name, target in ik_targets_R:
        ok = sim.solve_ik_right(target)
        if not ok:
            err = np.linalg.norm(target - sim.right_hand_pos)
            return None  # IK failed — skip this episode
        ik_results_R.append(sim.right_arm_q.copy())
    q_pre_R, q_app_R, q_sq_R, q_lft_R = ik_results_R

    # Build trajectories by interpolating between waypoints
    segments_L = [
        interpolate(q_home_L, q_pre_L, n1),      # home → pre-approach
        interpolate(q_pre_L,  q_app_L, n2),       # pre → approach
        interpolate(q_app_L,  q_sq_L,  n3),       # approach → squeeze
        interpolate(q_sq_L,   q_sq_L,  n4),       # squeeze hold (constant)
        interpolate(q_sq_L,   q_lft_L, n5),       # squeeze → lift
        interpolate(q_lft_L,  q_lft_L, n6),       # hold at top (constant)
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
    left_traj = np.concatenate([seg[:-1] for seg in segments_L[:-1]] +
                               [segments_L[-1]])
    right_traj = np.concatenate([seg[:-1] for seg in segments_R[:-1]] +
                                [segments_R[-1]])

    # Phase boundary frame indices
    phase_ends = []
    total = 0
    for seg in segments_L:
        total += len(seg) - 1
        phase_ends.append(total)
    phase_ends[-1] = len(left_traj) - 1  # last segment includes endpoint

    return {
        "left_traj": left_traj,
        "right_traj": right_traj,
        "phase_ends": phase_ends,
        "n_frames": len(left_traj),
    }


# ────────────────────────────────────────────────────────
# Episode generation
# ────────────────────────────────────────────────────────

def generate_episode(sim: PhysicsSim, rng: np.random.Generator,
                     noise_x: float = 0.02, noise_y: float = 0.02,
                     random_start: float = 0.0,
                     domain_rand: bool = False) -> dict:
    """Generate one bimanual box manipulation episode.

    Returns dict with HDF5 data arrays + metadata.
    """
    # Reset with noise on box position
    sim.reset_with_noise(rng, noise_x=noise_x, noise_y=noise_y)
    box_pos = sim.box_pos.copy()

    q_home_L = None
    q_home_R = None
    if random_start > 0:
        # Pass box position so random_arm_start rejects unreachable configs
        q_home_L, q_home_R = sim.random_arm_start(
            rng, arm='both', spread=random_start, reach_target=box_pos)

    if domain_rand:
        if not hasattr(sim, 'randomizer'):
            sim.randomizer = DomainRandomizer(sim.model, sim.data)
        sim.randomizer.randomize(rng)

    # Plan trajectory (returns None if IK fails)
    plan = plan_bimanual_trajectory(sim, box_pos, rng,
                                     q_home_L=q_home_L, q_home_R=q_home_R)
    if plan is None:
        # Restore domain rand if needed and signal failure
        if domain_rand and hasattr(sim, 'randomizer'):
            sim.randomizer.restore()
        return None  # caller should retry

    left_traj = plan["left_traj"]
    right_traj = plan["right_traj"]
    n_frames = plan["n_frames"]

    # Reset again (IK changed qpos) and apply same box noise
    box_dx = box_pos[0] - BOX_POS_NOMINAL[0]
    box_dy = box_pos[1] - BOX_POS_NOMINAL[1]
    sim.reset()
    sim.data.qpos[sim.box_qpos_adr + 0] += box_dx
    sim.data.qpos[sim.box_qpos_adr + 1] += box_dy
    mujoco.mj_forward(sim.model, sim.data)
    # Freeze base after resetting box
    sim.data.qpos[:7] = sim._base_qpos
    sim.data.qvel[:6] = 0.0

    # Recording buffers
    joint_pos_frames = []   # (T, 14) — both arms
    joint_vel_frames = []   # (T, 14) — both arms
    camera_frames = []      # (T, H, W, 3)
    action_frames = []      # (T, 14) — both arm targets

    # Playback with physics
    box_init_z = sim.box_pos[2]

    for t in range(n_frames):
        # Set PD targets
        sim.target_pos[LEFT_ARM_CTRL] = left_traj[t]
        sim.target_pos[RIGHT_ARM_CTRL] = right_traj[t]

        # Record observation BEFORE stepping (obs at time t, action applied at t)
        pos_all, vel_all = sim.get_obs()
        joint_pos_frames.append(
            np.concatenate([pos_all[LEFT_ARM_CTRL], pos_all[RIGHT_ARM_CTRL]]))
        joint_vel_frames.append(
            np.concatenate([vel_all[LEFT_ARM_CTRL], vel_all[RIGHT_ARM_CTRL]]))
        action_frames.append(
            np.concatenate([left_traj[t], right_traj[t]]).astype(np.float32))
        camera_frames.append(sim.render_camera())

        # Physics step
        sim.step_frame()

    # Evaluate success
    box_final_z = sim.box_pos[2]
    lift_cm = (box_final_z - box_init_z) * 100
    contacts = sim.get_palm_box_contacts()
    both_contact = contacts["left_contact"] and contacts["right_contact"]
    min_force = min(contacts["left_force"], contacts["right_force"])
    success = (lift_cm >= MIN_LIFT_CM) and both_contact and (min_force >= MIN_FORCE_N)

    return {
        "obs/joint_positions":  np.array(joint_pos_frames, dtype=np.float32),
        "obs/joint_velocities": np.array(joint_vel_frames, dtype=np.float32),
        "obs/camera_frames":    np.array(camera_frames, dtype=np.uint8),
        "action":               np.array(action_frames, dtype=np.float32),
        "meta": {
            "success": success,
            "lift_cm": lift_cm,
            "left_force": contacts["left_force"],
            "right_force": contacts["right_force"],
            "box_pos_init": box_pos.tolist(),
            "box_pos_final": sim.box_pos.tolist(),
            "n_frames": n_frames,
        },
    }


# ────────────────────────────────────────────────────────
# Save to HDF5
# ────────────────────────────────────────────────────────

def save_episode(ep_data: dict, episode_id: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode_id:04d}.hdf5"
    with h5py.File(path, "w") as f:
        obs_grp = f.create_group("obs")
        obs_grp.create_dataset("joint_positions",
                               data=ep_data["obs/joint_positions"],
                               compression="gzip")
        obs_grp.create_dataset("joint_velocities",
                               data=ep_data["obs/joint_velocities"],
                               compression="gzip")
        obs_grp.create_dataset("camera_frames",
                               data=ep_data["obs/camera_frames"],
                               compression="gzip")
        f.create_dataset("action", data=ep_data["action"], compression="gzip")

        # Metadata
        f.attrs["episode_id"] = episode_id
        f.attrs["task_description"] = TASK_LABEL
        f.attrs["fps"] = CONTROL_HZ
        f.attrs["timestamp"] = datetime.datetime.now().isoformat()
        f.attrs["action_dim"] = 14
        f.attrs["obs_dim"] = 14
        f.attrs["physics_mode"] = "mj_step"

        meta = ep_data["meta"]
        f.attrs["success"] = meta["success"]
        f.attrs["lift_cm"] = meta["lift_cm"]
        f.attrs["left_force_N"] = meta["left_force"]
        f.attrs["right_force_N"] = meta["right_force"]
        f.attrs["num_frames"] = meta["n_frames"]
    return path


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate bimanual box manipulation demos (physics-based)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes to generate")
    parser.add_argument("--output", default="data/bimanual_demos",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--noise-x", type=float, default=0.02,
                        help="Box X position noise range (default: 0.02)")
    parser.add_argument("--noise-y", type=float, default=0.02,
                        help="Box Y position noise range (default: 0.02)")
    parser.add_argument("--random-start", type=float, default=0.0,
                        help="Arm starting posture randomization spread (0=disabled, 0.25=moderate)")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Enable visual domain randomization")
    args = parser.parse_args()

    output_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.episodes} bimanual demos → {output_dir}/")
    print(f"Task: '{TASK_LABEL}'")
    print(f"Physics: mj_step (real contact, friction-based grasping)")
    print()

    sim = PhysicsSim()

    successes = 0
    ik_failures = 0
    physics_failures = 0
    ep_id = args.start_id
    t_start = time.time()
    max_retries = 5  # retry IK failures with new random params

    for i in range(args.episodes):
        t0 = time.time()

        # Retry loop for IK failures
        ep_data = None
        for attempt in range(max_retries):
            ep_data = generate_episode(
                sim, rng, noise_x=args.noise_x, noise_y=args.noise_y,
                random_start=args.random_start,
                domain_rand=args.domain_rand)
            if ep_data is not None:
                break
            ik_failures += 1

        if ep_data is None:
            print(f"  [{i+1}/{args.episodes}] SKIPPED — IK failed {max_retries} times")
            continue

        meta = ep_data["meta"]

        path = save_episode(ep_data, ep_id, output_dir)
        elapsed = time.time() - t0

        status = "OK" if meta["success"] else "FAIL"
        if meta["success"]:
            successes += 1
        else:
            physics_failures += 1

        print(f"  [{i+1}/{args.episodes}] ep_{ep_id:04d} — "
              f"{meta['n_frames']} frames ({meta['n_frames']/CONTROL_HZ:.1f}s) — "
              f"lift={meta['lift_cm']:.1f}cm "
              f"F=[{meta['left_force']:.1f},{meta['right_force']:.1f}]N "
              f"[{status}] — {elapsed:.1f}s → {path.name}")
        ep_id += 1

    total_time = time.time() - t_start
    saved = successes + physics_failures
    rate = successes / saved * 100 if saved > 0 else 0

    print(f"\nDone. {args.episodes} requested, {saved} saved in {total_time:.0f}s")
    print(f"Success: {successes}/{saved} ({rate:.0f}%)")
    if ik_failures > 0:
        print(f"IK failures (retried): {ik_failures}")
    if physics_failures > 0:
        print(f"Physics failures (saved): {physics_failures}")
    print(f"Output: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
