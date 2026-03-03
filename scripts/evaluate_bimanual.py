#!/usr/bin/env python3
"""
scripts/evaluate_bimanual.py

Evaluate the bimanual ACT model on physics-based box manipulation.
Runs N episodes with real mj_step physics, No weld constraints.

Success criteria:
  - Box lifted >= 3cm above initial position
  - Both palms in contact with box at end
  - Minimum contact force >= 2N per palm

Uses temporal ensembling (overlapping action chunks with exponential weighting)
for smooth action execution — same technique as the single-arm evaluator.

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/evaluate_bimanual.py \\
      --checkpoint data/bimanual_checkpoints/best.pt --episodes 20
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy
from physics_sim import (
    PhysicsSim, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, BOTH_ARMS_CTRL,
    CONTROL_HZ,
)

# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────

MAX_STEPS = 250          # max frames per episode (~8.3s at 30Hz)
CHUNK_EXEC = 5           # re-plan every N steps
ENSEMBLE_K = 0.01        # temporal ensembling decay

MIN_LIFT_CM = 3.0        # success threshold: box lifted >= 3cm
MIN_FORCE_N = 2.0        # minimum contact force on each palm


# ────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────

def evaluate_episode(model, sim, rng, device='cuda',
                     max_steps=MAX_STEPS, chunk_exec=CHUNK_EXEC,
                     ensemble_k=ENSEMBLE_K, verbose=False):
    """Run one bimanual episode. Returns dict with metrics."""
    sim.reset_with_noise(rng)
    box_init_z = sim.box_pos[2]

    chunk_size = model.chunk_size
    action_dim = model.action_dim

    # Temporal ensembling buffers
    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    for step in range(max_steps):
        # Re-plan every chunk_exec steps
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos_all, vel_all = sim.get_obs()
            # State: 14 pos + 14 vel (both arms only)
            state = np.concatenate([
                pos_all[LEFT_ARM_CTRL], pos_all[RIGHT_ARM_CTRL],
                vel_all[LEFT_ARM_CTRL], vel_all[RIGHT_ARM_CTRL],
            ])

            chunk = model.predict(image, state, task_id=0, device=device)
            # chunk shape: (chunk_size, 14)

            for i in range(chunk_size):
                w = math.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        # Get ensembled action
        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        # Apply: first 7 = left arm, last 7 = right arm
        sim.target_pos[LEFT_ARM_CTRL] = action[:7]
        sim.target_pos[RIGHT_ARM_CTRL] = action[7:]

        sim.step_frame()

        if verbose and step % 30 == 0:
            c = sim.get_palm_box_contacts()
            lift = (sim.box_pos[2] - box_init_z) * 100
            print(f"    step {step:3d}: box_z={sim.box_pos[2]:.3f} "
                  f"lift={lift:+.1f}cm "
                  f"F=[{c['left_force']:.0f},{c['right_force']:.0f}]N")

    # Final evaluation
    box_final_z = sim.box_pos[2]
    lift_cm = (box_final_z - box_init_z) * 100
    contacts = sim.get_palm_box_contacts()
    both_contact = contacts["left_contact"] and contacts["right_contact"]
    min_force = min(contacts["left_force"], contacts["right_force"])
    success = (lift_cm >= MIN_LIFT_CM) and both_contact and (min_force >= MIN_FORCE_N)

    return {
        "success": success,
        "lift_cm": lift_cm,
        "left_force": contacts["left_force"],
        "right_force": contacts["right_force"],
        "both_contact": both_contact,
        "box_final_z": box_final_z,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bimanual ACT model on physics-based box manipulation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to bimanual model checkpoint")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed (different from training seed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step progress")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['config']

    model = ACTPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_tasks=config['num_tasks'],
    ).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded bimanual ACT model [epoch {ckpt['epoch']}, "
          f"loss {ckpt['loss']:.6f}]")
    print(f"  state_dim={config['state_dim']}, action_dim={config['action_dim']}, "
          f"chunk_size={config['chunk_size']}")

    # Create sim
    sim = PhysicsSim()
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print(f"Bimanual Evaluation — {args.episodes} episodes")
    print(f"{'='*60}")
    print(f"  Task: pick up the green box with both hands")
    print(f"  Physics: mj_step (real contact, friction-based)")
    print(f"  Success: lift >= {MIN_LIFT_CM}cm + both palms in contact + force >= {MIN_FORCE_N}N")
    print(f"  Max steps: {MAX_STEPS} ({MAX_STEPS/CONTROL_HZ:.1f}s)")
    print()

    successes = 0
    lifts = []
    forces_l = []
    forces_r = []

    t_start = time.time()

    for i in range(args.episodes):
        t0 = time.time()
        result = evaluate_episode(
            model, sim, rng, device=args.device,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0

        status = "OK" if result["success"] else "FAIL"
        if result["success"]:
            successes += 1
        lifts.append(result["lift_cm"])
        forces_l.append(result["left_force"])
        forces_r.append(result["right_force"])

        print(f"  [{i+1:2d}/{args.episodes}] "
              f"lift={result['lift_cm']:+5.1f}cm "
              f"F=[{result['left_force']:.0f},{result['right_force']:.0f}]N "
              f"contact={'BOTH' if result['both_contact'] else 'PARTIAL'} "
              f"[{status}] — {elapsed:.1f}s")

    total_time = time.time() - t_start
    rate = successes / args.episodes * 100

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Success rate: {successes}/{args.episodes} ({rate:.0f}%)")
    print(f"  Lift:  mean={np.mean(lifts):.1f}cm, "
          f"min={np.min(lifts):.1f}cm, max={np.max(lifts):.1f}cm")
    print(f"  Force: L_mean={np.mean(forces_l):.1f}N, "
          f"R_mean={np.mean(forces_r):.1f}N")
    print(f"  Time:  {total_time:.0f}s total, "
          f"{total_time/args.episodes:.1f}s/episode")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
