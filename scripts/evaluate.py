#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluate a trained ACT policy in MuJoCo simulation.
Runs N episodes per task, measures success rate with task-specific criteria.

Success criteria:
  reach:  hand within 6cm of cube at end of episode
  grasp:  cube attached to hand (auto-grasp triggered)
  pick:   cube lifted above z=0.90
  place:  cube within 6cm of place target after release

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/evaluate.py --checkpoint data/checkpoints/best.pt
  MUJOCO_GL=egl python3 scripts/evaluate.py --checkpoint data/checkpoints/best.pt --episodes 20
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy, TASK_LABELS, task_to_id
from generate_demos import SimWrapper, RIGHT_ARM_CTRL


# ────────────────────────────────────────────────────────
# Success detectors
# ────────────────────────────────────────────────────────

def check_reach(sim):
    """Hand within 6cm of cube."""
    return np.linalg.norm(sim.hand_pos - sim.cube_pos) < 0.06


def check_grasp(sim, grasped):
    """Cube is attached to hand (auto-grasp was triggered)."""
    return grasped


def check_pick(sim, grasped):
    """Cube is lifted above z=0.90."""
    return grasped and sim.cube_pos[2] > 0.90


def check_place(sim, grasped):
    """Cube is within 6cm of place target, released, and near table height."""
    place = sim.place_pos
    if place is None:
        return False
    cube = sim.cube_pos
    return (not grasped
            and np.linalg.norm(cube[:2] - place[:2]) < 0.06
            and cube[2] < 0.87)


SUCCESS_FN = {
    "reach the red cube": lambda sim, g: check_reach(sim),
    "grasp the red cube": lambda sim, g: check_grasp(sim, g),
    "pick up the red cube": lambda sim, g: check_pick(sim, g),
    "place the red cube on the blue plate": lambda sim, g: check_place(sim, g),
}


# ────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────

def run_episode(model, sim, task_label, rng, device='cuda',
                max_steps=150, auto_grasp_dist=0.04, auto_release_delay=30,
                chunk_exec=5, ensemble_k=0.01):
    """Run one episode with the ACT policy in kinematic MuJoCo sim.

    Uses temporal ensembling (ACT paper, Zhao et al. 2023):
      - Every `chunk_exec` steps, query the model for a new action chunk
      - Overlapping chunks are exponentially-weighted averaged
      - This smooths out per-step prediction noise and reduces compounding error

    For composite tasks (pick, place), uses hierarchical task decomposition:
      Phase 1: Run with grasp task embedding until auto-grasp triggers
      Phase 2: Switch to pick/place task embedding for the remaining motion
    This prevents the model from averaging approach and lift phases.

    Args:
        chunk_exec: steps between re-planning (default 5)
        ensemble_k: temporal decay factor for weighting older chunks (default 0.01)

    Auto-grasp: when hand is within auto_grasp_dist of cube.
    Auto-release: for place task, when cube is near target and
                  grasped for at least auto_release_delay steps.

    Returns: (success, trajectory_length, final_hand_cube_dist)
    """
    sim.reset_with_noise(rng)
    chunk_size = model.chunk_size
    action_dim = model.action_dim

    # Hierarchical task decomposition:
    # Composite tasks (pick, place) first use "grasp" to approach,
    # then switch to their own embedding after grasp triggers.
    is_composite = ("pick" in task_label or "place" in task_label)
    approach_task_id = task_to_id("grasp the red cube") if is_composite else None
    final_task_id = task_to_id(task_label)
    active_task_id = approach_task_id if is_composite else final_task_id

    grasped = False
    grasp_step = -1
    released = False      # prevent re-grasping after release
    is_place = ("place" in task_label)
    place_target = sim.place_pos if is_place else None

    # Temporal ensembling buffers
    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    for step in range(max_steps):
        # Re-plan every chunk_exec steps (or on first step)
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos, vel = sim.get_obs()
            state = np.concatenate([pos, vel])

            chunk = model.predict(image, state, active_task_id, device=device)
            # Add chunk with exponential temporal weighting
            for i in range(chunk_size):
                w = np.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        # Get ensembled action for this step
        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        # Execute kinematically
        sim.target_pos[RIGHT_ARM_CTRL] = action[RIGHT_ARM_CTRL]
        sim.step_frame()

        # Auto-grasp (only if not already released for place task)
        if not grasped and not released:
            d = np.linalg.norm(sim.hand_pos - sim.cube_pos)
            if d < auto_grasp_dist:
                sim.set_weld(True)
                grasped = True
                grasp_step = step
                # Switch to final task embedding after grasp
                if is_composite:
                    active_task_id = final_task_id
                    # Reset ensembling buffers for new phase
                    action_sum[step+1:] = 0
                    weight_sum[step+1:] = 0

        # Auto-release (place task only)
        if is_place and grasped and place_target is not None:
            steps_grasped = step - grasp_step
            cube_near = np.linalg.norm(sim.cube_pos[:2] - place_target[:2]) < 0.06
            hand_low = sim.hand_pos[2] < place_target[2] + 0.12
            if steps_grasped > auto_release_delay and cube_near and hand_low:
                sim.set_weld(False)
                grasped = False
                released = True
                # In kinematic mode there's no gravity — simulate drop to table
                sim.data.qpos[sim.cube_qpos_adr + 2] = 0.825  # table surface
                mujoco.mj_forward(sim.model, sim.data)
                sim.set_weld(False)
                grasped = False

    # Check success
    check = SUCCESS_FN.get(task_label)
    success = check(sim, grasped) if check else False
    final_dist = np.linalg.norm(sim.hand_pos - sim.cube_pos)

    return success, max_steps, final_dist


def load_model(checkpoint_path, device='cuda'):
    """Load ACT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = ACTPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_tasks=config['num_tasks'],
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")
    print(f"  Tasks: {config.get('task_labels', 'N/A')}")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy in simulation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per task")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to evaluate (default: all known)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    model, config = load_model(args.checkpoint, device)
    sim = SimWrapper()
    rng = np.random.default_rng(args.seed)

    # Determine which tasks to evaluate
    task_labels_from_ckpt = config.get('task_labels', TASK_LABELS)
    if args.tasks:
        task_labels = [t for t in task_labels_from_ckpt
                       if any(key in t for key in args.tasks)]
    else:
        task_labels = task_labels_from_ckpt

    # Also filter by what's available in SUCCESS_FN
    task_labels = [t for t in task_labels if t in SUCCESS_FN]

    print(f"\n{'='*60}")
    print(f"Evaluation: {args.episodes} episodes × {len(task_labels)} tasks")
    print(f"{'='*60}")

    results = {}
    total_success = 0
    total_episodes = 0

    for task_label in task_labels:
        successes = 0
        dists = []

        t0 = time.time()
        for ep in range(args.episodes):
            ok, length, dist = run_episode(
                model, sim, task_label, rng,
                device=device, max_steps=args.max_steps,
            )
            successes += int(ok)
            dists.append(dist)

        elapsed = time.time() - t0
        rate = successes / args.episodes * 100
        avg_dist = np.mean(dists)
        results[task_label] = {
            'success_rate': rate,
            'successes': successes,
            'avg_dist': avg_dist,
        }
        total_success += successes
        total_episodes += args.episodes

        tag = "✓" if rate >= 50 else "✗"
        print(f"  {tag} {task_label:40s} — {successes}/{args.episodes} "
              f"({rate:5.1f}%) — avg_d={avg_dist:.3f} — {elapsed:.1f}s")

    overall_rate = total_success / total_episodes * 100 if total_episodes else 0
    print(f"\n{'─'*60}")
    print(f"  Overall: {total_success}/{total_episodes} ({overall_rate:.1f}%)")
    print(f"{'─'*60}")

    # Phase B success criterion: >50% on reach
    reach_rate = results.get("reach the red cube", {}).get('success_rate', 0)
    if reach_rate >= 50:
        print(f"\n  ✅ Phase B criterion met: reach success = {reach_rate:.0f}% ≥ 50%")
    else:
        print(f"\n  ❌ Phase B criterion NOT met: reach success = {reach_rate:.0f}% < 50%")


if __name__ == "__main__":
    main()
