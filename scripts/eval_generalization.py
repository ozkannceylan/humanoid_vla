#!/usr/bin/env python3
"""
scripts/eval_generalization.py

Comprehensive generalization evaluation for Phase F.
Tests ACT models across multiple distributions to measure true generalization
vs in-distribution performance.

Supports both single-arm (kinematic) and bimanual (physics) evaluation.

Test distributions:
  - in_dist:      Same noise range as training, different seeds
  - ood_position:  Wider position range (1.5x training range)
  - ood_visual:    Novel table/light colors (domain randomization ON during eval)
  - ood_posture:   Wider starting postures (1.5x training spread)
  - ood_combined:  All OOD factors combined

Usage:
  # Single-arm evaluation
  MUJOCO_GL=egl python3 scripts/eval_generalization.py \\
      --checkpoint data/checkpoints/best.pt \\
      --episodes 20 --train-noise-range 0.10

  # Bimanual evaluation
  MUJOCO_GL=egl python3 scripts/eval_generalization.py \\
      --checkpoint data/bimanual_checkpoints/best.pt \\
      --mode bimanual --episodes 20 --train-noise-x 0.08 --train-noise-y 0.06
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy, TASK_LABELS, task_to_id
from generate_demos import SimWrapper, RIGHT_ARM_CTRL
from domain_randomization import DomainRandomizer
from evaluate import run_episode, load_model, SUCCESS_FN
from physics_sim import PhysicsSim
from evaluate_bimanual import evaluate_episode as evaluate_bimanual_episode


DISTRIBUTIONS = {
    'in_dist': {
        'description': 'In-distribution (same range as training)',
        'noise_mult': 1.0,
        'start_mult': 1.0,
        'domain_rand': False,
    },
    'ood_position': {
        'description': 'OOD: wider object positions (1.5x range)',
        'noise_mult': 1.5,
        'start_mult': 1.0,
        'domain_rand': False,
    },
    'ood_visual': {
        'description': 'OOD: novel visual appearance',
        'noise_mult': 1.0,
        'start_mult': 1.0,
        'domain_rand': True,
    },
    'ood_posture': {
        'description': 'OOD: wider starting postures (1.5x spread)',
        'noise_mult': 1.0,
        'start_mult': 1.5,
        'domain_rand': False,
    },
    'ood_combined': {
        'description': 'OOD: all factors combined',
        'noise_mult': 1.5,
        'start_mult': 1.5,
        'domain_rand': True,
    },
}


def run_eval_suite(model, sim, task_label, rng, device, episodes,
                   noise_range, random_start, domain_rand, max_steps):
    """Run evaluation suite for one task under specific conditions."""
    successes = 0
    dists = []

    for ep in range(episodes):
        # Apply domain randomization if requested
        if domain_rand:
            sim.randomizer.randomize(rng)

        # Apply random starting posture
        if random_start > 0:
            sim.random_arm_start(rng, spread=random_start)

        ok, length, dist = run_episode(
            model, sim, task_label, rng,
            device=device, max_steps=max_steps,
            noise_range=noise_range,
        )
        successes += int(ok)
        dists.append(dist)

        # Restore visual state after domain rand
        if domain_rand:
            sim.randomizer.restore()

    return {
        'successes': successes,
        'episodes': episodes,
        'rate': successes / episodes * 100,
        'avg_dist': np.mean(dists),
    }


def run_bimanual_eval_suite(model, sim, rng, device, episodes,
                            noise_x, noise_y, random_start, domain_rand,
                            max_steps):
    """Run bimanual evaluation suite under specific conditions."""
    randomizer = DomainRandomizer(sim.model, sim.data)
    successes = 0
    lifts = []

    for ep in range(episodes):
        if domain_rand:
            randomizer.randomize(rng)

        result = evaluate_bimanual_episode(
            model, sim, rng, device=device, max_steps=max_steps,
            noise_x=noise_x, noise_y=noise_y,
            random_start=random_start,
        )
        successes += int(result['success'])
        lifts.append(result['lift_cm'])

        if domain_rand:
            randomizer.restore()

    return {
        'successes': successes,
        'episodes': episodes,
        'rate': successes / episodes * 100,
        'avg_dist': np.mean(lifts),  # reuse field for lift_cm
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generalization evaluation for Phase F models")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["single-arm", "bimanual"],
                        default="single-arm",
                        help="Evaluation mode (default: single-arm)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per task per distribution")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to evaluate (default: all, single-arm only)")
    parser.add_argument("--distributions", nargs="*", default=None,
                        help=f"Distributions to test (choices: {list(DISTRIBUTIONS)})")
    # Single-arm noise
    parser.add_argument("--train-noise-range", type=float, default=0.03,
                        help="Noise range used during training (single-arm)")
    # Bimanual noise (separate x/y)
    parser.add_argument("--train-noise-x", type=float, default=0.02,
                        help="Box x-noise used during training (bimanual)")
    parser.add_argument("--train-noise-y", type=float, default=0.02,
                        help="Box y-noise used during training (bimanual)")
    parser.add_argument("--train-random-start", type=float, default=0.0,
                        help="Random start spread used during training")
    parser.add_argument("--seed", type=int, default=200,
                        help="Eval seed (different from training seed 42)")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)
    is_bimanual = (args.mode == "bimanual")

    # Determine distributions
    dist_keys = args.distributions if args.distributions else list(DISTRIBUTIONS)
    dist_keys = [k for k in dist_keys if k in DISTRIBUTIONS]

    if is_bimanual:
        sim = PhysicsSim()
        task_labels = ["pick up the green box with both hands"]
    else:
        sim = SimWrapper()
        task_labels_from_ckpt = config.get('task_labels', TASK_LABELS)
        if args.tasks:
            task_labels = [t for t in task_labels_from_ckpt
                           if any(key in t for key in args.tasks)]
        else:
            task_labels = task_labels_from_ckpt
        task_labels = [t for t in task_labels if t in SUCCESS_FN]

    print(f"\n{'='*72}")
    print(f"Generalization Evaluation ({args.mode})")
    print(f"{'='*72}")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Mode:           {args.mode}")
    print(f"  Tasks:          {len(task_labels)}")
    print(f"  Distributions:  {len(dist_keys)}")
    print(f"  Episodes/task:  {args.episodes}")
    if is_bimanual:
        print(f"  Train noise:    x={args.train_noise_x}, y={args.train_noise_y}")
    else:
        print(f"  Train noise:    {args.train_noise_range}")
    print(f"  Train start:    {args.train_random_start}")
    print(f"{'='*72}\n")

    # Run evaluations
    all_results = {}
    t_total = time.time()

    for dist_key in dist_keys:
        dist_cfg = DISTRIBUTIONS[dist_key]
        random_start = args.train_random_start * dist_cfg['start_mult']
        domain_rand = dist_cfg['domain_rand']

        print(f"\n--- {dist_key}: {dist_cfg['description']} ---")

        rng = np.random.default_rng(args.seed)
        dist_results = {}

        if is_bimanual:
            noise_x = args.train_noise_x * dist_cfg['noise_mult']
            noise_y = args.train_noise_y * dist_cfg['noise_mult']
            print(f"    noise_x={noise_x:.3f}, noise_y={noise_y:.3f}, "
                  f"random_start={random_start:.2f}, domain_rand={domain_rand}")

            task_label = task_labels[0]
            t0 = time.time()
            result = run_bimanual_eval_suite(
                model, sim, rng, args.device, args.episodes,
                noise_x=noise_x, noise_y=noise_y,
                random_start=random_start,
                domain_rand=domain_rand,
                max_steps=args.max_steps,
            )
            elapsed = time.time() - t0
            dist_results[task_label] = result

            tag = "+" if result['rate'] >= 50 else "-"
            print(f"  {tag} {task_label:40s} "
                  f"{result['successes']:2d}/{result['episodes']:2d} "
                  f"({result['rate']:5.1f}%) "
                  f"avg_lift={result['avg_dist']:.1f}cm "
                  f"[{elapsed:.1f}s]")
        else:
            noise_range = args.train_noise_range * dist_cfg['noise_mult']
            print(f"    noise_range={noise_range:.3f}, "
                  f"random_start={random_start:.2f}, domain_rand={domain_rand}")

            for task_label in task_labels:
                t0 = time.time()
                result = run_eval_suite(
                    model, sim, task_label, rng, args.device, args.episodes,
                    noise_range=noise_range,
                    random_start=random_start,
                    domain_rand=domain_rand,
                    max_steps=args.max_steps,
                )
                elapsed = time.time() - t0
                dist_results[task_label] = result

                tag = "+" if result['rate'] >= 50 else "-"
                print(f"  {tag} {task_label:40s} "
                      f"{result['successes']:2d}/{result['episodes']:2d} "
                      f"({result['rate']:5.1f}%) "
                      f"avg_d={result['avg_dist']:.3f} "
                      f"[{elapsed:.1f}s]")

        all_results[dist_key] = dist_results

    # Summary table
    total_time = time.time() - t_total
    print(f"\n\n{'='*72}")
    print(f"SUMMARY ({args.mode})")
    print(f"{'='*72}")

    # Header
    header = f"{'Task':<42s}"
    for dk in dist_keys:
        short = dk.replace('ood_', '').replace('in_dist', 'in-dist')[:10]
        header += f" {short:>10s}"
    print(header)
    print("-" * len(header))

    # Per-task rows
    for task_label in task_labels:
        short_task = task_label[:40]
        row = f"{short_task:<42s}"
        for dk in dist_keys:
            r = all_results[dk].get(task_label, {})
            rate = r.get('rate', 0)
            row += f" {rate:9.1f}%"
        print(row)

    # Overall row
    print("-" * len(header))
    row = f"{'OVERALL':<42s}"
    for dk in dist_keys:
        total_s = sum(r['successes'] for r in all_results[dk].values())
        total_e = sum(r['episodes'] for r in all_results[dk].values())
        rate = total_s / total_e * 100 if total_e else 0
        row += f" {rate:9.1f}%"
    print(row)

    print(f"\nTotal evaluation time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
