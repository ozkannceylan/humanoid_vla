#!/usr/bin/env python3
"""
scripts/train_act.py

Train an ACT (Action Chunking with Transformers) policy on collected demos
using HuggingFace LeRobot's training script.

ACT paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
           Zhao et al., 2023 — https://arxiv.org/abs/2304.13705

Usage:
  # Make sure lerobot is installed first:
  pip3 install --break-system-packages lerobot

  # Convert your demos first:
  python3 scripts/convert_to_lerobot.py

  # Then train:
  python3 scripts/train_act.py --dataset data/lerobot_dataset

Training on RTX 4050 (6GB VRAM):
  - ACT ~50M params fits in 6GB with batch_size=8
  - Typical training: 50k-100k steps, ~2-4 hours
  - Eval every 5000 steps on 10 rollouts
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path


# ----- ACT hyperparameters tuned for RTX 4050 / G1 arm manipulation ------
# Adjust these if training diverges or OOM occurs.

ACT_OVERRIDES = [
    "policy=act",
    "policy.chunk_size=100",            # action chunking window (ACT default)
    "policy.n_action_steps=100",
    "policy.input_shapes.observation.state=[58]",  # 29 pos + 29 vel
    "policy.output_shapes.action=[29]",
    "policy.vision_backbone=resnet18",
    "policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
    "policy.use_separate_rgb_encoder_per_camera=false",
    "training.batch_size=8",            # safe for 6GB VRAM
    "training.lr=1e-5",
    "training.num_epochs=5000",
    "training.eval_freq=500",
    "training.save_freq=1000",
    "training.log_freq=50",
    "eval.n_episodes=10",
    "device=cuda",
]


def check_dataset(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path}\n"
            "Run: python3 scripts/convert_to_lerobot.py"
        )
    with open(info_path) as f:
        return json.load(f)


def check_lerobot():
    try:
        import lerobot  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/lerobot_dataset",
        help="Path to LeRobot-format dataset (output of convert_to_lerobot.py)",
    )
    parser.add_argument(
        "--output",
        default="data/act_training",
        help="Directory for training outputs (checkpoints, logs, evals)",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total gradient steps (default: 100k, ~2-4h on RTX 4050)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (reduce to 4 if OOM)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    info = check_dataset(dataset_path)

    print(f"\n=== ACT Training ===")
    print(f"Dataset : {dataset_path.absolute()}")
    print(f"  Episodes : {info['total_episodes']}")
    print(f"  Frames   : {info['total_frames']}")
    print(f"Output  : {Path(args.output).absolute()}")
    print(f"Steps   : {args.steps}")
    print(f"Batch   : {args.batch_size}")

    if not check_lerobot():
        print(
            "\n[ERROR] LeRobot is not installed.\n"
            "Install it with:\n"
            "  pip3 install --break-system-packages lerobot\n"
            "\nThen re-run this script."
        )
        sys.exit(1)

    # Build lerobot training command
    run_dir = str(Path(args.output).absolute())
    overrides = [
        f"dataset_repo_id={dataset_path.absolute()}",
        f"hydra.run.dir={run_dir}",
        f"training.num_epochs={args.steps}",   # lerobot uses epochs but maps to steps
        f"training.batch_size={args.batch_size}",
    ] + ACT_OVERRIDES

    if args.resume:
        overrides.append(f"resume=true")
        overrides.append(f"resume_path={args.resume}")

    cmd = [sys.executable, "-m", "lerobot.scripts.train"] + overrides

    print(f"\nLaunching:\n  {' '.join(cmd[:5])} ...")
    print("  (Training logs will appear below. Ctrl+C to interrupt.)\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n[ERROR] Training exited with code {result.returncode}")
            print(
                "Common fixes:\n"
                "  OOM     → reduce --batch_size to 4 or 2\n"
                "  Dataset → check data/lerobot_dataset/meta/info.json exists\n"
                "  Videos  → ensure ffmpeg is installed: sudo apt install -y ffmpeg"
            )
        else:
            print(f"\nTraining complete. Checkpoints in: {run_dir}")
            print(
                "\nNext step: evaluate the policy\n"
                "  ros2 run vla_mujoco_bridge vla_node --ros-args \\\n"
                f"    -p checkpoint:={run_dir}/checkpoints/last/pretrained_model"
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")


if __name__ == "__main__":
    main()
