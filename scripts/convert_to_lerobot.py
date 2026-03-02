#!/usr/bin/env python3
"""
scripts/convert_to_lerobot.py

Converts raw HDF5 demo episodes (recorded by demo_recorder) to a LeRobot
dataset compatible with ACT training.

Input:  data/demos/episode_NNNN.hdf5
Output: data/lerobot_dataset/  (LeRobot v2 format)
        data/lerobot_dataset/meta/info.json
        data/lerobot_dataset/data/chunk-000/episode_NNNN.parquet
        data/lerobot_dataset/videos/chunk-000/observation.images.ego_camera/episode_NNNN.mp4

Usage:
  python3 scripts/convert_to_lerobot.py
  python3 scripts/convert_to_lerobot.py --demos data/demos --out data/lerobot_dataset

LeRobot format reference:
  https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("h5py required: pip3 install --break-system-packages h5py")

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas required: pip3 install --break-system-packages pandas")

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python required: pip3 install --break-system-packages opencv-python")


def convert_episode(ep_path: Path, out_dir: Path, ep_idx: int, fps: int) -> dict:
    """Convert one HDF5 episode to parquet rows + mp4 video."""
    with h5py.File(ep_path, "r") as f:
        pos = f["obs/joint_positions"][:]       # (T, 29)
        vel = f["obs/joint_velocities"][:]       # (T, 29)
        act = f["action"][:]                     # (T, 29)
        imgs = f["obs/camera_frames"][:]         # (T, H, W, 3) uint8 RGB
        task = str(f.attrs.get("task_description", "reach red cube"))
        n_frames = len(pos)

    T, H, W, C = imgs.shape

    # --- Write video ---
    video_dir = out_dir / "videos" / "chunk-000" / "observation.images.ego_camera"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    for rgb_frame in imgs:
        writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # --- Build parquet rows ---
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in range(T):
        row = {
            "index": ep_idx * 10000 + t,          # global frame index (unique)
            "episode_index": ep_idx,
            "frame_index": t,
            "timestamp": t / fps,
            "task": task,
            "next.done": bool(t == T - 1),
        }
        # Observation state (joint positions + velocities concatenated)
        state = np.concatenate([pos[t], vel[t]]).tolist()
        for i, v in enumerate(state):
            row[f"observation.state.{i}"] = float(v)
        # Action (target joint positions)
        for i, v in enumerate(act[t]):
            row[f"action.{i}"] = float(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    return {
        "episode_index": ep_idx,
        "tasks": [task],
        "length": T,
    }


def build_info(episodes_meta: list[dict], fps: int, n_joints: int) -> dict:
    total_frames = sum(e["length"] for e in episodes_meta)
    return {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "fps": fps,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [n_joints * 2],   # pos + vel
                "names": [f"j{i}" for i in range(n_joints * 2)],
            },
            "action": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": [f"j{i}" for i in range(n_joints)],
            },
            "observation.images.ego_camera": {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"],
                "info": {"video.fps": fps, "video.codec": "mp4v"},
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="data/demos", help="Input HDF5 demo directory")
    parser.add_argument("--out", default="data/lerobot_dataset", help="Output LeRobot dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    args = parser.parse_args()

    demos_dir = Path(args.demos)
    out_dir = Path(args.out)

    ep_files = sorted(demos_dir.glob("episode_*.hdf5"))
    if not ep_files:
        raise SystemExit(f"No episode HDF5 files found in {demos_dir}")

    print(f"Converting {len(ep_files)} episodes → {out_dir}")

    if out_dir.exists():
        print(f"Removing existing output dir: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    episodes_meta = []
    for ep_idx, ep_path in enumerate(ep_files):
        print(f"  [{ep_idx+1}/{len(ep_files)}] {ep_path.name} ...", end=" ", flush=True)
        meta = convert_episode(ep_path, out_dir, ep_idx, args.fps)
        episodes_meta.append(meta)
        print(f"{meta['length']} frames OK")

    # Write meta/info.json
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = build_info(episodes_meta, args.fps, n_joints=29)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Write meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    print(f"\nDone. Dataset at: {out_dir.absolute()}")
    print(f"  {info['total_episodes']} episodes, {info['total_frames']} total frames")
    print(f"\nNext step:\n  python3 scripts/train_act.py --dataset {out_dir}")


if __name__ == "__main__":
    main()
