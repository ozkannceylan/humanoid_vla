#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.data.episode_recorder import load_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quick side-by-side visualization for one recorded episode")
    parser.add_argument("episode", help="Path to episode_XXXXXX.npz")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode)
    arrays, meta = load_episode(episode_path)
    static_frames = arrays["observation_images_cam_high"]
    wrist_frames = arrays["observation_images_cam_left_wrist"]
    assert len(static_frames) == len(wrist_frames)

    out_path = Path(args.output) if args.output else episode_path.with_suffix(".preview.mp4")
    height = max(static_frames.shape[1], wrist_frames.shape[1])
    width = static_frames.shape[2] + wrist_frames.shape[2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), int(meta.get("control_freq", 50)), (width, height))
    for static_frame, wrist_frame in zip(static_frames, wrist_frames, strict=True):
        combined = np.concatenate([static_frame, wrist_frame], axis=1)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    writer.release()
    print(json.dumps({"preview": str(out_path), "frames": int(len(static_frames)), "success": bool(meta.get("success", False))}, indent=2))


if __name__ == "__main__":
    main()
