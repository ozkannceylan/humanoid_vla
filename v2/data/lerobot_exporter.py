from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from v2.data.episode_recorder import load_episode

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset as convert_v21_to_v30
except Exception:  # pragma: no cover - optional during static analysis
    LeRobotDataset = None
    convert_v21_to_v30 = None


@dataclass
class ExportSummary:
    output_dir: Path
    episodes_exported: int
    frames_exported: int
    failed_episodes_skipped: int
    validated: bool


class LeRobotV2Exporter:
    def __init__(self, fps: int, robot_type: str = "unitree_g1"):
        self.fps = int(fps)
        self.robot_type = robot_type
        self.camera_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist"]
        self.state_key = "observation.state"
        self.action_key = "action"

    def export(self, raw_episode_dir: Path, output_dir: Path, filter_failures: bool = True, codec: str = "mp4v") -> ExportSummary:
        raw_episode_dir = Path(raw_episode_dir)
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        episode_npz = sorted(raw_episode_dir.glob("episode_*.npz"))
        task_to_index: dict[str, int] = {}
        episodes_meta: list[dict[str, Any]] = []
        stats_lines: list[dict[str, Any]] = []
        exported = 0
        frames = 0
        skipped = 0

        data_chunk_dir = output_dir / "data" / "chunk-000"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)

        for export_index, npz_path in enumerate(episode_npz):
            arrays, meta = load_episode(npz_path)
            if filter_failures and not meta.get("success", False):
                skipped += 1
                continue
            task_name = f"pick_and_place/{meta.get('grasp_strategy', 'unknown')}"
            task_index = task_to_index.setdefault(task_name, len(task_to_index))
            self._write_episode_videos(output_dir, exported, arrays, codec)
            parquet_path = data_chunk_dir / f"episode_{exported:06d}.parquet"
            self._write_episode_parquet(parquet_path, exported, task_index, arrays)
            steps = int(arrays["action"].shape[0])
            episodes_meta.append({"episode_index": exported, "tasks": [task_name], "length": steps})
            stats_lines.append({"episode_index": exported, "stats": self._compute_episode_stats(arrays)})
            exported += 1
            frames += steps

        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "episodes.jsonl").write_text("\n".join(json.dumps(line) for line in episodes_meta) + ("\n" if episodes_meta else ""))
        tasks_lines = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_index.items(), key=lambda item: item[1])]
        (meta_dir / "tasks.jsonl").write_text("\n".join(json.dumps(line) for line in tasks_lines) + ("\n" if tasks_lines else ""))
        (meta_dir / "episodes_stats.jsonl").write_text("\n".join(json.dumps(line) for line in stats_lines) + ("\n" if stats_lines else ""))
        info = self._build_info(exported, frames, len(task_to_index))
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

        validated = self.validate_dataset(output_dir)
        return ExportSummary(
            output_dir=output_dir,
            episodes_exported=exported,
            frames_exported=frames,
            failed_episodes_skipped=skipped,
            validated=validated,
        )

    def _build_info(self, total_episodes: int, total_frames: int, total_tasks: int) -> dict[str, Any]:
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_chunks": 1,
            "total_videos": total_episodes * len(self.camera_keys),
            "fps": self.fps,
            "splits": {},
            "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                self.state_key: {
                    "dtype": "float32",
                    "shape": [29],
                    "names": [f"joint_{i}" for i in range(29)],
                },
                self.action_key: {
                    "dtype": "float32",
                    "shape": [7],
                    "names": [f"right_arm_joint_{i}" for i in range(7)],
                },
                self.camera_keys[0]: {
                    "dtype": "video",
                    "shape": [3, 224, 224],
                    "names": ["channels", "height", "width"],
                    "info": {"video.fps": self.fps, "video.codec": "mp4v"},
                },
                self.camera_keys[1]: {
                    "dtype": "video",
                    "shape": [3, 224, 224],
                    "names": ["channels", "height", "width"],
                    "info": {"video.fps": self.fps, "video.codec": "mp4v"},
                },
            },
        }

    def _write_episode_videos(self, output_dir: Path, episode_index: int, arrays: dict[str, np.ndarray], codec: str) -> None:
        mapping = {
            self.camera_keys[0]: arrays["observation_images_cam_high"],
            self.camera_keys[1]: arrays["observation_images_cam_left_wrist"],
        }
        for video_key, frames in mapping.items():
            cam_dir = output_dir / "videos" / "chunk-000" / video_key
            cam_dir.mkdir(parents=True, exist_ok=True)
            path = cam_dir / f"episode_{episode_index:06d}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, (frames.shape[2], frames.shape[1]))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

    def _write_episode_parquet(self, path: Path, episode_index: int, task_index: int, arrays: dict[str, np.ndarray]) -> None:
        state = arrays["observation_state"].astype(np.float32)
        action = arrays["action"].astype(np.float32)
        num_steps = state.shape[0]
        table = pa.table({
            "index": pa.array(np.arange(num_steps, dtype=np.int64) + episode_index * 1_000_000),
            "episode_index": pa.array(np.full(num_steps, episode_index, dtype=np.int64)),
            "frame_index": pa.array(np.arange(num_steps, dtype=np.int64)),
            "timestamp": pa.array(np.arange(num_steps, dtype=np.float32) / self.fps),
            "task_index": pa.array(np.full(num_steps, task_index, dtype=np.int64)),
            "next.done": pa.array([False] * (num_steps - 1) + [True]),
            self.state_key: self._fixed_size_array(state),
            self.action_key: self._fixed_size_array(action),
        })
        pq.write_table(table, path)

    def _fixed_size_array(self, array: np.ndarray) -> pa.Array:
        array = np.asarray(array)
        flat = pa.array(array.reshape(-1).tolist(), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, array.shape[1])

    def _compute_episode_stats(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        state = arrays["observation_state"].astype(np.float32)
        action = arrays["action"].astype(np.float32)
        return {
            self.state_key: self._stats_dict(state),
            self.action_key: self._stats_dict(action),
        }

    def _stats_dict(self, values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": values.mean(axis=0).astype(np.float32).tolist(),
            "std": values.std(axis=0).astype(np.float32).tolist(),
            "min": values.min(axis=0).astype(np.float32).tolist(),
            "max": values.max(axis=0).astype(np.float32).tolist(),
            "count": [int(values.shape[0])],
        }

    def validate_dataset(self, output_dir: Path) -> bool:
        if LeRobotDataset is None or convert_v21_to_v30 is None:
            return False
        with tempfile.TemporaryDirectory(prefix="lerobot_v2_validate_") as tmp_dir:
            repo_id = "local/v2_validation"
            working_parent = Path(tmp_dir)
            working_root = working_parent / repo_id
            working_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output_dir, working_root)
            convert_v21_to_v30(
                repo_id=repo_id,
                root=working_parent,
                push_to_hub=False,
                force_conversion=True,
            )
            ds = LeRobotDataset(repo_id, root=working_root, video_backend="pyav")
            _ = ds[0] if len(ds) > 0 else None
        return True
