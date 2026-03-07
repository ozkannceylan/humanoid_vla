"""v2/data/episode_recorder.py — Record bimanual physics episodes.

Records per-step:
  - ego camera frame (H, W, 3)
  - 14D arm joint positions  (7 left + 7 right)
  - 14D arm joint velocities
  - 14D actions (joint targets)
  - palm contact forces (left_force, right_force)
  - box position (3D)

Output: compressed .npz + .json metadata per episode.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from v2.common import LEFT_ARM_CTRL, RIGHT_ARM_CTRL, MIN_LIFT_CM, MIN_FORCE_N
from v2.env.table_env_v2 import TableEnvV2
from v2.ik.trajectory_generator import BimanualPlan


TASK_LABEL = "pick up the green box with both hands"


@dataclass
class EpisodeRecord:
    episode_id: int
    success: bool
    path: Path
    num_steps: int
    meta: dict[str, Any]


class EpisodeRecorder:
    """Execute a bimanual plan with full physics and record observations."""

    def __init__(self, env: TableEnvV2, control_freq: int):
        self.env = env
        self.control_freq = int(control_freq)

    def execute_plan(
        self,
        plan: BimanualPlan,
        episode_id: int,
        output_dir: Path,
        randomization_meta: dict[str, Any],
        step_callback: Callable[[], None] | None = None,
    ) -> EpisodeRecord:
        """Play back the planned trajectory with mj_step physics.

        For each frame:
          1. Set PD targets for both arms
          2. Record observation BEFORE stepping
          3. Run physics step_frame()
          4. Call step_callback (e.g. viewer.sync)
        """
        env = self.env
        n_frames = plan.n_frames

        # Recording buffers
        camera_frames: list[np.ndarray] = []
        joint_pos_frames: list[np.ndarray] = []   # (T, 14)
        joint_vel_frames: list[np.ndarray] = []   # (T, 14)
        action_frames: list[np.ndarray] = []       # (T, 14)
        left_force_frames: list[float] = []
        right_force_frames: list[float] = []
        box_pos_frames: list[np.ndarray] = []

        box_init_z = env.box_pos[2]

        for t in range(n_frames):
            # Set PD targets
            env.target_pos[LEFT_ARM_CTRL] = plan.left_traj[t]
            env.target_pos[RIGHT_ARM_CTRL] = plan.right_traj[t]

            # Record observation BEFORE stepping
            arm_pos, arm_vel = env.get_arm_obs()
            joint_pos_frames.append(arm_pos)
            joint_vel_frames.append(arm_vel)
            action_frames.append(
                np.concatenate([plan.left_traj[t], plan.right_traj[t]]).astype(np.float32))
            camera_frames.append(env.render_camera())
            contacts = env.get_palm_box_contacts()
            left_force_frames.append(contacts["left_force"])
            right_force_frames.append(contacts["right_force"])
            box_pos_frames.append(env.box_pos)

            # Physics step
            env.step_frame()

            if step_callback is not None:
                step_callback()

        # ---- Evaluate success ----
        box_final_z = env.box_pos[2]
        lift_cm = (box_final_z - box_init_z) * 100
        final_contacts = env.get_palm_box_contacts()
        both_contact = final_contacts["left_contact"] and final_contacts["right_contact"]
        min_force = min(final_contacts["left_force"], final_contacts["right_force"])
        success = bool(
            lift_cm >= MIN_LIFT_CM
            and both_contact
            and min_force >= MIN_FORCE_N
        )

        # ---- Build metadata ----
        meta = _jsonable({
            "episode_id": int(episode_id),
            "task": TASK_LABEL,
            "success": success,
            "control_freq": self.control_freq,
            "num_steps": n_frames,
            "lift_cm": float(lift_cm),
            "final_left_force": float(final_contacts["left_force"]),
            "final_right_force": float(final_contacts["right_force"]),
            "box_pos_init": box_pos_frames[0].tolist() if box_pos_frames else [],
            "box_pos_final": env.box_pos.tolist(),
            "grasp_params": asdict(plan.params),
            **randomization_meta,
        })

        # ---- Save ----
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_path = output_dir / f"episode_{episode_id:06d}.npz"
        np.savez_compressed(
            episode_path,
            observation_images_ego=np.stack(camera_frames).astype(np.uint8),
            observation_joint_positions=np.array(joint_pos_frames, dtype=np.float32),
            observation_joint_velocities=np.array(joint_vel_frames, dtype=np.float32),
            action=np.array(action_frames, dtype=np.float32),
            left_force=np.array(left_force_frames, dtype=np.float32),
            right_force=np.array(right_force_frames, dtype=np.float32),
            box_position=np.array(box_pos_frames, dtype=np.float32),
        )
        meta_path = output_dir / f"episode_{episode_id:06d}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return EpisodeRecord(
            episode_id=episode_id,
            success=success,
            path=episode_path,
            num_steps=n_frames,
            meta=meta,
        )


def load_episode(record_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    record_path = Path(record_path)
    data = np.load(record_path)
    meta_path = record_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    return {key: data[key] for key in data.files}, meta


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
