#!/usr/bin/env python3
"""v2/scripts/generate_episodes.py — Generate bimanual box manipulation episodes.

Uses full mj_step physics with PD control.  Both arms squeeze the green box
from the sides using friction-only grasping.

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 v2/scripts/generate_episodes.py --num-episodes 10
  python3 v2/scripts/generate_episodes.py --num-episodes 5 --viewer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.common import LEFT_ARM_CTRL, RIGHT_ARM_CTRL, BOX_NOMINAL_POS
from v2.data.episode_recorder import EpisodeRecorder
from v2.env.table_env_v2 import TableEnvV2
from v2.ik.trajectory_generator import plan_bimanual_trajectory
from v2.randomization.domain_randomizer import DomainRandomizer

import mujoco


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def existing_episode_ids(raw_dir: Path) -> set[int]:
    ids: set[int] = set()
    for p in raw_dir.glob("episode_*.npz"):
        ids.add(int(p.stem.split("_")[-1]))
    for p in raw_dir.glob("episode_*.failed.json"):
        ids.add(int(p.stem.split("_")[-1].split(".")[0]))
    return ids


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(i) for i in value]
    return value


def generate_one_episode(
    env: TableEnvV2,
    randomizer: DomainRandomizer,
    recorder: EpisodeRecorder,
    cfg: dict,
    raw_dir: Path,
    episode_id: int,
    seed: int,
    step_callback=None,
) -> dict[str, Any]:
    """Generate one bimanual episode, retrying IK up to max_attempts times."""
    max_attempts = cfg.get("trajectory", {}).get("max_planning_attempts", 10)

    last_failure: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)

        # Reset + randomise
        env.reset()
        rand = randomizer.apply(rng)
        box_pos = env.box_pos.copy()

        # Plan trajectory
        plan = plan_bimanual_trajectory(
            env, box_pos, rng,
            config=cfg.get("trajectory"),
        )

        if not plan.success or plan.n_frames == 0:
            last_failure = {
                "episode_id": episode_id,
                "success": False,
                "attempt": attempt + 1,
                "failure_reason": plan.failure_reason,
            }
            continue

        # Reset again to replay under physics
        # (IK solving changed qpos; we need a clean start with the same box noise)
        box_dx = box_pos[0] - BOX_NOMINAL_POS[0]
        box_dy = box_pos[1] - BOX_NOMINAL_POS[1]
        env.reset()
        # Re-apply domain randomization colours etc (cheap)
        randomizer.restore()
        # Re-apply the same box displacement
        env.data.qpos[env.box_qpos_adr + 0] += box_dx
        env.data.qpos[env.box_qpos_adr + 1] += box_dy
        mujoco.mj_forward(env.model, env.data)
        env.data.qpos[:7] = env._base_qpos
        env.data.qvel[:6] = 0.0

        record = recorder.execute_plan(
            plan=plan,
            episode_id=episode_id,
            output_dir=raw_dir,
            randomization_meta=jsonable(asdict(rand)),
            step_callback=step_callback,
        )

        return {
            "episode_id": record.episode_id,
            "success": record.success,
            "num_steps": record.num_steps,
            "path": str(record.path),
            "meta": record.meta,
            "attempts": attempt + 1,
        }

    # All attempts failed
    failure_path = raw_dir / f"episode_{episode_id:06d}.failed.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(
        jsonable(last_failure or {"episode_id": episode_id, "success": False}),
        indent=2))
    return {
        "episode_id": episode_id,
        "success": False,
        "num_steps": 0,
        "path": str(failure_path),
        "attempts": max_attempts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate bimanual box manipulation episodes (physics-based)")
    p.add_argument("--config", default="v2/config/data_gen_config.yaml")
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--output-dir", default="v2/output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--viewer", action="store_true", default=False,
                   help="Open the MuJoCo interactive viewer (real-time)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    out_root = Path(args.output_dir)
    raw_dir = out_root / cfg.get("data", {}).get("raw_episode_dir", "raw_episodes")
    raw_dir.mkdir(parents=True, exist_ok=True)

    done_ids = existing_episode_ids(raw_dir) if args.resume else set()
    target_ids = [eid for eid in range(args.num_episodes) if eid not in done_ids]

    if not target_ids:
        print("All episodes already generated.  Use --no-resume to overwrite.")
        return

    # Create shared env + helpers
    env = TableEnvV2(cfg)
    randomizer = DomainRandomizer(env, cfg)
    recorder = EpisodeRecorder(env, control_freq=env.control_hz)

    results: list[dict[str, Any]] = []
    t_start = time.time()

    print(f"Generating {len(target_ids)} bimanual episodes -> {raw_dir}/")
    print(f"Physics: mj_step, PD control, friction-only grasping")
    print(f"Control Hz: {env.control_hz}, Substeps: {env.substeps}")
    print()

    if args.viewer:
        import mujoco.viewer as mj_viewer
        print("[viewer] Launching MuJoCo passive viewer")

        with mj_viewer.launch_passive(env.model, env.data) as viewer:
            def _sync():
                viewer.sync()

            iterator = tqdm(target_ids, desc="episodes (viewer)")
            for eid in iterator:
                if not viewer.is_running():
                    print("[viewer] Window closed -- stopping early.")
                    break
                result = generate_one_episode(
                    env, randomizer, recorder, cfg, raw_dir,
                    eid, args.seed + eid, step_callback=_sync)
                results.append(result)
                status = "OK" if result["success"] else "FAIL"
                meta = result.get("meta", {})
                iterator.set_postfix(
                    status=status,
                    lift=f"{meta.get('lift_cm', 0):.1f}cm",
                    steps=result["num_steps"])
    else:
        iterator = tqdm(target_ids, desc="episodes")
        for eid in iterator:
            result = generate_one_episode(
                env, randomizer, recorder, cfg, raw_dir,
                eid, args.seed + eid)
            results.append(result)
            status = "OK" if result["success"] else "FAIL"
            meta = result.get("meta", {})
            iterator.set_postfix(
                status=status,
                lift=f"{meta.get('lift_cm', 0):.1f}cm",
                steps=result["num_steps"])

    env.close()

    # ---- Summary ----
    elapsed = time.time() - t_start
    successes = sum(1 for r in results if r.get("success"))
    total = len(results)
    avg_steps = sum(r.get("num_steps", 0) for r in results) / max(total, 1)
    size_mb = sum(p.stat().st_size for p in raw_dir.glob("episode_*.npz")) / (1024 * 1024)

    print(f"\nSummary ({elapsed:.0f}s)")
    print(f"  episodes: {total}")
    print(f"  successes: {successes}/{total} ({100*successes/max(total,1):.0f}%)")
    print(f"  avg trajectory length: {avg_steps:.0f} steps")
    print(f"  raw data size: {size_mb:.1f} MB")
    print(f"  output: {raw_dir.absolute()}")


if __name__ == "__main__":
    main()
