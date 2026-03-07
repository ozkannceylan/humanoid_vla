from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from v2.data.episode_recorder import EpisodeRecorder
from v2.data.lerobot_exporter import LeRobotV2Exporter
from v2.env.table_env_v2 import TableEnvV2
from v2.ik.trajectory_generator import IKTrajectoryGenerator
from v2.randomization.domain_randomizer import DomainRandomizer


def load_config() -> dict:
    return yaml.safe_load(Path("/home/ozkan/projects/humanoid_vla/v2/config/data_gen_config.yaml").read_text())


class TestExport(unittest.TestCase):
    def test_export_pipeline(self) -> None:
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            env = TableEnvV2(cfg)
            try:
                rng = np.random.default_rng(7)
                randomizer = DomainRandomizer(env, cfg)
                planner = IKTrajectoryGenerator(env, cfg)
                recorder = EpisodeRecorder(env, control_freq=cfg["env"]["control_freq"])
                env.reset()
                rand = randomizer.apply(rng)
                plan = planner.generate_pick_and_place(rng, env.object_pos.copy(), env.place_pos.copy())
                self.assertGreater(plan.actions.shape[0], 0)
                raw_dir = tmp_root / "raw"
                recorder.execute_plan(plan, 0, raw_dir, asdict(rand))
            finally:
                env.close()

            exporter = LeRobotV2Exporter(fps=cfg["env"]["control_freq"])
            summary = exporter.export(raw_dir, tmp_root / "dataset", filter_failures=True)
            self.assertGreaterEqual(summary.episodes_exported, 1)
            self.assertTrue((tmp_root / "dataset" / "meta" / "info.json").exists())
            self.assertTrue((tmp_root / "dataset" / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000000.mp4").exists())


if __name__ == "__main__":
    unittest.main()
