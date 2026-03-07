from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from v2.env.table_env_v2 import TableEnvV2
from v2.ik.trajectory_generator import IKTrajectoryGenerator
from v2.randomization.domain_randomizer import DomainRandomizer


def load_config() -> dict:
    return yaml.safe_load(Path("/home/ozkan/projects/humanoid_vla/v2/config/data_gen_config.yaml").read_text())


class TestIKSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.env = TableEnvV2(self.cfg)

    def tearDown(self) -> None:
        self.env.close()

    def test_render_and_save_frame(self) -> None:
        obs = self.env.get_observation()
        self.assertEqual(obs["static_cam_image"].shape, (224, 224, 3))
        self.assertEqual(obs["wrist_cam_image"].shape, (224, 224, 3))
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "static.png"
            ok = cv2.imwrite(str(out), cv2.cvtColor(obs["static_cam_image"], cv2.COLOR_RGB2BGR))
            self.assertTrue(ok)
            self.assertTrue(out.exists())

    def test_basic_pick_and_place_plan(self) -> None:
        rng = np.random.default_rng(7)
        randomizer = DomainRandomizer(self.env, self.cfg)
        randomizer.apply(rng)
        planner = IKTrajectoryGenerator(self.env, self.cfg)
        plan = planner.generate_pick_and_place(rng, self.env.object_pos.copy(), self.env.place_pos.copy())
        self.assertGreater(plan.actions.shape[0], 0)
        self.assertGreaterEqual(len(plan.phases), 6)


if __name__ == "__main__":
    unittest.main()
