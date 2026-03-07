from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from v2.env.table_env_v2 import TableEnvV2
from v2.randomization.domain_randomizer import DomainRandomizer


def load_config() -> dict:
    return yaml.safe_load(Path("/home/ozkan/projects/humanoid_vla/v2/config/data_gen_config.yaml").read_text())


class TestRandomization(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.env = TableEnvV2(self.cfg)
        self.randomizer = DomainRandomizer(self.env, self.cfg)

    def tearDown(self) -> None:
        self.env.close()

    def test_randomization_changes_scene(self) -> None:
        rng = np.random.default_rng(123)
        first = self.randomizer.apply(rng)
        pos1 = first.object_position.copy()
        rgba1 = self.env.get_object_rgba().copy()
        second = self.randomizer.apply(rng)
        pos2 = second.object_position.copy()
        rgba2 = self.env.get_object_rgba().copy()
        self.assertFalse(np.allclose(pos1, pos2))
        self.assertFalse(np.allclose(rgba1, rgba2))
        self.assertTrue(self.cfg["object"]["position_range"]["x"][0] <= pos2[0] <= self.cfg["object"]["position_range"]["x"][1])


if __name__ == "__main__":
    unittest.main()
