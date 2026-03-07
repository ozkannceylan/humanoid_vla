"""v2/randomization/domain_randomizer.py — Domain randomization for bimanual physics.

Randomises visual appearance + physics properties per episode:
  - Box colour (green hue preserved)
  - Box mass, friction (critical for grasp stability)
  - Box position noise
  - Table colour, floor colour
  - Lighting direction, intensity, colour temperature
  - Ego camera pose jitter
  - Distractor objects (visibility, position, colour)
  - Robot arm PD gains (small perturbation)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from v2.common import (
    BOX_NOMINAL_POS,
    EGO_CAMERA_NAME,
    TABLE_CENTER,
    quat_mul,
    euler_deg_to_quat,
    spherical_to_cartesian,
)
from v2.env.table_env_v2 import TableEnvV2


# ---- Colour palettes ----

TABLE_VARIATIONS = [
    np.array([0.62, 0.40, 0.22, 1.0]),
    np.array([0.50, 0.36, 0.24, 1.0]),
    np.array([0.72, 0.72, 0.72, 1.0]),
    np.array([0.35, 0.37, 0.42, 1.0]),
    np.array([0.85, 0.82, 0.78, 1.0]),
]

FLOOR_VARIATIONS = [
    np.array([0.16, 0.22, 0.30, 1.0]),
    np.array([0.24, 0.24, 0.24, 1.0]),
    np.array([0.28, 0.22, 0.18, 1.0]),
    np.array([0.18, 0.28, 0.22, 1.0]),
]

COLOR_TEMPS = {
    "warm": np.array([1.0, 0.90, 0.80], dtype=np.float64),
    "neutral": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    "cool": np.array([0.85, 0.92, 1.0], dtype=np.float64),
}


@dataclass
class EpisodeRandomization:
    """Record of randomisation applied to an episode."""
    box_pos_noise: np.ndarray       # (dx, dy) applied
    box_mass: float
    box_friction: float             # slide friction
    box_color_rgba: np.ndarray
    table_rgba: np.ndarray
    floor_rgba: np.ndarray
    light_intensity: float
    ego_cam_jitter: np.ndarray      # (3,) position delta


class DomainRandomizer:
    """Apply per-episode visual + physics randomisation."""

    def __init__(self, env: TableEnvV2, config: dict[str, Any]):
        self.env = env
        self.config = config
        rand_cfg = config.get("randomization", {})
        self.box_cfg = rand_cfg.get("box", {})
        self.light_cfg = rand_cfg.get("lighting", {})
        self.camera_cfg = rand_cfg.get("camera", {})
        self.robot_cfg = rand_cfg.get("robot", {})
        self.distractor_cfg = rand_cfg.get("distractors", {})

        # Save nominal values for restore
        self._nominal = {
            "box_mass": env.get_box_mass(),
            "box_friction": env.get_box_friction().copy(),
            "box_color": env.get_box_color().copy(),
            "table_rgba": env.model.geom_rgba[env.table_geom_id].copy(),
            "floor_rgba": env.model.geom_rgba[env.floor_geom_id].copy(),
        }

        if env.model.nlight > 0:
            self._nominal["light_pos"] = env.model.light_pos[0].copy()
            self._nominal["light_diffuse"] = env.model.light_diffuse[0].copy()

        try:
            self._nominal["ego_cam"] = env.get_camera_pose(EGO_CAMERA_NAME)
        except KeyError:
            self._nominal["ego_cam"] = None

        for i, gid in enumerate(env.distractor_geom_ids):
            self._nominal[f"dist_{i}_rgba"] = env.model.geom_rgba[gid].copy()

        # Robot dynamics
        self._nominal["dof_damping"] = env.model.dof_damping.copy()
        self._nominal["actuator_gainprm"] = env.model.actuator_gainprm.copy()

    def restore(self) -> None:
        """Restore all nominal values."""
        env = self.env
        env.set_box_mass(self._nominal["box_mass"])
        env.set_box_friction(self._nominal["box_friction"])
        env.set_box_color(self._nominal["box_color"])
        env.set_table_color(self._nominal["table_rgba"])
        env.set_floor_color(self._nominal["floor_rgba"])
        if env.model.nlight > 0 and "light_pos" in self._nominal:
            env.set_light(self._nominal["light_pos"], self._nominal["light_diffuse"])
        if self._nominal["ego_cam"] is not None:
            pos, quat = self._nominal["ego_cam"]
            env.set_camera_pose(EGO_CAMERA_NAME, pos, quat)
        for i, gid in enumerate(env.distractor_geom_ids):
            env.model.geom_rgba[gid] = self._nominal[f"dist_{i}_rgba"]
        env.model.dof_damping[:] = self._nominal["dof_damping"]
        env.model.actuator_gainprm[:] = self._nominal["actuator_gainprm"]

    def apply(self, rng: np.random.Generator) -> EpisodeRandomization:
        """Apply all randomisations for one episode.  Returns metadata."""
        self.restore()
        env = self.env

        # ---- Box position noise ----
        noise_x = self.box_cfg.get("noise_x", 0.02)
        noise_y = self.box_cfg.get("noise_y", 0.02)
        env.reset_box_with_noise(rng, noise_x=noise_x, noise_y=noise_y)
        box_dx = env.box_pos[0] - BOX_NOMINAL_POS[0]
        box_dy = env.box_pos[1] - BOX_NOMINAL_POS[1]

        # ---- Box mass ----
        mass_lo, mass_hi = self.box_cfg.get("mass_range", [0.2, 0.5])
        box_mass = float(rng.uniform(mass_lo, mass_hi))
        env.set_box_mass(box_mass)

        # ---- Box friction (critical for grasp!) ----
        fric_lo, fric_hi = self.box_cfg.get("friction_range", [1.0, 2.0])
        box_fric = float(rng.uniform(fric_lo, fric_hi))
        env.set_box_friction(np.array([box_fric, 0.005, 0.0001], dtype=np.float64))

        # ---- Box colour (keep greenish) ----
        box_rgba = np.array([
            rng.uniform(0.05, 0.30),
            rng.uniform(0.45, 0.90),
            rng.uniform(0.05, 0.30),
            1.0], dtype=np.float64)
        env.set_box_color(box_rgba)

        # ---- Table colour ----
        table_rgba = TABLE_VARIATIONS[int(rng.integers(len(TABLE_VARIATIONS)))]
        env.set_table_color(table_rgba)

        # ---- Floor colour ----
        floor_rgba = FLOOR_VARIATIONS[int(rng.integers(len(FLOOR_VARIATIONS)))]
        env.set_floor_color(floor_rgba)

        # ---- Lighting ----
        light_intensity = float(rng.uniform(
            self.light_cfg.get("intensity_min", 0.3),
            self.light_cfg.get("intensity_max", 0.9)))
        if env.model.nlight > 0 and "light_pos" in self._nominal:
            azimuth = float(rng.uniform(0, 360))
            elevation = float(rng.uniform(20, 80))
            direction = spherical_to_cartesian(1.6, azimuth, elevation)
            light_pos = TABLE_CENTER + direction
            temps = list(COLOR_TEMPS.keys())
            color_temp = temps[int(rng.integers(len(temps)))]
            diffuse = COLOR_TEMPS[color_temp] * light_intensity
            env.set_light(light_pos, diffuse)

        # ---- Ego camera jitter ----
        cam_jitter = np.zeros(3)
        jitter_cm = self.camera_cfg.get("position_jitter_cm", 2.0)
        jitter_deg = self.camera_cfg.get("rotation_jitter_deg", 3.0)
        if self._nominal["ego_cam"] is not None:
            nom_pos, nom_quat = self._nominal["ego_cam"]
            cam_jitter = rng.uniform(-jitter_cm / 100.0, jitter_cm / 100.0, 3)
            new_pos = nom_pos + cam_jitter
            dtheta = rng.uniform(-np.deg2rad(jitter_deg), np.deg2rad(jitter_deg), 3)
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            new_quat = quat_mul(nom_quat, dq)
            env.set_camera_pose(EGO_CAMERA_NAME, new_pos, new_quat)

        # ---- Distractors ----
        show_prob = self.distractor_cfg.get("show_probability", 0.5)
        for gid, bid in zip(env.distractor_geom_ids, env.distractor_body_ids):
            if rng.random() < show_prob:
                env.model.geom_rgba[gid] = [
                    rng.uniform(0.2, 1.0),
                    rng.uniform(0.2, 1.0),
                    rng.uniform(0.2, 1.0),
                    1.0]
                env.model.body_pos[bid] = [
                    0.3 + rng.uniform(-0.15, 0.15),
                    -0.1 + rng.uniform(-0.15, 0.15),
                    0.83]
            else:
                env.model.geom_rgba[gid, 3] = 0.0

        # ---- Robot dynamics perturbation ----
        if self.robot_cfg.get("enabled", False):
            damping_lo, damping_hi = self.robot_cfg.get("damping_scale", [0.9, 1.1])
            gain_lo, gain_hi = self.robot_cfg.get("gain_scale", [0.95, 1.05])
            env.model.dof_damping[:] = self._nominal["dof_damping"] * rng.uniform(damping_lo, damping_hi)
            env.model.actuator_gainprm[:] = self._nominal["actuator_gainprm"]
            if env.model.actuator_gainprm.size:
                env.model.actuator_gainprm[:, 0] *= rng.uniform(gain_lo, gain_hi)

        mujoco.mj_forward(env.model, env.data)

        return EpisodeRandomization(
            box_pos_noise=np.array([box_dx, box_dy], dtype=np.float32),
            box_mass=box_mass,
            box_friction=box_fric,
            box_color_rgba=box_rgba.astype(np.float32),
            table_rgba=table_rgba.astype(np.float32),
            floor_rgba=floor_rgba.astype(np.float32),
            light_intensity=light_intensity,
            ego_cam_jitter=cam_jitter.astype(np.float32),
        )
