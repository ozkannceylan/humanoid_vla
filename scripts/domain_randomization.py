#!/usr/bin/env python3
"""
scripts/domain_randomization.py

Runtime visual randomization for MuJoCo scenes (Phase F).
Modifies model attributes (geom_rgba, light_pos, cam_pos, etc.) at runtime
before calling mj_forward — no XML changes needed per episode.

Usage:
    randomizer = DomainRandomizer(model, data)
    randomizer.randomize(rng)  # call after reset, before recording
    randomizer.restore()       # optional: restore nominal appearance
"""

import numpy as np
import mujoco


class DomainRandomizer:
    """Runtime visual domain randomization for MuJoCo.

    Randomizes:
      - Table color (brown/grey/white spectrum)
      - Floor color variation
      - Lighting direction and intensity
      - Object color (hue-preserving brightness variation)
      - Distractor object visibility and position
      - Camera pose perturbation (small position + orientation jitter)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Look up geom IDs (returns -1 if not found)
        def _geom_id(name):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            return gid if gid >= 0 else None

        def _body_id(name):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            return bid if bid >= 0 else None

        def _cam_id(name):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            return cid if cid >= 0 else None

        # Table geom (unnamed in XML — it's the first geom in the "table" body)
        table_body_id = _body_id("table")
        self.table_geom_id = None
        if table_body_id is not None:
            # Find first geom belonging to table body
            for i in range(model.ngeom):
                if model.geom_bodyid[i] == table_body_id and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                    self.table_geom_id = i
                    break

        self.floor_geom_id = _geom_id("floor")
        self.cube_geom_id = _geom_id("cube_geom")
        self.box_geom_id = _geom_id("box_geom")
        self.cam_id = _cam_id("ego_camera")

        # Distractor geoms
        self.distractor_geom_ids = []
        self.distractor_body_ids = []
        for name_g, name_b in [("dist_box", "distractor_0"),
                                ("dist_cyl", "distractor_1"),
                                ("dist_sphere", "distractor_2")]:
            gid = _geom_id(name_g)
            bid = _body_id(name_b)
            if gid is not None and bid is not None:
                self.distractor_geom_ids.append(gid)
                self.distractor_body_ids.append(bid)

        # Save nominal values for restore
        self._nominal = {}
        if self.table_geom_id is not None:
            self._nominal['table_rgba'] = model.geom_rgba[self.table_geom_id].copy()
        if self.floor_geom_id is not None:
            self._nominal['floor_rgba'] = model.geom_rgba[self.floor_geom_id].copy()
        if self.cube_geom_id is not None:
            self._nominal['cube_rgba'] = model.geom_rgba[self.cube_geom_id].copy()
        if self.box_geom_id is not None:
            self._nominal['box_rgba'] = model.geom_rgba[self.box_geom_id].copy()
        if model.nlight > 0:
            self._nominal['light_pos'] = model.light_pos[0].copy()
            self._nominal['light_diffuse'] = model.light_diffuse[0].copy()
            self._nominal['light_dir'] = model.light_dir[0].copy()
        if self.cam_id is not None:
            self._nominal['cam_pos'] = model.cam_pos[self.cam_id].copy()
            self._nominal['cam_quat'] = model.cam_quat[self.cam_id].copy()
        for i, gid in enumerate(self.distractor_geom_ids):
            self._nominal[f'dist_{i}_rgba'] = model.geom_rgba[gid].copy()

    def randomize(self, rng: np.random.Generator,
                  visual: bool = True, camera: bool = True):
        """Apply random visual changes to the scene.

        Args:
            rng: numpy random generator
            visual: randomize colors, lighting, distractors
            camera: randomize ego camera pose
        """
        model = self.model

        if visual:
            # Table color: brown/grey/white spectrum
            if self.table_geom_id is not None:
                model.geom_rgba[self.table_geom_id] = [
                    rng.uniform(0.3, 0.9),
                    rng.uniform(0.2, 0.7),
                    rng.uniform(0.1, 0.5),
                    1.0]

            # Floor color variation
            if self.floor_geom_id is not None:
                model.geom_rgba[self.floor_geom_id][:3] = rng.uniform(0.1, 0.5, 3)

            # Lighting
            if model.nlight > 0:
                # Position perturbation (±0.5m)
                model.light_pos[0] = self._nominal['light_pos'] + rng.uniform(-0.5, 0.5, 3)
                # Intensity variation
                intensity = rng.uniform(0.3, 0.9)
                model.light_diffuse[0] = [intensity] * 3

            # Red cube: vary brightness, keep reddish
            if self.cube_geom_id is not None:
                model.geom_rgba[self.cube_geom_id] = [
                    rng.uniform(0.6, 1.0),
                    rng.uniform(0.0, 0.25),
                    rng.uniform(0.0, 0.15),
                    1.0]

            # Green box: vary brightness, keep greenish
            if self.box_geom_id is not None:
                model.geom_rgba[self.box_geom_id] = [
                    rng.uniform(0.0, 0.3),
                    rng.uniform(0.4, 0.9),
                    rng.uniform(0.0, 0.3),
                    1.0]

            # Distractors: show/hide + reposition on table
            for i, (gid, bid) in enumerate(zip(self.distractor_geom_ids,
                                                self.distractor_body_ids)):
                if rng.random() < 0.5:
                    # Show with random color and position on table
                    model.geom_rgba[gid] = [
                        rng.uniform(0.2, 1.0),
                        rng.uniform(0.2, 1.0),
                        rng.uniform(0.2, 1.0),
                        1.0]
                    model.body_pos[bid] = [
                        0.3 + rng.uniform(-0.15, 0.15),
                        -0.1 + rng.uniform(-0.15, 0.15),
                        0.83]
                else:
                    # Hide
                    model.geom_rgba[gid, 3] = 0.0

        if camera and self.cam_id is not None:
            # Position perturbation: ±2cm
            model.cam_pos[self.cam_id] = (
                self._nominal['cam_pos'] + rng.uniform(-0.02, 0.02, 3))
            # Orientation perturbation: ±3° (0.052 rad) per axis
            dtheta = rng.uniform(-0.052, 0.052, 3)
            # Small-angle quaternion: [1, dx/2, dy/2, dz/2] normalized
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            # Multiply nominal quat by delta
            q0 = self._nominal['cam_quat'].copy()
            model.cam_quat[self.cam_id] = _quat_mul(q0, dq)

    def restore(self):
        """Restore all nominal visual properties."""
        model = self.model
        for key, val in self._nominal.items():
            if key == 'table_rgba' and self.table_geom_id is not None:
                model.geom_rgba[self.table_geom_id] = val
            elif key == 'floor_rgba' and self.floor_geom_id is not None:
                model.geom_rgba[self.floor_geom_id] = val
            elif key == 'cube_rgba' and self.cube_geom_id is not None:
                model.geom_rgba[self.cube_geom_id] = val
            elif key == 'box_rgba' and self.box_geom_id is not None:
                model.geom_rgba[self.box_geom_id] = val
            elif key == 'light_pos' and model.nlight > 0:
                model.light_pos[0] = val
            elif key == 'light_diffuse' and model.nlight > 0:
                model.light_diffuse[0] = val
            elif key == 'light_dir' and model.nlight > 0:
                model.light_dir[0] = val
            elif key == 'cam_pos' and self.cam_id is not None:
                model.cam_pos[self.cam_id] = val
            elif key == 'cam_quat' and self.cam_id is not None:
                model.cam_quat[self.cam_id] = val
            elif key.startswith('dist_'):
                idx = int(key.split('_')[1])
                if idx < len(self.distractor_geom_ids):
                    model.geom_rgba[self.distractor_geom_ids[idx]] = val


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
