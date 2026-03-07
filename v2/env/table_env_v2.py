"""v2/env/table_env_v2.py — Physics-based bimanual manipulation environment.

Uses mj_step with PD controller for real contact forces and friction-based
grasping.  Both arms squeeze the green box from the sides — no weld constraints.

Architecture:
  Physics rate : 500 Hz  (mj_step)
  Control rate : 30 Hz   (1 frame = target_pos -> PD -> 16x mj_step -> render)
  PD controller: tau = Kp*(q_des - q) - Kd*q_dot + gravity_comp, clipped
  Fixed base   : pelvis + legs + waist frozen every substep (only arms move)
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from v2.common import (
    ACTUATED_DOF_START,
    ACTUATED_DOF_END,
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    BOX_JOINT_NAME,
    BOX_VISUAL_GEOM_NAME,
    EGO_CAMERA_NAME,
    KD,
    KP,
    LEFT_ARM_CTRL,
    LEFT_ARM_DOF,
    LEFT_HAND_SITE,
    LEFT_PALM_GEOM,
    NUM_ACTUATORS,
    RIGHT_ARM_CTRL,
    RIGHT_ARM_DOF,
    RIGHT_HAND_SITE,
    RIGHT_PALM_GEOM,
    SOURCE_SCENE_XML,
    TABLE_BODY_NAME,
)


class TableEnvV2:
    """MuJoCo environment with full physics for bimanual box manipulation.

    The robot holds a green box purely by friction from both palms squeezing
    inward.  No weld constraints.  PD controller converts position targets to
    torques.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        env_cfg = config["env"]
        self.control_hz = int(env_cfg.get("control_freq", 30))
        physics_hz = int(env_cfg.get("physics_hz", 500))
        self.substeps = physics_hz // self.control_hz

        cam_cfg = env_cfg.get("camera", {})
        self.render_h, self.render_w = cam_cfg.get("resolution", [480, 640])

        # Load model directly from the scene XML (no generated copy needed)
        self.model = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE_XML))
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # ---- Cache IDs ----
        self.left_hand_site_id = self._site_id(LEFT_HAND_SITE)
        self.right_hand_site_id = self._site_id(RIGHT_HAND_SITE)
        self.box_body_id = self._body_id(BOX_BODY_NAME)
        self.box_joint_id = self._joint_id(BOX_JOINT_NAME)
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.box_geom_id = self._geom_id(BOX_GEOM_NAME)
        self.box_visual_geom_id = self._geom_id(BOX_VISUAL_GEOM_NAME)
        self.left_palm_geom_id = self._geom_id(LEFT_PALM_GEOM)
        self.right_palm_geom_id = self._geom_id(RIGHT_PALM_GEOM)
        self.table_body_id = self._body_id(TABLE_BODY_NAME)
        self.table_geom_id = self._find_table_geom_id()
        self.floor_geom_id = self._geom_id("floor")
        self.ego_camera_id = self._camera_id(EGO_CAMERA_NAME)

        # Hide red_cube and place_marker -- not used in bimanual sim
        for name in ("cube_geom", "place_marker"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_rgba[gid, 3] = 0.0
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

        # ---- Arm joint addresses & limits ----
        self.left_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in LEFT_ARM_CTRL], dtype=np.int32)
        self.right_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in RIGHT_ARM_CTRL], dtype=np.int32)

        self.left_arm_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in LEFT_ARM_CTRL], dtype=np.float64)
        self.left_arm_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in LEFT_ARM_CTRL], dtype=np.float64)
        self.right_arm_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in RIGHT_ARM_CTRL], dtype=np.float64)
        self.right_arm_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in RIGHT_ARM_CTRL], dtype=np.float64)

        self._ctrlrange = self.model.actuator_ctrlrange.copy()

        # ---- Locked (non-arm) joints ----
        _non_arm_ctrl = list(range(15))  # ctrl [0..14] = legs + waist
        self._locked_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in _non_arm_ctrl], dtype=np.int32)
        self._locked_qvel_idx = self._locked_qpos_adr - 1

        # ---- PD target vector (29 actuators) ----
        self.target_pos = np.zeros(NUM_ACTUATORS, dtype=np.float64)

        # ---- Save initial state ----
        self._base_qpos = self.data.qpos[:7].copy()
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()
        self._init_box_qpos = self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 7].copy()

        # Distractor IDs (for domain randomization)
        self.distractor_geom_ids: list[int] = []
        self.distractor_body_ids: list[int] = []
        for gname, bname in [("dist_box", "distractor_0"),
                              ("dist_cyl", "distractor_1"),
                              ("dist_sphere", "distractor_2")]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if gid >= 0 and bid >= 0:
                self.distractor_geom_ids.append(gid)
                self.distractor_body_ids.append(bid)

        # Renderer (offscreen)
        self.renderer = mujoco.Renderer(self.model, height=self.render_h, width=self.render_w)

    # ---- Cleanup ----

    def close(self) -> None:
        self.renderer.close()

    # ---- ID helpers ----

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"Site not found: {name}")
        return sid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body not found: {name}")
        return bid

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found: {name}")
        return jid

    def _geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise KeyError(f"Geom not found: {name}")
        return gid

    def _camera_id(self, name: str) -> int:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise KeyError(f"Camera not found: {name}")
        return cid

    def _find_table_geom_id(self) -> int:
        for gid in range(self.model.ngeom):
            if (self.model.geom_bodyid[gid] == self.table_body_id
                    and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX):
                return gid
        raise RuntimeError("Table geom not found")

    # ---- Properties ----

    @property
    def left_hand_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.left_hand_site_id].copy()

    @property
    def right_hand_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.right_hand_site_id].copy()

    @property
    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_body_id].copy()

    @property
    def left_arm_q(self) -> np.ndarray:
        return self.data.qpos[self.left_arm_qpos_adr].copy()

    @property
    def right_arm_q(self) -> np.ndarray:
        return self.data.qpos[self.right_arm_qpos_adr].copy()

    # ---- Reset ----

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()
        self.target_pos[:] = 0.0

    def reset_box_with_noise(self, rng: np.random.Generator,
                             noise_x: float = 0.02, noise_y: float = 0.02) -> None:
        """Displace the box randomly and let it settle on the table."""
        self.data.qpos[self.box_qpos_adr + 0] += rng.uniform(-noise_x, noise_x)
        self.data.qpos[self.box_qpos_adr + 1] += rng.uniform(-noise_y, noise_y)
        mujoco.mj_forward(self.model, self.data)
        # Let box settle on table
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        # Re-freeze base after settling
        self.data.qpos[:7] = self._base_qpos
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- PD Controller + Physics Step ----

    def _compute_pd_torques(self) -> np.ndarray:
        """tau = Kp*(q_des - q) - Kd*q_dot + gravity_comp, clipped."""
        q = self.data.actuator_length.copy()
        qd = self.data.actuator_velocity.copy()
        tau = KP * (self.target_pos - q) - KD * qd
        # Gravity + Coriolis compensation
        tau += self.data.qfrc_bias[ACTUATED_DOF_START:ACTUATED_DOF_END]
        tau = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        return tau

    def step_frame(self) -> None:
        """Run one control frame: substeps x (PD -> mj_step -> freeze non-arms)."""
        for _ in range(self.substeps):
            np.copyto(self.data.ctrl, self._compute_pd_torques())
            mujoco.mj_step(self.model, self.data)
            # Fixed base: freeze pelvis
            self.data.qpos[:7] = self._base_qpos
            self.data.qvel[:6] = 0.0
            # Freeze legs + waist
            self.data.qpos[self._locked_qpos_adr] = self._locked_qpos_vals
            self.data.qvel[self._locked_qvel_idx] = 0.0

    # ---- IK Solvers ----

    def solve_ik_left(self, target_xyz: np.ndarray, max_iter: int = 500,
                      tol: float = 0.01, step: float = 0.02,
                      damping: float = 0.05) -> bool:
        """Position-only IK for the left arm."""
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        for _ in range(max_iter):
            error = target_xyz - self.left_hand_pos
            if np.linalg.norm(error) < tol:
                return True
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, None,
                              self.left_hand_site_id)
            J = jacp[:, LEFT_ARM_DOF]
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            dq_n = np.linalg.norm(dq)
            if dq_n > step:
                dq *= step / dq_n
            q = self.data.qpos[self.left_arm_qpos_adr] + dq
            self.data.qpos[self.left_arm_qpos_adr] = np.clip(
                q, self.left_arm_lo, self.left_arm_hi)
            mujoco.mj_forward(self.model, self.data)
        return False

    def solve_ik_right(self, target_xyz: np.ndarray, max_iter: int = 500,
                       tol: float = 0.01, step: float = 0.02,
                       damping: float = 0.05) -> bool:
        """Position-only IK for the right arm."""
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        for _ in range(max_iter):
            error = target_xyz - self.right_hand_pos
            if np.linalg.norm(error) < tol:
                return True
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, None,
                              self.right_hand_site_id)
            J = jacp[:, RIGHT_ARM_DOF]
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            dq_n = np.linalg.norm(dq)
            if dq_n > step:
                dq *= step / dq_n
            q = self.data.qpos[self.right_arm_qpos_adr] + dq
            self.data.qpos[self.right_arm_qpos_adr] = np.clip(
                q, self.right_arm_lo, self.right_arm_hi)
            mujoco.mj_forward(self.model, self.data)
        return False

    # ---- Observation ----

    def render_camera(self, camera_name: str | None = None) -> np.ndarray:
        """Render a camera image. Defaults to ego_camera."""
        if camera_name is None:
            camera_name = EGO_CAMERA_NAME
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_positions, joint_velocities) for all 29 actuators."""
        return (
            self.data.actuator_length.copy().astype(np.float32),
            self.data.actuator_velocity.copy().astype(np.float32),
        )

    def get_arm_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (14D arm positions, 14D arm velocities)."""
        pos_all, vel_all = self.get_obs()
        arm_pos = np.concatenate([pos_all[LEFT_ARM_CTRL], pos_all[RIGHT_ARM_CTRL]])
        arm_vel = np.concatenate([vel_all[LEFT_ARM_CTRL], vel_all[RIGHT_ARM_CTRL]])
        return arm_pos, arm_vel

    # ---- Contact Monitoring ----

    def get_palm_box_contacts(self) -> dict[str, Any]:
        """Check for active contacts between palm pads and the green box."""
        left_contact = False
        right_contact = False
        left_force = 0.0
        right_force = 0.0

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.box_geom_id in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                fn = np.linalg.norm(force[:3])
                if self.left_palm_geom_id in pair:
                    left_contact = True
                    left_force += fn
                if self.right_palm_geom_id in pair:
                    right_contact = True
                    right_force += fn

        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_force": left_force,
            "right_force": right_force,
        }

    # ---- Randomization helpers ----

    def set_box_color(self, rgba: np.ndarray) -> None:
        """Set green box visual colour (collision geom stays invisible)."""
        self.model.geom_rgba[self.box_visual_geom_id] = np.asarray(rgba, dtype=np.float64)

    def get_box_color(self) -> np.ndarray:
        return self.model.geom_rgba[self.box_visual_geom_id].copy()

    def set_box_mass(self, mass: float) -> None:
        self.model.body_mass[self.box_body_id] = float(mass)

    def get_box_mass(self) -> float:
        return float(self.model.body_mass[self.box_body_id])

    def set_box_friction(self, friction_triplet: np.ndarray) -> None:
        self.model.geom_friction[self.box_geom_id] = np.asarray(friction_triplet, dtype=np.float64)

    def get_box_friction(self) -> np.ndarray:
        return self.model.geom_friction[self.box_geom_id].copy()

    def set_table_color(self, rgba: np.ndarray) -> None:
        self.model.geom_rgba[self.table_geom_id] = np.asarray(rgba, dtype=np.float64)

    def set_floor_color(self, rgba: np.ndarray) -> None:
        self.model.geom_rgba[self.floor_geom_id] = np.asarray(rgba, dtype=np.float64)

    def set_light(self, pos: np.ndarray, diffuse: np.ndarray) -> None:
        if self.model.nlight <= 0:
            return
        self.model.light_pos[0] = np.asarray(pos, dtype=np.float64)
        self.model.light_diffuse[0] = np.asarray(diffuse, dtype=np.float64)

    def set_camera_pose(self, camera_name: str, pos: np.ndarray,
                        quat: np.ndarray) -> None:
        cid = self._camera_id(camera_name)
        self.model.cam_pos[cid] = np.asarray(pos, dtype=np.float64)
        self.model.cam_quat[cid] = np.asarray(quat, dtype=np.float64)

    def get_camera_pose(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        cid = self._camera_id(camera_name)
        return self.model.cam_pos[cid].copy(), self.model.cam_quat[cid].copy()
