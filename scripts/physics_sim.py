#!/usr/bin/env python3
"""
scripts/physics_sim.py

Physics-based MuJoCo simulation wrapper for bimanual manipulation.
Uses mj_step (real contact forces, friction, gravity) — NOT kinematic mode.

This is fundamentally different from the kinematic SimWrapper in generate_demos.py:
  - Kinematic: set qpos directly → mj_forward (no forces, no contact)
  - Physics:   set ctrl torques → mj_step (real dynamics, contact, friction)

The robot holds a box purely by friction from both palms squeezing inward.
No weld constraints. The PD controller converts position targets to torques.

Architecture:
  Control rate: 30 Hz (one "frame" = target_pos → render → record)
  Physics rate: 500 Hz (multiple mj_step per control frame)
  PD controller: τ = Kp*(q_des - q) - Kd*q̇ + gravity_comp
  Fixed base: pelvis kinematically frozen each substep
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sim", "g1_with_camera.xml"))

CAMERA_NAME = "ego_camera"
RENDER_W, RENDER_H = 640, 480

NUM_ACTUATORS = 29
PHYSICS_HZ = 500
CONTROL_HZ = 30
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ   # ~16-17 physics steps per control frame

# Arm actuator indices (into ctrl[])
LEFT_ARM_CTRL = np.array([15, 16, 17, 18, 19, 20, 21])
RIGHT_ARM_CTRL = np.array([22, 23, 24, 25, 26, 27, 28])
BOTH_ARMS_CTRL = np.concatenate([LEFT_ARM_CTRL, RIGHT_ARM_CTRL])  # 14 DOF

# Arm DOF indices (into qvel[])
LEFT_ARM_DOF = np.array([21, 22, 23, 24, 25, 26, 27])
RIGHT_ARM_DOF = np.array([28, 29, 30, 31, 32, 33, 34])

# PD gains — stiff enough for trajectory tracking + squeeze force maintenance.
# Higher than real-time bridge gains, but lower than kinematic demo gains.
# Arms need enough stiffness to maintain pressure against the box.
# fmt: off
_KP = np.array([
    44, 44, 44, 70, 25, 25,         # left leg
    44, 44, 44, 70, 25, 25,         # right leg
    44, 25, 25,                      # waist
    40, 40, 40, 40, 10, 10, 10,     # left arm
    40, 40, 40, 40, 10, 10, 10,     # right arm
], dtype=np.float64)

_KD = np.array([
     4,  4,  4,  7, 2.5, 2.5,
     4,  4,  4,  7, 2.5, 2.5,
     4, 2.5, 2.5,
     4,  4,  4,  4,  1,  1,  1,    # left arm
     4,  4,  4,  4,  1,  1,  1,    # right arm
], dtype=np.float64)
# fmt: on

# Freejoint DOF offset: floating_base(7) occupies qpos[0:7], qvel[0:6]
_ACTUATED_DOF_START = 6
_ACTUATED_DOF_END = 35


# ────────────────────────────────────────────────────────
# Physics simulation wrapper
# ────────────────────────────────────────────────────────

class PhysicsSim:
    """MuJoCo simulation with real physics (mj_step), PD control, gravity comp.

    Unlike SimWrapper (kinematic), this wrapper:
      - Runs mj_step for real contact forces and friction
      - Uses PD controller to track position targets
      - Does NOT use weld constraints — grasping is via friction only
      - Supports bimanual control (both arms)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Cache body/site/geom IDs
        self.left_hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
        self.right_hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
        self.box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        self.left_palm_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_palm_pad")
        self.right_palm_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_palm_pad")

        # Also keep cube IDs for backward compat
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")

        # Arm joint qpos addresses and limits
        self.left_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in LEFT_ARM_CTRL])
        self.right_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in RIGHT_ARM_CTRL])

        # Joint limits for both arms
        self.left_arm_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in LEFT_ARM_CTRL])
        self.left_arm_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in LEFT_ARM_CTRL])
        self.right_arm_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in RIGHT_ARM_CTRL])
        self.right_arm_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in RIGHT_ARM_CTRL])

        # ctrlrange for torque clipping
        self._ctrlrange = self.model.actuator_ctrlrange.copy()

        # Save initial state
        self._base_qpos = self.data.qpos[:7].copy()
        self._init_box_qpos = self.data.qpos[self.box_qpos_adr:self.box_qpos_adr+7].copy()

        # Indices of non-arm actuated qpos (legs + waist, locked during physics)
        # Legs: ctrl [0..11] = 12 DOF, Waist: ctrl [12..14] = 3 DOF → total 15
        # These joints are frozen kinematically so only arms move under physics.
        _non_arm_ctrl = list(range(15))  # ctrl indices 0..14
        self._locked_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in _non_arm_ctrl])
        self._locked_qvel_adr = np.array([
            self.model.actuator_trnid[ci, 0]  # DOF index for 1-DOF joints
            for ci in _non_arm_ctrl])
        # Compute correct qvel addresses: for non-floating joints, qvel_adr = qpos_adr - 1
        # (because floating base uses 7 qpos but 6 qvel)
        self._locked_qvel_idx = self._locked_qpos_adr - 1  # maps to qvel indices
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()

        # Target positions for PD controller
        self.target_pos = np.zeros(NUM_ACTUATORS)

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)

    # ── Properties ──────────────────────────────────────

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

    # ── Reset ───────────────────────────────────────────

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()
        self.target_pos[:] = 0.0

    def reset_with_noise(self, rng: np.random.Generator):
        """Reset with small random displacement to box position."""
        self.reset()
        self.data.qpos[self.box_qpos_adr + 0] += rng.uniform(-0.02, 0.02)
        self.data.qpos[self.box_qpos_adr + 1] += rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.model, self.data)
        # Let box settle on table (a few physics steps)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        # Freeze base after settling
        self.data.qpos[:7] = self._base_qpos
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ── PD Controller + Physics Step ────────────────────

    def _compute_pd_torques(self) -> np.ndarray:
        """τ = Kp*(q_des - q) - Kd*q̇ + gravity_comp, clipped to ctrlrange."""
        q = self.data.actuator_length.copy()
        qd = self.data.actuator_velocity.copy()
        tau = _KP * (self.target_pos - q) - _KD * qd
        # Gravity + Coriolis compensation
        tau += self.data.qfrc_bias[_ACTUATED_DOF_START:_ACTUATED_DOF_END]
        tau = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        return tau

    def step_frame(self):
        """Run one control frame: multiple physics substeps with PD control.

        This is the core physics loop:
        1. Compute PD torques from target_pos
        2. Set data.ctrl
        3. Run SUBSTEPS × mj_step (500Hz physics)
        4. Freeze pelvis + legs + waist after each step (only arms move)
        """
        for _ in range(SUBSTEPS):
            np.copyto(self.data.ctrl, self._compute_pd_torques())
            mujoco.mj_step(self.model, self.data)
            # Fixed base: freeze pelvis
            self.data.qpos[:7] = self._base_qpos
            self.data.qvel[:6] = 0.0
            # Freeze legs + waist (non-arm actuated joints)
            self.data.qpos[self._locked_qpos_adr] = self._locked_qpos_vals
            self.data.qvel[self._locked_qvel_idx] = 0.0

    # ── IK Solvers ──────────────────────────────────────

    def solve_ik_left(self, target_xyz: np.ndarray, max_iter=500,
                      tol=0.01, step=0.02, damping=0.05) -> bool:
        """Position-only IK for left arm using iterative Jacobian."""
        for _ in range(max_iter):
            error = target_xyz - self.left_hand_pos
            if np.linalg.norm(error) < tol:
                return True
            jacp = np.zeros((3, self.model.nv))
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

    def solve_ik_right(self, target_xyz: np.ndarray, max_iter=500,
                       tol=0.01, step=0.02, damping=0.05) -> bool:
        """Position-only IK for right arm using iterative Jacobian."""
        for _ in range(max_iter):
            error = target_xyz - self.right_hand_pos
            if np.linalg.norm(error) < tol:
                return True
            jacp = np.zeros((3, self.model.nv))
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

    # ── Observation ─────────────────────────────────────

    def render_camera(self) -> np.ndarray:
        """Render ego camera image (480×640×3 RGB uint8)."""
        self.renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self.renderer.render().copy()

    def get_obs(self):
        """Return (joint_positions, joint_velocities) for all 29 actuators."""
        return (
            self.data.actuator_length.copy().astype(np.float32),
            self.data.actuator_velocity.copy().astype(np.float32),
        )

    # ── Contact Monitoring ──────────────────────────────

    def get_palm_box_contacts(self) -> dict:
        """Check for active contacts between palm pads and the green box.

        Returns dict with:
          left_contact: bool — left palm touching box
          right_contact: bool — right palm touching box
          left_force: float — total contact force (normal) on left palm
          right_force: float — total contact force (normal) on right palm
        """
        left_contact = False
        right_contact = False
        left_force = 0.0
        right_force = 0.0

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}

            if self.box_geom_id in pair:
                # Get contact force magnitude
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                fn = np.linalg.norm(force[:3])  # normal force

                if self.left_palm_geom_id in pair:
                    left_contact = True
                    left_force += fn
                if self.right_palm_geom_id in pair:
                    right_contact = True
                    right_force += fn

        return {
            'left_contact': left_contact,
            'right_contact': right_contact,
            'left_force': left_force,
            'right_force': right_force,
        }
