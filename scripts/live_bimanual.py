#!/usr/bin/env python3
"""
scripts/live_bimanual.py

Watch bimanual box manipulation in MuJoCo's interactive 3D viewer.
Both hands squeeze a green box from the sides using real physics (mj_step)
and lift it via friction — no weld constraints.

Two modes:
  1. Scripted expert (default) — runs IK-planned trajectory
  2. ACT model inference — runs trained bimanual ACT policy

Controls:
  - Mouse drag: rotate camera
  - Scroll: zoom
  - Double-click body: track it
  - Close window to exit

Usage:
  cd ~/projects/humanoid_vla
  # Scripted expert
  python3 scripts/live_bimanual.py
  python3 scripts/live_bimanual.py --loop --speed 0.3

  # Trained ACT model
  python3 scripts/live_bimanual.py --checkpoint data/bimanual_checkpoints/best.pt
  python3 scripts/live_bimanual.py --checkpoint data/bimanual_checkpoints/best.pt --loop
"""

import argparse
import math
import os
import sys
import time

# Must NOT use EGL for interactive viewer
os.environ.pop("MUJOCO_GL", None)

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics_sim import (
    PhysicsSim, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, BOTH_ARMS_CTRL,
    CONTROL_HZ, SUBSTEPS, NUM_ACTUATORS,
    _KP, _KD, _ACTUATED_DOF_START, _ACTUATED_DOF_END,
    MODEL_PATH, CAMERA_NAME,
)
from generate_bimanual_demos import (
    plan_bimanual_trajectory, BOX_POS_NOMINAL,
    SQUEEZE_OFFSET_Y, LIFT_OFFSET_Z,
)


class LivePhysicsSim:
    """Physics sim for the interactive viewer.

    Same PD controller + joint locking as PhysicsSim, but does NOT
    create an offscreen renderer (the viewer handles rendering).
    Optionally creates an offscreen renderer for ACT model camera input.
    """

    def __init__(self, model_path: str = MODEL_PATH, need_camera: bool = False):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Reuse PhysicsSim's cached IDs via a temporary instance
        _tmp = PhysicsSim.__new__(PhysicsSim)
        _tmp.model = self.model
        _tmp.data = self.data

        # Cache IDs
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

        # Hide red cube and place marker — not used in bimanual sim.
        for name in ("cube_geom", "place_marker"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_rgba[gid, 3] = 0.0
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

        # Arm joint addresses
        self.left_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in LEFT_ARM_CTRL])
        self.right_arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in RIGHT_ARM_CTRL])

        # Joint limits
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

        self._ctrlrange = self.model.actuator_ctrlrange.copy()
        self._base_qpos = self.data.qpos[:7].copy()

        # Lock legs + waist
        _non_arm_ctrl = list(range(15))
        self._locked_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in _non_arm_ctrl])
        self._locked_qvel_idx = self._locked_qpos_adr - 1
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()

        self.target_pos = np.zeros(NUM_ACTUATORS)

        # Offscreen renderer for ACT model camera input (optional)
        self._renderer = None
        if need_camera:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # Properties
    @property
    def left_hand_pos(self):
        return self.data.site_xpos[self.left_hand_site_id].copy()

    @property
    def right_hand_pos(self):
        return self.data.site_xpos[self.right_hand_site_id].copy()

    @property
    def box_pos(self):
        return self.data.xpos[self.box_body_id].copy()

    def render_camera(self):
        """Render ego camera for ACT model input. Requires need_camera=True."""
        assert self._renderer is not None, "need_camera=True required"
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self._renderer.render().copy()

    def get_obs(self):
        """Return (positions, velocities) arrays for all actuators."""
        return (
            self.data.actuator_length.copy().astype(np.float32),
            self.data.actuator_velocity.copy().astype(np.float32),
        )

    @property
    def left_arm_q(self):
        return self.data.qpos[self.left_arm_qpos_adr].copy()

    @property
    def right_arm_q(self):
        return self.data.qpos[self.right_arm_qpos_adr].copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()
        self._locked_qpos_vals = self.data.qpos[self._locked_qpos_adr].copy()
        self.target_pos[:] = 0.0

    def reset_with_noise(self, rng):
        self.reset()
        self.data.qpos[self.box_qpos_adr + 0] += rng.uniform(-0.02, 0.02)
        self.data.qpos[self.box_qpos_adr + 1] += rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:7] = self._base_qpos
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _compute_pd_torques(self):
        q = self.data.actuator_length.copy()
        qd = self.data.actuator_velocity.copy()
        tau = _KP * (self.target_pos - q) - _KD * qd
        tau += self.data.qfrc_bias[_ACTUATED_DOF_START:_ACTUATED_DOF_END]
        return np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])

    def step_frame(self):
        for _ in range(SUBSTEPS):
            np.copyto(self.data.ctrl, self._compute_pd_torques())
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[:7] = self._base_qpos
            self.data.qvel[:6] = 0.0
            self.data.qpos[self._locked_qpos_adr] = self._locked_qpos_vals
            self.data.qvel[self._locked_qvel_idx] = 0.0

    def get_palm_box_contacts(self):
        left_f = right_f = 0.0
        left_c = right_c = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.box_geom_id in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                fn = np.linalg.norm(force[:3])
                if self.left_palm_geom_id in pair:
                    left_c = True
                    left_f += fn
                if self.right_palm_geom_id in pair:
                    right_c = True
                    right_f += fn
        return {"left_contact": left_c, "right_contact": right_c,
                "left_force": left_f, "right_force": right_f}

    # IK solvers (same as PhysicsSim)
    def solve_ik_left(self, target_xyz, max_iter=500, tol=0.01,
                      step=0.02, damping=0.05):
        from physics_sim import LEFT_ARM_DOF
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

    def solve_ik_right(self, target_xyz, max_iter=500, tol=0.01,
                       step=0.02, damping=0.05):
        from physics_sim import RIGHT_ARM_DOF
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


def run_scripted_episode(sim, viewer, rng, speed=0.5):
    """Run one bimanual episode with scripted expert. Returns lift in cm."""
    sim.reset_with_noise(rng)
    box_pos = sim.box_pos.copy()
    box_init_z = box_pos[2]

    # Plan trajectory (uses IK — modifies qpos temporarily)
    plan = plan_bimanual_trajectory(sim, box_pos, rng)
    left_traj = plan["left_traj"]
    right_traj = plan["right_traj"]
    n_frames = plan["n_frames"]

    # Reset and re-apply box noise
    box_dx = box_pos[0] - BOX_POS_NOMINAL[0]
    box_dy = box_pos[1] - BOX_POS_NOMINAL[1]
    sim.reset()
    sim.data.qpos[sim.box_qpos_adr + 0] += box_dx
    sim.data.qpos[sim.box_qpos_adr + 1] += box_dy
    mujoco.mj_forward(sim.model, sim.data)
    sim.data.qpos[:7] = sim._base_qpos
    sim.data.qvel[:6] = 0.0

    frame_dt = (1.0 / CONTROL_HZ) / speed

    for t in range(n_frames):
        if not viewer.is_running():
            return 0.0

        t0 = time.perf_counter()

        sim.target_pos[LEFT_ARM_CTRL] = left_traj[t]
        sim.target_pos[RIGHT_ARM_CTRL] = right_traj[t]

        with viewer.lock():
            sim.step_frame()
        viewer.sync()

        # Status every 30 frames (~1s)
        if t % 30 == 0:
            c = sim.get_palm_box_contacts()
            lift = (sim.box_pos[2] - box_init_z) * 100
            contact_str = ""
            if c["left_contact"] or c["right_contact"]:
                contact_str = f" | L={c['left_force']:.0f}N R={c['right_force']:.0f}N"
            print(f"    frame {t:3d}/{n_frames} | box_z={sim.box_pos[2]:.3f} "
                  f"lift={lift:+.1f}cm{contact_str}")

        # Real-time pacing
        elapsed = time.perf_counter() - t0
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # Final result
    c = sim.get_palm_box_contacts()
    lift_cm = (sim.box_pos[2] - box_init_z) * 100
    both = c["left_contact"] and c["right_contact"]
    return lift_cm


def run_act_episode(model_act, sim, viewer, rng, device='cuda',
                    max_steps=250, chunk_exec=5, ensemble_k=0.01,
                    speed=0.5):
    """Run one bimanual episode with trained ACT model. Returns lift in cm."""
    sim.reset_with_noise(rng)
    box_init_z = sim.box_pos[2]

    chunk_size = model_act.chunk_size
    action_dim = model_act.action_dim

    # Temporal ensembling buffers
    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    frame_dt = (1.0 / CONTROL_HZ) / speed

    for step in range(max_steps):
        if not viewer.is_running():
            return 0.0

        t0 = time.perf_counter()

        # Re-plan every chunk_exec steps
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos_all, vel_all = sim.get_obs()
            state = np.concatenate([
                pos_all[LEFT_ARM_CTRL], pos_all[RIGHT_ARM_CTRL],
                vel_all[LEFT_ARM_CTRL], vel_all[RIGHT_ARM_CTRL],
            ])
            chunk = model_act.predict(image, state, task_id=0, device=device)
            for i in range(chunk_size):
                w = math.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        # Get ensembled action
        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        # Apply: first 7 = left arm, last 7 = right arm
        sim.target_pos[LEFT_ARM_CTRL] = action[:7]
        sim.target_pos[RIGHT_ARM_CTRL] = action[7:]

        with viewer.lock():
            sim.step_frame()
        viewer.sync()

        # Status every 30 frames
        if step % 30 == 0:
            c = sim.get_palm_box_contacts()
            lift = (sim.box_pos[2] - box_init_z) * 100
            contact_str = ""
            if c["left_contact"] or c["right_contact"]:
                contact_str = f" | L={c['left_force']:.0f}N R={c['right_force']:.0f}N"
            print(f"    step {step:3d}/{max_steps} | box_z={sim.box_pos[2]:.3f} "
                  f"lift={lift:+.1f}cm{contact_str}")

        # Real-time pacing
        elapsed = time.perf_counter() - t0
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # Final result
    c = sim.get_palm_box_contacts()
    lift_cm = (sim.box_pos[2] - box_init_z) * 100
    return lift_cm


def main():
    parser = argparse.ArgumentParser(
        description="Live bimanual box manipulation in MuJoCo viewer")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to bimanual ACT checkpoint (omit for scripted expert)")
    parser.add_argument("--loop", action="store_true",
                        help="Repeat continuously")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed (0.5 = slow-mo, 1.0 = real-time)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="Device for ACT inference (default: cuda if available)")
    args = parser.parse_args()

    use_act = args.checkpoint is not None

    # Load ACT model if checkpoint provided
    model_act = None
    device = args.device
    if use_act:
        import torch
        from act_model import ACTPolicy
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt['config']
        model_act = ACTPolicy(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            chunk_size=config['chunk_size'],
            hidden_dim=config['hidden_dim'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_tasks=config['num_tasks'],
        ).to(device)
        model_act.load_state_dict(ckpt['model_state_dict'])
        model_act.eval()
        print(f"Loaded bimanual ACT model [epoch {ckpt['epoch']}, "
              f"loss {ckpt['loss']:.6f}]")

    sim = LivePhysicsSim(need_camera=use_act)
    rng = np.random.default_rng(args.seed)

    mode_str = "ACT Policy" if use_act else "Scripted Expert"

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Bimanual Box Manipulation — {mode_str:<25s}║")
    print("║                                                         ║")
    print("║  Both hands squeeze the green box and lift it.          ║")
    print("║  No weld constraints — friction only!                   ║")
    print("║                                                         ║")
    print("║  Mouse drag: rotate | Scroll: zoom | Double-click: track║")
    print("║  Close window to exit                                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    try:
        episode = 0
        while viewer.is_running():
            episode += 1
            print(f"\n{'━'*50}")
            print(f"  Episode {episode}: pick up the green box with both hands")
            print(f"  Mode: {mode_str} | Speed: {args.speed}x | Physics: mj_step")
            print(f"{'━'*50}")

            # Brief pause to show initial state
            sim.reset_with_noise(rng)
            with viewer.lock():
                mujoco.mj_forward(sim.model, sim.data)
            viewer.sync()
            time.sleep(1.5)

            if use_act:
                lift_cm = run_act_episode(
                    model_act, sim, viewer, rng, device=device,
                    speed=args.speed)
            else:
                lift_cm = run_scripted_episode(sim, viewer, rng, speed=args.speed)

            if lift_cm >= 3.0:
                print(f"\n  Result: SUCCESS — lifted {lift_cm:.1f}cm")
            else:
                print(f"\n  Result: FAILED — lifted {lift_cm:.1f}cm")

            # Hold final pose
            time.sleep(2.5)

            if not args.loop:
                print("\n  Close the viewer window to exit.")
                while viewer.is_running():
                    time.sleep(0.1)
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if sim._renderer is not None:
            sim._renderer.close()
        viewer.close()


if __name__ == "__main__":
    main()
