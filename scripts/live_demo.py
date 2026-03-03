#!/usr/bin/env python3
"""
scripts/live_demo.py

Run the trained ACT policy with MuJoCo's interactive viewer open.
You can rotate, zoom, and watch the robot perform tasks in real-time.

Controls:
  - Mouse drag: rotate camera
  - Scroll: zoom
  - Double-click body: track it
  - Space: pause/unpause (in some MuJoCo versions)

Usage:
  cd ~/projects/humanoid_vla
  python3 scripts/live_demo.py --checkpoint data/checkpoints/best.pt
  python3 scripts/live_demo.py --checkpoint data/checkpoints/best.pt --task reach
  python3 scripts/live_demo.py --checkpoint data/checkpoints/best.pt --task place --loop
"""

import argparse
import math
import os
import sys
import time

# Do NOT set MUJOCO_GL=egl — we need GLFW for the interactive viewer
# Remove EGL if it was set
os.environ.pop("MUJOCO_GL", None)

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy, TASK_LABELS, task_to_id
from generate_demos import SimWrapper, RIGHT_ARM_CTRL, CAMERA_NAME, MODEL_PATH


TASKS = [
    "reach the red cube",
    "grasp the red cube",
    "pick up the red cube",
    "place the red cube on the blue plate",
]

TASK_SHORT = {
    "reach": "reach the red cube",
    "grasp": "grasp the red cube",
    "pick": "pick up the red cube",
    "place": "place the red cube on the blue plate",
}


class LiveSim:
    """Sim wrapper that uses the interactive viewer instead of offscreen-only."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Cache IDs (same as SimWrapper)
        self.hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self.weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_adr = self.model.jnt_qposadr[self.cube_joint_id]

        pid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "place_site")
        self.place_site_id = pid if pid >= 0 else None

        # Arm joint info
        self.arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in RIGHT_ARM_CTRL])
        self.arm_pos_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in RIGHT_ARM_CTRL])
        self.arm_pos_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in RIGHT_ARM_CTRL])

        self._base_qpos = self.data.qpos[:7].copy()
        self.target_pos = np.zeros(29)

        # Offscreen renderer for the ACT model's camera input
        # (the model needs the ego camera image even when we're using the viewer)
        self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    @property
    def hand_pos(self):
        return self.data.site_xpos[self.hand_site_id].copy()

    @property
    def cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    @property
    def place_pos(self):
        if self.place_site_id is None:
            return None
        return self.data.site_xpos[self.place_site_id].copy()

    def set_weld(self, enabled):
        self.data.eq_active[self.weld_id] = 1 if enabled else 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()
        self.target_pos[:] = 0.0
        self.set_weld(False)

    def reset_with_noise(self, rng):
        self.reset()
        self.data.qpos[self.cube_qpos_adr + 0] += rng.uniform(-0.03, 0.03)
        self.data.qpos[self.cube_qpos_adr + 1] += rng.uniform(-0.03, 0.03)
        mujoco.mj_forward(self.model, self.data)

    def step_frame(self):
        self.data.qpos[:7] = self._base_qpos
        self.data.qvel[:6] = 0.0
        self.data.qpos[self.arm_qpos_adr] = self.target_pos[RIGHT_ARM_CTRL]
        self.data.qvel[28:35] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.data.eq_active[self.weld_id]:
            hand_xyz = self.data.site_xpos[self.hand_site_id]
            self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = hand_xyz
            mujoco.mj_forward(self.model, self.data)

    def render_camera(self):
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self._renderer.render().copy()

    def get_obs(self):
        return (
            self.data.actuator_length.copy().astype(np.float32),
            self.data.actuator_velocity.copy().astype(np.float32),
        )


def run_live_episode(model_act, sim, viewer, task_label, rng, device='cuda',
                     max_steps=250, chunk_exec=5, ensemble_k=0.01,
                     auto_grasp_dist=0.04, auto_release_delay=30,
                     real_time_factor=1.0):
    """Run one episode with live viewer. Returns success bool."""
    sim.reset_with_noise(rng)
    chunk_size = model_act.chunk_size
    action_dim = model_act.action_dim

    is_composite = ("pick" in task_label or "place" in task_label)
    approach_task_id = task_to_id("grasp the red cube") if is_composite else None
    final_task_id = task_to_id(task_label)
    active_task_id = approach_task_id if is_composite else final_task_id

    grasped = False
    grasp_step = -1
    released = False
    is_place = ("place" in task_label)
    place_target = sim.place_pos if is_place else None

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    frame_dt = 1.0 / 30.0 / real_time_factor  # ~33ms per frame at 1x speed

    for step in range(max_steps):
        if not viewer.is_running():
            return False

        t0 = time.perf_counter()

        # Re-plan
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos, vel = sim.get_obs()
            state = np.concatenate([pos, vel])
            chunk = model_act.predict(image, state, active_task_id, device=device)
            for i in range(chunk_size):
                w = math.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        sim.target_pos[RIGHT_ARM_CTRL] = action[RIGHT_ARM_CTRL]

        with viewer.lock():
            sim.step_frame()
        viewer.sync()

        # Auto-grasp
        if not grasped and not released:
            d = np.linalg.norm(sim.hand_pos - sim.cube_pos)
            if d < auto_grasp_dist:
                sim.set_weld(True)
                grasped = True
                grasp_step = step
                if is_composite:
                    active_task_id = final_task_id
                    action_sum[step+1:] = 0
                    weight_sum[step+1:] = 0

        # Auto-release
        if is_place and grasped and place_target is not None:
            steps_grasped = step - grasp_step
            cube_near = np.linalg.norm(sim.cube_pos[:2] - place_target[:2]) < 0.06
            hand_low = sim.hand_pos[2] < place_target[2] + 0.12
            if steps_grasped > auto_release_delay and cube_near and hand_low:
                with viewer.lock():
                    sim.set_weld(False)
                    sim.data.qpos[sim.cube_qpos_adr + 2] = 0.825
                    mujoco.mj_forward(sim.model, sim.data)
                    sim.set_weld(False)
                grasped = False
                released = True
                viewer.sync()

        # Real-time pacing
        elapsed = time.perf_counter() - t0
        sleep = frame_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

        # Status print every 50 steps
        if step % 50 == 0:
            d_hand_cube = np.linalg.norm(sim.hand_pos - sim.cube_pos)
            state_str = "GRASPED" if grasped else ("RELEASED" if released else "approaching")
            print(f"    step {step:3d} | hand-cube: {d_hand_cube:.3f}m | {state_str}")

    # Determine success
    from evaluate import SUCCESS_FN
    check = SUCCESS_FN.get(task_label)
    return check(sim, grasped) if check else False


def main():
    parser = argparse.ArgumentParser(description="Live ACT demo in MuJoCo viewer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=None,
                        help="Task to run: reach/grasp/pick/place (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loop", action="store_true",
                        help="Loop continuously through tasks")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed (0.5 = half speed, 1.0 = real-time)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['config']
    model_act = ACTPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_tasks=config['num_tasks'],
    ).to(args.device)
    model_act.load_state_dict(ckpt['model_state_dict'])
    model_act.eval()
    print(f"Loaded model [epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}]")

    # Choose tasks
    if args.task:
        task_label = TASK_SHORT.get(args.task, args.task)
        tasks = [task_label]
    else:
        tasks = TASKS

    # Create sim + viewer
    sim = LiveSim()
    rng = np.random.default_rng(args.seed)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  MuJoCo Live Demo — ACT Policy                         ║")
    print("║  Mouse drag: rotate | Scroll: zoom | Double-click: track║")
    print("║  Close window to exit                                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    try:
        while viewer.is_running():
            for task_label in tasks:
                if not viewer.is_running():
                    break

                print(f"\n{'━'*50}")
                print(f"  Task: {task_label}")
                print(f"{'━'*50}")

                # Brief pause to show initial state
                sim.reset_with_noise(rng)
                with viewer.lock():
                    mujoco.mj_forward(sim.model, sim.data)
                viewer.sync()
                time.sleep(1.5)

                success = run_live_episode(
                    model_act, sim, viewer, task_label, rng,
                    device=args.device, real_time_factor=args.speed,
                )

                result = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"\n  Result: {result}")

                # Hold final pose
                time.sleep(2.0)

            if not args.loop:
                print("\n  All tasks complete. Close the viewer window to exit.")
                while viewer.is_running():
                    time.sleep(0.1)
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sim._renderer.close()
        viewer.close()


if __name__ == "__main__":
    main()
