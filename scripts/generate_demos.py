#!/usr/bin/env python3
"""
scripts/generate_demos.py

Scripted expert that auto-generates manipulation demos for ACT training.
Uses iterative Jacobian IK in kinematic mode (directly sets joint positions)
to produce smooth reach/grasp/pick trajectories with camera observations.

Architecture:
  1. Solve IK for waypoints (pure kinematics — no dynamics, instant convergence)
  2. Interpolate between waypoints in joint space (smooth 30 Hz trajectory)
  3. Playback through physics to render camera + allow cube dynamics
  4. Record obs (joint_pos, joint_vel, camera) + action to HDF5

Tasks generated:
  - "reach the red cube"   → arm moves above cube → descends to cube
  - "grasp the red cube"   → reach + activate weld
  - "pick up the red cube" → reach + grasp + lift

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/generate_demos.py --task reach --episodes 20
  MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20
"""

import os
import sys
import argparse
import datetime
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

try:
    import h5py
except ImportError:
    sys.exit("h5py required: pip3 install --break-system-packages h5py")


# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sim", "g1_with_camera.xml"))

CAMERA_NAME = "ego_camera"
RENDER_W, RENDER_H = 640, 480
FPS = 30
PHYSICS_HZ = 500
STEPS_PER_FRAME = PHYSICS_HZ // FPS

NUM_ACTUATORS = 29

# Right arm: 7 joints
RIGHT_ARM_CTRL = np.array([22, 23, 24, 25, 26, 27, 28])
RIGHT_ARM_DOF = np.array([28, 29, 30, 31, 32, 33, 34])

# PD gains (for playback phase — stiffer than real-time for clean tracking)
_KP = np.array([
    44, 44, 44, 70, 25, 25,
    44, 44, 44, 70, 25, 25,
    44, 25, 25,
    80, 80, 80, 80, 20, 20, 20,   # left arm (high for stability)
    80, 80, 80, 80, 20, 20, 20,   # right arm (high for clean tracking)
], dtype=np.float64)

_KD = np.array([
     4,  4,  4,  7, 2.5, 2.5,
     4,  4,  4,  7, 2.5, 2.5,
     4, 2.5, 2.5,
    8, 8, 8, 8, 2, 2, 2,
    8, 8, 8, 8, 2, 2, 2,
], dtype=np.float64)


# ────────────────────────────────────────────────────────
# Sim wrapper
# ────────────────────────────────────────────────────────

class SimWrapper:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Cache IDs
        self.hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self.weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos_adr = self.model.jnt_qposadr[self.cube_joint_id]

        # Place target site (Phase C — may not exist in older scene XMLs)
        pid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "place_site")
        self.place_site_id = pid if pid >= 0 else None

        # Arm joint qpos addresses and limits
        self.arm_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[ci, 0]]
            for ci in RIGHT_ARM_CTRL])
        self.arm_pos_lo = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 0] * 0.95
            for ci in RIGHT_ARM_CTRL])
        self.arm_pos_hi = np.array([
            self.model.jnt_range[self.model.actuator_trnid[ci, 0], 1] * 0.95
            for ci in RIGHT_ARM_CTRL])

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)
        self._ctrlrange = self.model.actuator_ctrlrange.copy()
        self._base_qpos = self.data.qpos[:7].copy()
        self.target_pos = np.zeros(NUM_ACTUATORS)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()
        self.target_pos[:] = 0.0
        self.set_weld(False)

    def reset_with_noise(self, rng: np.random.Generator):
        self.reset()
        self.data.qpos[self.cube_qpos_adr + 0] += rng.uniform(-0.03, 0.03)
        self.data.qpos[self.cube_qpos_adr + 1] += rng.uniform(-0.03, 0.03)
        mujoco.mj_forward(self.model, self.data)

    @property
    def hand_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.hand_site_id].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    @property
    def place_pos(self):
        """Position of the place target site, or None if not in scene."""
        if self.place_site_id is None:
            return None
        return self.data.site_xpos[self.place_site_id].copy()

    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    def set_weld(self, enabled: bool):
        self.data.eq_active[self.weld_id] = 1 if enabled else 0

    # ── Kinematic IK (Phase 1) ──────────────────────────

    def solve_ik(self, target_xyz: np.ndarray, max_iter: int = 500,
                 tol: float = 0.01, step: float = 0.02, damping: float = 0.05) -> bool:
        """Iterative Jacobian IK — directly sets qpos, no dynamics.
        Returns True if converged within tolerance."""
        for _ in range(max_iter):
            error = target_xyz - self.hand_pos
            if np.linalg.norm(error) < tol:
                return True
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.hand_site_id)
            J = jacp[:, RIGHT_ARM_DOF]
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            dq_n = np.linalg.norm(dq)
            if dq_n > step:
                dq *= step / dq_n
            q = self.data.qpos[self.arm_qpos_adr] + dq
            self.data.qpos[self.arm_qpos_adr] = np.clip(q, self.arm_pos_lo, self.arm_pos_hi)
            mujoco.mj_forward(self.model, self.data)
        return False

    # ── Physics playback (Phase 3) ──────────────────────

    def step_frame(self):
        """One frame: purely kinematic — set qpos + mj_forward.
        For demo generation, no dynamics needed. The arm trajectory is precise
        and the cube stays pinned at its initial position (or follows hand via weld)."""
        # Pin base
        self.data.qpos[:7] = self._base_qpos
        self.data.qvel[:6] = 0.0
        # Pin arm to target
        self.data.qpos[self.arm_qpos_adr] = self.target_pos[RIGHT_ARM_CTRL]
        self.data.qvel[RIGHT_ARM_DOF] = 0.0
        # Update kinematics (no dynamics/physics)
        mujoco.mj_forward(self.model, self.data)
        # Enforce weld constraint kinematically: move cube with hand
        if self.data.eq_active[self.weld_id]:
            hand_xyz = self.data.site_xpos[self.hand_site_id]
            self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = hand_xyz
            mujoco.mj_forward(self.model, self.data)

    def render_camera(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self.renderer.render().copy()

    def get_obs(self):
        return (
            self.data.actuator_length.copy().astype(np.float32),
            self.data.actuator_velocity.copy().astype(np.float32),
        )


# ────────────────────────────────────────────────────────
# Trajectory planning
# ────────────────────────────────────────────────────────

def interpolate_trajectory(configs: list[np.ndarray],
                           frames_per_segment: list[int]) -> np.ndarray:
    """Linearly interpolate between arm joint configs. Returns (T, 7) array."""
    assert len(frames_per_segment) == len(configs) - 1
    traj = []
    for i, n_frames in enumerate(frames_per_segment):
        q0 = configs[i]
        q1 = configs[i + 1]
        for t in range(n_frames):
            alpha = t / max(n_frames - 1, 1)
            traj.append(q0 + alpha * (q1 - q0))
    return np.array(traj)


# ────────────────────────────────────────────────────────
# Recording
# ────────────────────────────────────────────────────────

class EpisodeRecorder:
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.frames = []
        self.actions = []

    def record(self, sim: SimWrapper):
        pos, vel = sim.get_obs()
        self.positions.append(pos)
        self.velocities.append(vel)
        self.actions.append(sim.target_pos.astype(np.float32).copy())
        self.frames.append(sim.render_camera())

    def pack(self) -> dict:
        return {
            "obs/joint_positions":  np.stack(self.positions),
            "obs/joint_velocities": np.stack(self.velocities),
            "obs/camera_frames":    np.stack(self.frames),
            "action":               np.stack(self.actions),
        }


# ────────────────────────────────────────────────────────
# Expert policies
# ────────────────────────────────────────────────────────

def generate_reach(sim: SimWrapper, rng: np.random.Generator) -> dict:
    sim.reset_with_noise(rng)
    cube = sim.cube_pos.copy()

    # Plan: home → pre-reach (8cm above) → reach (just above cube)
    waypoints = [
        cube + np.array([0, 0, 0.08]),
        cube + np.array([0, 0, 0.02]),
    ]
    return _kinematic_record(sim, rng, waypoints, [40, 25], hold=20)


def generate_grasp(sim: SimWrapper, rng: np.random.Generator) -> dict:
    sim.reset_with_noise(rng)
    cube = sim.cube_pos.copy()
    waypoints = [
        cube + np.array([0, 0, 0.08]),
        cube + np.array([0, 0, 0.02]),
    ]
    return _kinematic_record(sim, rng, waypoints, [40, 25], hold=20, grasp_after=True)


def generate_pick(sim: SimWrapper, rng: np.random.Generator) -> dict:
    sim.reset_with_noise(rng)
    cube = sim.cube_pos.copy()
    waypoints = [
        cube + np.array([0, 0, 0.08]),
        cube + np.array([0, 0, 0.02]),
        cube + np.array([0, 0, 0.17]),  # lift
    ]
    return _kinematic_record(sim, rng, waypoints, [40, 25, 35],
                              hold=20, grasp_after_wp=1)


def generate_place(sim: SimWrapper, rng: np.random.Generator) -> dict:
    """Pick up cube, move to place target, lower and release."""
    sim.reset_with_noise(rng)
    cube = sim.cube_pos.copy()
    place = sim.place_pos.copy()
    # Waypoints: approach → descend → [GRASP] → lift → above target → lower → [RELEASE]
    lift_z = max(cube[2], place[2]) + 0.13  # clearance above both positions
    waypoints = [
        cube + np.array([0, 0, 0.08]),       # above cube
        cube + np.array([0, 0, 0.02]),       # at cube (grasp here)
        np.array([cube[0], cube[1], lift_z]),  # lift
        np.array([place[0], place[1], lift_z]),  # above target
        place + np.array([0, 0, 0.02]),      # lower to target (release here)
    ]
    return _kinematic_record(sim, rng, waypoints, [35, 25, 25, 30, 25],
                              hold=20, grasp_after_wp=1, release_after_wp=4)


def _kinematic_record(sim: SimWrapper, rng: np.random.Generator,
                      waypoints: list, frames_per: list, hold: int = 15,
                      grasp_after: bool = False, grasp_after_wp: int = -1,
                      release_after_wp: int = -1) -> dict:
    """
    Unified generation: solve IK for waypoints, interpolate, playback with physics.

    Uses a two-pass approach:
    Pass 1: Kinematic IK to find joint configs for each waypoint
    Pass 2: Reset sim, replay interpolated trajectory through physics + record

    grasp_after: activate weld after all waypoints (for grasp task)
    grasp_after_wp: activate weld after waypoint index N (for pick task)
    release_after_wp: deactivate weld after waypoint index N (for place task)
    """
    # Save cube noise state
    cube_dx = sim.data.qpos[sim.cube_qpos_adr + 0] - sim.model.qpos0[sim.cube_qpos_adr + 0]
    cube_dy = sim.data.qpos[sim.cube_qpos_adr + 1] - sim.model.qpos0[sim.cube_qpos_adr + 1]

    # Pass 1: IK solve (pure kinematics — no physics)
    configs = [sim.arm_q.copy()]
    for wp in waypoints:
        sim.solve_ik(wp)
        configs.append(sim.arm_q.copy())

    # Interpolate joint trajectory
    traj = interpolate_trajectory(configs, frames_per)  # (T, 7)

    # Pass 2: Reset and playback (pure kinematics)
    sim.reset()
    sim.data.qpos[sim.cube_qpos_adr + 0] += cube_dx
    sim.data.qpos[sim.cube_qpos_adr + 1] += cube_dy
    mujoco.mj_forward(sim.model, sim.data)

    rec = EpisodeRecorder()

    # Compute frame indices where each waypoint is reached
    wp_frame_idx = []
    total = 0
    for n in frames_per:
        total += n
        wp_frame_idx.append(total)

    grasp_done = False

    for t in range(len(traj)):
        sim.target_pos[RIGHT_ARM_CTRL] = traj[t]
        rec.record(sim)
        sim.step_frame()

        # Check if we should activate grasp
        if not grasp_done and grasp_after_wp >= 0 and t >= wp_frame_idx[grasp_after_wp] - 1:
            sim.set_weld(True)
            grasp_done = True

        # Check if we should release
        if grasp_done and release_after_wp >= 0 and t >= wp_frame_idx[release_after_wp] - 1:
            sim.set_weld(False)
            grasp_done = False

    # Grasp after all waypoints
    if grasp_after and not grasp_done:
        sim.set_weld(True)

    # Hold at final position
    for _ in range(hold):
        rec.record(sim)
        sim.step_frame()

    return rec.pack()


# ────────────────────────────────────────────────────────
# Save episode to HDF5
# ────────────────────────────────────────────────────────

def save_episode(ep_data: dict, episode_id: int, task: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode_id:04d}.hdf5"
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.create_dataset("joint_positions", data=ep_data["obs/joint_positions"],
                           compression="gzip")
        obs.create_dataset("joint_velocities", data=ep_data["obs/joint_velocities"],
                           compression="gzip")
        obs.create_dataset("camera_frames", data=ep_data["obs/camera_frames"],
                           compression="gzip")
        f.create_dataset("action", data=ep_data["action"], compression="gzip")
        f.attrs["episode_id"] = episode_id
        f.attrs["task_description"] = task
        f.attrs["fps"] = FPS
        f.attrs["timestamp"] = datetime.datetime.now().isoformat()
        f.attrs["num_frames"] = len(ep_data["action"])
    return path


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

TASK_MAP = {
    "reach": ("reach the red cube", generate_reach),
    "grasp": ("grasp the red cube", generate_grasp),
    "pick":  ("pick up the red cube", generate_pick),
    "place": ("place the red cube on the blue plate", generate_place),
}


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate manipulation demos via scripted expert")
    parser.add_argument("--task", choices=list(TASK_MAP), default="reach")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="data/demos")
    parser.add_argument("--start-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)
    sim = SimWrapper()

    tasks = list(TASK_MAP) if args.all_tasks else [args.task]

    if args.start_id >= 0:
        ep_id = args.start_id
    else:
        import glob
        existing = sorted(glob.glob(str(output_dir / "episode_*.hdf5")))
        ep_id = 0 if not existing else int(Path(existing[-1]).stem.split("_")[1]) + 1

    total = args.episodes * len(tasks)
    done = 0
    converged = 0

    for task_key in tasks:
        task_label, gen_fn = TASK_MAP[task_key]
        print(f"\n{'='*60}")
        print(f"Generating {args.episodes} episodes: '{task_label}'")
        print(f"{'='*60}")

        for i in range(args.episodes):
            t0 = time.time()
            ep_data = gen_fn(sim, rng)
            n_frames = len(ep_data["action"])
            path = save_episode(ep_data, ep_id, task_label, output_dir)
            elapsed = time.time() - t0
            done += 1

            final_hand = sim.hand_pos
            final_cube = sim.cube_pos
            final_dist = np.linalg.norm(final_hand - final_cube)
            ok = "OK" if final_dist < 0.06 else "FAR"
            if final_dist < 0.06:
                converged += 1

            print(f"  [{done}/{total}] ep_{ep_id:04d} — {n_frames} frames"
                  f" ({n_frames/FPS:.1f}s) — d={final_dist:.3f} [{ok}]"
                  f" — {elapsed:.1f}s -> {path.name}")
            ep_id += 1

    rate = converged / done * 100 if done else 0
    print(f"\nDone. {done} episodes saved to {output_dir.absolute()}")
    print(f"Convergence: {converged}/{done} ({rate:.0f}%)")
    print(f"\nNext steps:")
    print(f"  python3 scripts/convert_to_lerobot.py --demos {output_dir}")
    print(f"  python3 scripts/train_act.py")


if __name__ == "__main__":
    main()
