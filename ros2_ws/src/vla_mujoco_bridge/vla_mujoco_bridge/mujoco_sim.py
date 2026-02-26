"""
mujoco_sim.py
Owns all MuJoCo state for the Unitree G1 simulation.

Threading model:
  - Physics loop runs on main thread (run_physics_loop blocks)
  - ROS2 executor runs on background threads
  - All shared state access is protected by self.lock

PD Controller:
  G1 uses torque actuators (<motor> type). To follow position targets we
  implement τ = Kp*(q_des - q) - Kd*q_dot, clipped to ctrlrange.
  Gains are tuned per joint group based on torque limits.
"""

import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

os.environ.setdefault("MUJOCO_GL", "egl")

MODEL_PATH = "/media/orka/storage/robotics/sim/g1_with_camera.xml"
CAMERA_NAME = "ego_camera"
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# PD gains indexed by actuator index (29 joints)
# Tuned as: Kp ~ 0.5 * ctrlrange_max, Kd ~ 0.1 * Kp
# fmt: off
_KP = np.array([
    44, 44, 44, 70, 25, 25,   # left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    44, 44, 44, 70, 25, 25,   # right leg: same order
    44, 25, 25,               # waist:     yaw, roll, pitch
    12, 12, 12, 12,  3,  3,  3,  # left arm:  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    12, 12, 12, 12,  3,  3,  3,  # right arm: same order
], dtype=np.float64)

_KD = np.array([
     4,  4,  4,  7, 2.5, 2.5,
     4,  4,  4,  7, 2.5, 2.5,
     4, 2.5, 2.5,
    1.2, 1.2, 1.2, 1.2, 0.3, 0.3, 0.3,
    1.2, 1.2, 1.2, 1.2, 0.3, 0.3, 0.3,
], dtype=np.float64)
# fmt: on


class MujocoSim:
    """
    Encapsulates MuJoCo model, data, renderer, and physics loop.

    External API (thread-safe, called from ROS2 threads):
      get_joint_state()      -> (positions, velocities, names)
      get_latest_frame()     -> RGB numpy array or None
      set_joint_command(pos) -> set target joint positions
      stop()                 -> signal physics loop to exit
    """

    def __init__(self, render_hz: float = 30.0, physics_hz: float = 500.0):
        self.render_hz = render_hz
        self.physics_hz = physics_hz
        self.physics_dt = 1.0 / physics_hz
        self.render_interval = 1.0 / render_hz

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Validate camera
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )
        if cam_id < 0:
            raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in model")
        self.cam_id = cam_id

        # Actuator names in ctrl[] order
        self.actuator_names = [
            self.model.actuator(i).name for i in range(self.model.nu)
        ]

        # ctrlrange for PD output clipping
        self._ctrlrange = self.model.actuator_ctrlrange.copy()  # (nu, 2)

        # Thread safety
        self.lock = threading.Lock()
        self._running = False

        # Shared state updated by physics loop, read by ROS2 publishers
        self.latest_frame: np.ndarray | None = None
        self.latest_joint_pos = np.zeros(self.model.nu)
        self.latest_joint_vel = np.zeros(self.model.nu)

        # Position target set by ROS2 subscriber / teleop
        self._target_pos = np.zeros(self.model.nu)
        self._cmd_dirty = False

        # Renderer is created inside run_physics_loop() to bind EGL to that thread
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

    # ------------------------------------------------------------------
    # Thread-safe external API
    # ------------------------------------------------------------------

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with self.lock:
            return (
                self.latest_joint_pos.copy(),
                self.latest_joint_vel.copy(),
                list(self.actuator_names),
            )

    def get_latest_frame(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def set_joint_command(self, positions: np.ndarray) -> None:
        """Set desired joint positions (radians). PD controller applies torques."""
        with self.lock:
            n = min(len(positions), self.model.nu)
            self._target_pos[:n] = positions[:n]
            self._cmd_dirty = True

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal: PD controller
    # ------------------------------------------------------------------

    def _compute_pd_torques(self) -> np.ndarray:
        """τ = Kp*(q_des - q) - Kd*q_dot, clipped to ctrlrange."""
        q = self.data.actuator_length.copy()
        qd = self.data.actuator_velocity.copy()
        tau = _KP * (self._target_pos - q) - _KD * qd
        # Clip each joint to its torque limit
        tau = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        return tau

    # ------------------------------------------------------------------
    # Physics loop (blocks calling thread until viewer closes or stop())
    # ------------------------------------------------------------------

    def run_physics_loop(self, launch_viewer: bool = True) -> None:
        """
        Run physics at physics_hz, render camera at render_hz, sync viewer.
        MUST be called from the thread that should own the EGL context.
        """
        # Create EGL renderer on this thread
        self._renderer = mujoco.Renderer(
            self.model, height=RENDER_HEIGHT, width=RENDER_WIDTH
        )

        if launch_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self._running = True
        last_render = 0.0

        try:
            while self._running:
                if launch_viewer and not self._viewer.is_running():
                    break

                t0 = time.perf_counter()

                with self.lock:
                    # Apply PD controller
                    np.copyto(self.data.ctrl, self._compute_pd_torques())

                    # Physics step
                    mujoco.mj_step(self.model, self.data)

                    # Snapshot joint state
                    np.copyto(self.latest_joint_pos, self.data.actuator_length)
                    np.copyto(self.latest_joint_vel, self.data.actuator_velocity)

                # Camera render (outside lock — renderer snapshot is independent)
                now = time.perf_counter()
                if now - last_render >= self.render_interval:
                    self._renderer.update_scene(self.data, camera=CAMERA_NAME)
                    frame = self._renderer.render()
                    with self.lock:
                        self.latest_frame = frame
                    last_render = now

                if launch_viewer:
                    self._viewer.sync()

                # Maintain timestep
                sleep_t = self.physics_dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            self._renderer.close()
            if launch_viewer and self._viewer is not None:
                self._viewer.close()
