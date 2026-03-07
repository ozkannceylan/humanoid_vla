"""
mujoco_sim.py
Owns all MuJoCo state for the Unitree G1 simulation.

Threading model:
  - Physics loop runs on main thread (run_physics_loop blocks)
  - ROS2 executor runs on background threads
  - All shared state access is protected by self.lock

PD Controller:
  G1 uses torque actuators (<motor> type). To follow position targets we
  implement τ = Kp*(q_des - q) - Kd*q_dot + qfrc_bias[6:35] (gravity comp),
  clipped to ctrlrange.

Gravity compensation:
  data.qfrc_bias[6:35] contains the gravity + Coriolis forces for the 29
  actuated joints (DOF indices 6-34; DOFs 0-5 are the floating base freejoint).
  Adding this to the PD torques makes the robot hold its pose against gravity.

Fixed base mode:
  When fixed_base=True, the pelvis (qpos[0:7]) is kinematically frozen each
  step at the initial standing height. This is the standard approach for arm
  manipulation demos when locomotion is not yet implemented.
"""

import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "sim", "g1_with_camera.xml"
)
CAMERA_NAME = "ego_camera"
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# Indices into qfrc_bias for the 29 actuated joints.
# G1 freejoint (floating_base_joint) occupies DOFs 0-5; actuated joints are 6-34.
_ACTUATED_DOF_START = 6
_ACTUATED_DOF_END = 35  # exclusive

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

    Parameters
    ----------
    model_path   : path to the MJCF scene XML (default: g1_with_camera.xml)
    gravity_comp : add qfrc_bias[6:35] to PD torques — robot holds pose (default: True)
    fixed_base   : kinematically freeze pelvis at initial height — for arm demos (default: False)
    render_hz    : camera render rate
    physics_hz   : physics step rate

    External API (thread-safe, called from ROS2 threads):
      get_joint_state()      -> (positions, velocities, names)
      get_latest_frame()     -> RGB numpy array or None
      set_joint_command(pos) -> set target joint positions
      stop()                 -> signal physics loop to exit
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        gravity_comp: bool = True,
        fixed_base: bool = False,
        render_hz: float = 30.0,
        physics_hz: float = 500.0,
    ):
        self.gravity_comp = gravity_comp
        self.fixed_base = fixed_base
        self.render_hz = render_hz
        self.physics_hz = physics_hz
        self.physics_dt = 1.0 / physics_hz
        self.render_interval = 1.0 / render_hz

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Store initial pelvis pose for fixed_base freeze
        mujoco.mj_forward(self.model, self.data)
        self._base_qpos = self.data.qpos[:7].copy()  # xyz + quaternion

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

    def get_site_xpos(self, site_name: str) -> np.ndarray | None:
        """Return Cartesian position [x,y,z] of a named site (thread-safe)."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            return None
        with self.lock:
            return self.data.site_xpos[sid].copy()

    def get_body_xpos(self, body_name: str) -> np.ndarray | None:
        """Return Cartesian position [x,y,z] of a named body (thread-safe)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None
        with self.lock:
            return self.data.xpos[bid].copy()

    def get_site_jacp(self, site_name: str) -> np.ndarray | None:
        """Return positional Jacobian (3, nv) for a named site (thread-safe)."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            return None
        jacp = np.zeros((3, self.model.nv))
        with self.lock:
            mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
        return jacp

    def set_grasp(self, enabled: bool) -> bool:
        """Enable/disable the grasp weld constraint. Returns True if found."""
        eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        if eid < 0:
            return False
        with self.lock:
            # MuJoCo 3.x: eq_active0 (initial), but runtime toggle uses eq_active on data
            self.model.eq_active0[eid] = 1 if enabled else 0
            # Also force it in data for immediate effect
            self.data.eq_active[eid] = 1 if enabled else 0
        return True

    # ------------------------------------------------------------------
    # Internal: PD controller
    # ------------------------------------------------------------------

    def _compute_pd_torques(self) -> np.ndarray:
        """τ = Kp*(q_des - q) - Kd*q_dot [+ qfrc_bias], clipped to ctrlrange."""
        q = self.data.actuator_length.copy()
        qd = self.data.actuator_velocity.copy()
        tau = _KP * (self._target_pos - q) - _KD * qd
        if self.gravity_comp:
            # qfrc_bias[6:35] = gravity + Coriolis for the 29 actuated joints
            tau += self.data.qfrc_bias[_ACTUATED_DOF_START:_ACTUATED_DOF_END]
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

                    # Fixed-base: freeze pelvis at initial standing pose
                    if self.fixed_base:
                        self.data.qpos[:7] = self._base_qpos
                        self.data.qvel[:6] = 0.0
                        mujoco.mj_forward(self.model, self.data)

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
