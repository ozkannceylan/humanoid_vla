"""
demo_recorder.py
Records teleoperation demonstrations to HDF5 files for ACT/LeRobot training.

Each episode contains synchronised (observation, action) pairs at ~30 Hz.
  obs/joint_positions   (T, 29)       float32  — from /joint_states
  obs/joint_velocities  (T, 29)       float32  — from /joint_states
  obs/camera_frames     (T, H, W, 3)  uint8    — from /camera/image_raw
  action                (T, 29)       float32  — from /joint_commands
  attrs: episode_id, task_description, fps, timestamp

Output: data/demos/episode_NNNN.hdf5 relative to the working directory.

Controls (keyboard in this terminal):
  Enter   — start recording a new episode
  Esc     — stop recording and SAVE the episode
  d       — stop recording and DISCARD the episode
  p       — print current episode stats
  Ctrl+C  — quit

Typical workflow:
  1. Start bridge_node (fixed_base + gravity_comp) in terminal 1
  2. Start arm_teleop_node in terminal 2
  3. Start demo_recorder in terminal 3 — press Enter to start, Esc to save
  4. Repeat for 20-50 episodes per task
  5. Run scripts/convert_to_lerobot.py to convert to LeRobot format

Run: ros2 run vla_mujoco_bridge demo_recorder [--ros-args -p task:='reach red cube']
"""

import os
import time
import threading
import datetime
import glob
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
# cv_bridge removed — manual conversion used instead (avoids NumPy ABI conflict)

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from pynput import keyboard as kb
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False

OUTPUT_DIR = Path("data/demos")
RECORD_HZ = 30
DT = 1.0 / RECORD_HZ


class DemoRecorderNode(Node):

    def __init__(self):
        super().__init__("demo_recorder_node")
        self.declare_parameter("task", "reach red cube")
        self._task = self.get_parameter("task").value

        cbg = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # Latest ROS2 data (updated async)
        self._latest_joints: np.ndarray | None = None
        self._latest_vels: np.ndarray | None = None
        self._latest_action: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None

        self.create_subscription(JointState, "/joint_states", self._on_joints, 10, callback_group=cbg)
        self.create_subscription(JointState, "/joint_commands", self._on_cmd, 10, callback_group=cbg)
        self.create_subscription(Image, "/camera/image_raw", self._on_image, 1, callback_group=cbg)

        # Recording state
        self._recording = False
        self._buf_pos: list[np.ndarray] = []
        self._buf_vel: list[np.ndarray] = []
        self._buf_act: list[np.ndarray] = []
        self._buf_img: list[np.ndarray] = []
        self._ep_start: float = 0.0

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._episode_id = self._next_episode_id()

        if not _HAS_H5PY:
            self.get_logger().error("h5py not installed — run: pip3 install --break-system-packages h5py")
        if not _HAS_PYNPUT:
            self.get_logger().error("pynput not installed — run: pip3 install --break-system-packages pynput")

        self.get_logger().info(
            f"\nDemo recorder ready — task: '{self._task}'\n"
            f"  Output dir : {OUTPUT_DIR.absolute()}\n"
            f"  Next episode: {self._episode_id:04d}\n"
            "\nControls (keyboard in THIS terminal):\n"
            "  Enter → start  |  Esc → save  |  d → discard  |  p → stats  |  Ctrl+C → quit"
        )

        # Recording timer — samples at record_hz when recording is active
        self._record_timer = self.create_timer(DT, self._record_tick, callback_group=cbg)

    # ------------------------------------------------------------------
    # ROS2 subscribers
    # ------------------------------------------------------------------

    def _on_joints(self, msg: JointState):
        with self._lock:
            self._latest_joints = np.array(msg.position, dtype=np.float32)
            self._latest_vels = np.array(msg.velocity, dtype=np.float32) if msg.velocity else np.zeros(29, np.float32)

    def _on_cmd(self, msg: JointState):
        with self._lock:
            if msg.position:
                self._latest_action = np.array(msg.position, dtype=np.float32)

    def _on_image(self, msg: Image):
        try:
            # Manual Image→numpy — avoids cv_bridge / NumPy ABI issues
            frame = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            if msg.encoding == "bgr8":
                frame = frame[:, :, ::-1].copy()  # BGR→RGB
        except Exception:
            return
        with self._lock:
            self._latest_frame = frame

    # ------------------------------------------------------------------
    # Recording logic
    # ------------------------------------------------------------------

    def _record_tick(self):
        if not self._recording:
            return
        with self._lock:
            pos = self._latest_joints
            vel = self._latest_vels
            act = self._latest_action
            img = self._latest_frame
        if pos is None or img is None:
            return
        act_out = act if act is not None else np.zeros(29, np.float32)
        self._buf_pos.append(pos.copy())
        self._buf_vel.append(vel.copy() if vel is not None else np.zeros_like(pos))
        self._buf_act.append(act_out.copy())
        self._buf_img.append(img.copy())

    def start_episode(self):
        if self._recording:
            self.get_logger().warn("Already recording.")
            return
        self._buf_pos.clear()
        self._buf_vel.clear()
        self._buf_act.clear()
        self._buf_img.clear()
        self._recording = True
        self._ep_start = time.time()
        self.get_logger().info(f"Recording episode {self._episode_id:04d} ... (Esc=save  d=discard)")

    def stop_episode(self, save: bool):
        if not self._recording:
            self.get_logger().warn("Not currently recording.")
            return
        self._recording = False
        n = len(self._buf_pos)
        duration = time.time() - self._ep_start
        if not save:
            self.get_logger().info(f"Episode {self._episode_id:04d} discarded ({n} frames, {duration:.1f}s)")
            return
        if n < 5:
            self.get_logger().warn(f"Episode too short ({n} frames) — discarding.")
            return
        self._save_episode()

    def _save_episode(self):
        if not _HAS_H5PY:
            self.get_logger().error("Cannot save — h5py not installed.")
            return
        path = OUTPUT_DIR / f"episode_{self._episode_id:04d}.hdf5"
        pos_arr = np.stack(self._buf_pos, axis=0)   # (T, 29)
        vel_arr = np.stack(self._buf_vel, axis=0)   # (T, 29)
        act_arr = np.stack(self._buf_act, axis=0)   # (T, 29)
        img_arr = np.stack(self._buf_img, axis=0)   # (T, H, W, 3)

        with h5py.File(path, "w") as f:
            obs = f.create_group("obs")
            obs.create_dataset("joint_positions", data=pos_arr, compression="gzip")
            obs.create_dataset("joint_velocities", data=vel_arr, compression="gzip")
            obs.create_dataset("camera_frames", data=img_arr, compression="gzip")
            f.create_dataset("action", data=act_arr, compression="gzip")
            f.attrs["episode_id"] = self._episode_id
            f.attrs["task_description"] = self._task
            f.attrs["fps"] = RECORD_HZ
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()
            f.attrs["num_frames"] = len(self._buf_pos)

        self.get_logger().info(
            f"Saved episode {self._episode_id:04d} → {path}  "
            f"({len(self._buf_pos)} frames, {img_arr.shape[1]}x{img_arr.shape[2]})"
        )
        self._episode_id = self._next_episode_id()

    def print_stats(self):
        n = len(self._buf_pos)
        status = "RECORDING" if self._recording else "idle"
        self.get_logger().info(
            f"[{status}] episode {self._episode_id:04d} — {n} frames buffered "
            f"({n/RECORD_HZ:.1f}s) — task: '{self._task}'"
        )

    @staticmethod
    def _next_episode_id() -> int:
        existing = sorted(glob.glob(str(OUTPUT_DIR / "episode_*.hdf5")))
        if not existing:
            return 0
        last = int(Path(existing[-1]).stem.split("_")[1])
        return last + 1


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorderNode()

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    if _HAS_PYNPUT:
        stop_event = threading.Event()

        def on_press(key):
            try:
                char = key.char
                if char == "d":
                    node.stop_episode(save=False)
                elif char == "p":
                    node.print_stats()
            except AttributeError:
                if key == kb.Key.enter:
                    node.start_episode()
                elif key == kb.Key.esc:
                    node.stop_episode(save=True)
                elif key == kb.Key.ctrl_l or key == kb.Key.ctrl_r:
                    stop_event.set()

        listener = kb.Listener(on_press=on_press)
        listener.start()
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
    else:
        # Fallback: line-based stdin control
        node.get_logger().info("pynput not available — using stdin (Enter/d/p/quit)")
        try:
            while rclpy.ok():
                cmd = input()
                if cmd == "":
                    node.start_episode()
                elif cmd == "d":
                    node.stop_episode(save=False)
                elif cmd == "p":
                    node.print_stats()
                elif cmd in ("q", "quit"):
                    break
                else:
                    node.stop_episode(save=True)
        except (KeyboardInterrupt, EOFError):
            pass

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
