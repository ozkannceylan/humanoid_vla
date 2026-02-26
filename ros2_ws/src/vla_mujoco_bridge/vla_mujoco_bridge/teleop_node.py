"""
teleop_node.py
Keyboard teleoperation: key presses → position targets → /joint_commands.

Key bindings (joint position targets, radians):
  w / s   — Left hip pitch     +/-
  a / d   — Left hip roll      +/-
  q / e   — Waist yaw          +/-
  i / k   — Right shoulder pitch +/-
  j / l   — Right shoulder yaw   +/-
  r       — Reset all joints to zero
  ESC     — Quit

Run: ros2 run vla_mujoco_bridge teleop_node
NOTE: Run in a graphical terminal (requires X display for pynput).
"""

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pynput import keyboard

# -----------------------------------------------------------------------
# Actuator order (must match ctrl[] in g1_29dof.xml)
# Verified against model actuator list printed by test_g1.py
# -----------------------------------------------------------------------
JOINT_NAMES = [
    "left_hip_pitch",      # 0
    "left_hip_roll",       # 1
    "left_hip_yaw",        # 2
    "left_knee",           # 3
    "left_ankle_pitch",    # 4
    "left_ankle_roll",     # 5
    "right_hip_pitch",     # 6
    "right_hip_roll",      # 7
    "right_hip_yaw",       # 8
    "right_knee",          # 9
    "right_ankle_pitch",   # 10
    "right_ankle_roll",    # 11
    "waist_yaw",           # 12
    "waist_roll",          # 13
    "waist_pitch",         # 14
    "left_shoulder_pitch", # 15
    "left_shoulder_roll",  # 16
    "left_shoulder_yaw",   # 17
    "left_elbow",          # 18
    "left_wrist_roll",     # 19
    "left_wrist_pitch",    # 20
    "left_wrist_yaw",      # 21
    "right_shoulder_pitch",# 22
    "right_shoulder_roll", # 23
    "right_shoulder_yaw",  # 24
    "right_elbow",         # 25
    "right_wrist_roll",    # 26
    "right_wrist_pitch",   # 27
    "right_wrist_yaw",     # 28
]

DELTA = 0.05  # radians per keypress

# key char → (joint_index, delta)
KEY_MAP = {
    "w": (0,  +DELTA),   # left_hip_pitch+
    "s": (0,  -DELTA),   # left_hip_pitch-
    "a": (1,  +DELTA),   # left_hip_roll+
    "d": (1,  -DELTA),   # left_hip_roll-
    "q": (12, +DELTA),   # waist_yaw+
    "e": (12, -DELTA),   # waist_yaw-
    "i": (22, +DELTA),   # right_shoulder_pitch+
    "k": (22, -DELTA),   # right_shoulder_pitch-
    "j": (24, +DELTA),   # right_shoulder_yaw+
    "l": (24, -DELTA),   # right_shoulder_yaw-
}

# Conservative joint position limits (radians)
LIMITS: dict[int, tuple[float, float]] = {
    0:  (-1.5,  1.5),   # hip pitch
    1:  (-0.5,  0.5),   # hip roll
    12: (-2.0,  2.0),   # waist yaw
    22: (-3.0,  1.0),   # right shoulder pitch
    24: (-2.0,  2.0),   # right shoulder yaw
}


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")
        self._pub = self.create_publisher(JointState, "/joint_commands", 10)
        self._lock = threading.Lock()
        self._positions = np.zeros(len(JOINT_NAMES))

        # Publish target positions at 20 Hz
        self.create_timer(0.05, self._publish)

        # Start keyboard listener (runs on its own thread)
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

        self._print_help()

    def _print_help(self):
        print("\n--- G1 Keyboard Teleoperation ---")
        print("  w/s  : Left hip pitch +/-")
        print("  a/d  : Left hip roll  +/-")
        print("  q/e  : Waist yaw      +/-")
        print("  i/k  : R shoulder pitch +/-")
        print("  j/l  : R shoulder yaw   +/-")
        print("  r    : Reset all to zero")
        print("  ESC  : Quit")
        print("---------------------------------\n")

    def _on_press(self, key):
        try:
            char = key.char
        except AttributeError:
            if key == keyboard.Key.esc:
                rclpy.shutdown()
            return

        if char == "r":
            with self._lock:
                self._positions[:] = 0.0
            self.get_logger().info("Reset all joints to zero")
            return

        if char in KEY_MAP:
            idx, delta = KEY_MAP[char]
            with self._lock:
                new_val = self._positions[idx] + delta
                if idx in LIMITS:
                    new_val = float(np.clip(new_val, *LIMITS[idx]))
                self._positions[idx] = new_val

    def _publish(self):
        with self._lock:
            pos = self._positions.copy()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = pos.tolist()
        self._pub.publish(msg)

    def destroy_node(self):
        self._listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
