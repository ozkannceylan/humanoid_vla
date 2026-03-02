"""
arm_teleop_node.py
Keyboard teleoperation targeting only the 14 arm joints (bimanual).
All other joints (legs, waist) are commanded to 0.0 so gravity comp holds them.

Joint indices in the 29-actuator ctrl[] array:
  15: left_shoulder_pitch   22: right_shoulder_pitch
  16: left_shoulder_roll    23: right_shoulder_roll
  17: left_shoulder_yaw     24: right_shoulder_yaw
  18: left_elbow            25: right_elbow
  19: left_wrist_roll       26: right_wrist_roll
  20: left_wrist_pitch      27: right_wrist_pitch
  21: left_wrist_yaw        28: right_wrist_yaw

Key bindings:
  RIGHT ARM            LEFT ARM
  W / S  shoulder_pitch +/-    I / K  shoulder_pitch +/-
  A / D  shoulder_roll  +/-    J / L  shoulder_roll  +/-
  Q / E  shoulder_yaw   +/-    U / O  shoulder_yaw   +/-
  R / F  elbow          +/-    7 / 8  elbow          +/-
  Z / X  wrist_pitch    +/-    9 / 0  wrist_pitch    +/-
  space  reset all arms to zero
  ESC    quit

Run: ros2 run vla_mujoco_bridge arm_teleop_node
NOTE: Run in a graphical terminal (pynput requires X display).
"""

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pynput import keyboard

NUM_JOINTS = 29
DELTA = 0.04  # radians per keypress

# Joint limits (radians) — conservative, well within MJCF ctrlrange
_ARM_LIMITS: dict[int, tuple[float, float]] = {
    15: (-2.8,  2.8),   # L shoulder_pitch
    16: (-1.5,  2.5),   # L shoulder_roll
    17: (-2.0,  2.0),   # L shoulder_yaw
    18: (-0.1,  2.8),   # L elbow
    19: (-1.97, 1.97),  # L wrist_roll
    20: (-1.61, 1.61),  # L wrist_pitch
    21: (-1.97, 1.97),  # L wrist_yaw
    22: (-2.8,  2.8),   # R shoulder_pitch
    23: (-2.5,  1.5),   # R shoulder_roll   (mirrored)
    24: (-2.0,  2.0),   # R shoulder_yaw
    25: (-0.1,  2.8),   # R elbow
    26: (-1.97, 1.97),  # R wrist_roll
    27: (-1.61, 1.61),  # R wrist_pitch
    28: (-1.97, 1.97),  # R wrist_yaw
}

# key char → (joint_index, sign)
_KEY_MAP: dict[str, tuple[int, float]] = {
    # Right arm
    "w": (22, +1), "s": (22, -1),   # R shoulder_pitch
    "a": (23, +1), "d": (23, -1),   # R shoulder_roll
    "q": (24, +1), "e": (24, -1),   # R shoulder_yaw
    "r": (25, +1), "f": (25, -1),   # R elbow
    "z": (27, +1), "x": (27, -1),   # R wrist_pitch
    # Left arm
    "i": (15, +1), "k": (15, -1),   # L shoulder_pitch
    "j": (16, +1), "l": (16, -1),   # L shoulder_roll
    "u": (17, +1), "o": (17, -1),   # L shoulder_yaw
    "7": (18, +1), "8": (18, -1),   # L elbow
    "9": (20, +1), "0": (20, -1),   # L wrist_pitch
}


class ArmTeleopNode(Node):
    def __init__(self):
        super().__init__("arm_teleop_node")
        self._pub = self.create_publisher(JointState, "/joint_commands", 10)
        self._positions = np.zeros(NUM_JOINTS)
        self._lock = threading.Lock()

        self.get_logger().info(
            "Arm teleop ready.\n"
            "  RIGHT ARM: W/S=shoulder_pitch  A/D=roll  Q/E=yaw  R/F=elbow  Z/X=wrist_pitch\n"
            "  LEFT  ARM: I/K=shoulder_pitch  J/L=roll  U/O=yaw  7/8=elbow  9/0=wrist_pitch\n"
            "  SPACE=reset  ESC=quit"
        )

    def apply_key(self, char: str) -> None:
        if char == " ":
            with self._lock:
                self._positions[15:29] = 0.0
            self.get_logger().info("Arms reset to zero.")
            self._publish()
            return

        mapping = _KEY_MAP.get(char)
        if mapping is None:
            return
        idx, sign = mapping
        lo, hi = _ARM_LIMITS[idx]
        with self._lock:
            self._positions[idx] = float(
                np.clip(self._positions[idx] + sign * DELTA, lo, hi)
            )
        self._publish()

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            msg.position = self._positions.tolist()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmTeleopNode()

    stop_event = threading.Event()

    def on_press(key):
        try:
            char = key.char
        except AttributeError:
            if key == keyboard.Key.space:
                node.apply_key(" ")
            elif key == keyboard.Key.esc:
                node.get_logger().info("ESC — shutting down.")
                stop_event.set()
            return
        node.apply_key(char)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        listener.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
