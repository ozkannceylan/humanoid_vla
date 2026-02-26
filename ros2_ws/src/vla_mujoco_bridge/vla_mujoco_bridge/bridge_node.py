"""
bridge_node.py
ROS2 node that bridges MuJoCo simulation to ROS2 topics.

Publishes:
  /joint_states       sensor_msgs/JointState   ~100 Hz
  /camera/image_raw   sensor_msgs/Image        ~30 Hz

Subscribes:
  /joint_commands     sensor_msgs/JointState   (on demand)

Run: ros2 run vla_mujoco_bridge bridge_node
"""

import os
import threading
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
import cv2
from cv_bridge import CvBridge

from vla_mujoco_bridge.mujoco_sim import MujocoSim


class MujocoBridgeNode(Node):
    def __init__(self, sim: MujocoSim):
        super().__init__("mujoco_bridge_node")
        self.sim = sim
        self.bridge = CvBridge()
        cbg = ReentrantCallbackGroup()

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 1)

        self.create_subscription(
            JointState, "/joint_commands", self._on_joint_cmd, 10,
            callback_group=cbg,
        )
        self.create_timer(0.01, self._pub_joints, callback_group=cbg)      # 100 Hz
        self.create_timer(1.0 / 30.0, self._pub_camera, callback_group=cbg) # 30 Hz

        self.get_logger().info("MuJoCo bridge node ready")

    def _pub_joints(self):
        pos, vel, names = self.sim.get_joint_state()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = pos.tolist()
        msg.velocity = vel.tolist()
        self.joint_pub.publish(msg)

    def _pub_camera(self):
        frame = self.sim.get_latest_frame()
        if frame is None:
            return
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_camera"
        self.image_pub.publish(msg)

    def _on_joint_cmd(self, msg: JointState):
        if msg.position:
            self.sim.set_joint_command(np.array(msg.position))


def main(args=None):
    rclpy.init(args=args)

    sim = MujocoSim(render_hz=30.0, physics_hz=500.0)
    node = MujocoBridgeNode(sim)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    node.get_logger().info("Starting physics loop (viewer must be closed to exit)")
    try:
        sim.run_physics_loop(launch_viewer=True)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
        executor.shutdown()
        rclpy.shutdown()
