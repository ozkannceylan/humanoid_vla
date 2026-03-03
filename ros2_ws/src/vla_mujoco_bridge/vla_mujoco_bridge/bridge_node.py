"""
bridge_node.py
ROS2 node that bridges MuJoCo simulation to ROS2 topics.

Publishes:
  /joint_states       sensor_msgs/JointState   ~100 Hz
  /camera/image_raw   sensor_msgs/Image        ~30 Hz

Subscribes:
  /joint_commands     sensor_msgs/JointState   (on demand)

Parameters (set via --ros-args -p key:=value):
  model_path    (string)  path to MJCF scene XML
  gravity_comp  (bool)    enable gravity compensation  [default: true]
  fixed_base    (bool)    freeze pelvis at standing height [default: false]
  render_hz     (double)  camera render frequency       [default: 30.0]
  physics_hz    (double)  physics step frequency        [default: 500.0]

Run:
  ros2 run vla_mujoco_bridge bridge_node
  ros2 run vla_mujoco_bridge bridge_node \\
    --ros-args -p model_path:=/path/to/scene.xml -p fixed_base:=true
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
from std_srvs.srv import SetBool
from geometry_msgs.msg import PointStamped

from vla_mujoco_bridge.mujoco_sim import MujocoSim


class MujocoBridgeNode(Node):
    def __init__(self, sim: MujocoSim):
        super().__init__("mujoco_bridge_node")
        self.sim = sim
        cbg = ReentrantCallbackGroup()

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 1)

        self.create_subscription(
            JointState, "/joint_commands", self._on_joint_cmd, 10,
            callback_group=cbg,
        )
        self.create_timer(0.01, self._pub_joints, callback_group=cbg)      # 100 Hz
        self.create_timer(1.0 / 30.0, self._pub_camera, callback_group=cbg) # 30 Hz

        # Services for task_commander
        self.create_service(SetBool, "/grasp", self._srv_grasp, callback_group=cbg)

        # Publish hand & cube positions for task_commander IK
        self.hand_pub = self.create_publisher(PointStamped, "/hand_pos", 10)
        self.cube_pub = self.create_publisher(PointStamped, "/cube_pos", 10)
        self.create_timer(0.05, self._pub_positions, callback_group=cbg)  # 20 Hz

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
        # Manual RGB→Image — avoids cv_bridge / NumPy ABI issues
        msg = Image()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "rgb8"
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_camera"
        self.image_pub.publish(msg)

    def _on_joint_cmd(self, msg: JointState):
        if msg.position:
            self.sim.set_joint_command(np.array(msg.position))

    def _srv_grasp(self, request, response):
        ok = self.sim.set_grasp(request.data)
        response.success = ok
        response.message = ("grasp enabled" if request.data else "grasp released") if ok else "weld not found"
        self.get_logger().info(response.message)
        return response

    def _pub_positions(self):
        stamp = self.get_clock().now().to_msg()
        hand = self.sim.get_site_xpos("right_hand_site")
        if hand is not None:
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.point.x, msg.point.y, msg.point.z = hand.tolist()
            self.hand_pub.publish(msg)
        cube = self.sim.get_body_xpos("red_cube")
        if cube is not None:
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.point.x, msg.point.y, msg.point.z = cube.tolist()
            self.cube_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Read parameters from a temporary node before creating MujocoSim
    param_node = rclpy.create_node("_bridge_params")
    param_node.declare_parameter("model_path", "")
    param_node.declare_parameter("gravity_comp", True)
    param_node.declare_parameter("fixed_base", False)
    param_node.declare_parameter("render_hz", 30.0)
    param_node.declare_parameter("physics_hz", 500.0)

    from vla_mujoco_bridge.mujoco_sim import DEFAULT_MODEL_PATH
    model_path = param_node.get_parameter("model_path").value or DEFAULT_MODEL_PATH
    gravity_comp = param_node.get_parameter("gravity_comp").value
    fixed_base = param_node.get_parameter("fixed_base").value
    render_hz = param_node.get_parameter("render_hz").value
    physics_hz = param_node.get_parameter("physics_hz").value
    param_node.destroy_node()

    sim = MujocoSim(
        model_path=model_path,
        gravity_comp=gravity_comp,
        fixed_base=fixed_base,
        render_hz=render_hz,
        physics_hz=physics_hz,
    )
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
