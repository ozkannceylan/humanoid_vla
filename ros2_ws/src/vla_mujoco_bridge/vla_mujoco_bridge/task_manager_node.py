#!/usr/bin/env python3
"""
task_manager_node.py

ROS2 node that accepts natural language task commands and runs
VLA (ACT model) inference in MuJoCo simulation.

Subscribes:
  /vla/task_goal    std_msgs/String   Natural language command

Publishes:
  /vla/status       std_msgs/String   JSON: {step, progress, status, result}
  /camera/image_raw sensor_msgs/Image Ego camera at 30Hz during execution

Parameters:
  single_arm_checkpoint  (string)  Path to single-arm ACT model
  bimanual_checkpoint    (string)  Path to bimanual ACT model
  device                 (string)  'cuda' or 'cpu'

Supported task commands:
  Single-arm (uses kinematic sim + weld constraint):
    - "reach the red cube"
    - "grasp the red cube"
    - "pick up the red cube"
    - "place the red cube on the blue plate"

  Bimanual (uses physics sim + friction grasping):
    - "pick up the green box with both hands"
    - Any command containing "box" or "both hands" or "bimanual"

Architecture:
  Telegram → OpenClaw → RosClaw → rosbridge (WebSocket)
      → /vla/task_goal (String) → THIS NODE → ACT inference → MuJoCo
      → /vla/status (JSON) → rosbridge → RosClaw → Telegram

Run standalone:
  MUJOCO_GL=egl ros2 run vla_mujoco_bridge task_manager_node \\
    --ros-args -p single_arm_checkpoint:=data/checkpoints/best.pt \\
               -p bimanual_checkpoint:=data/bimanual_checkpoints/best.pt

Run via launch:
  ros2 launch vla_mujoco_bridge vla_system.launch.py
"""

import json
import math
import os
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Paths for scripts/ imports
_SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from act_model import ACTPolicy, TASK_LABELS, task_to_id

# Lazy imports — loaded only when needed (avoid loading physics_sim for single-arm)
_PhysicsSim = None
_SimWrapper = None


def _lazy_physics():
    global _PhysicsSim
    if _PhysicsSim is None:
        from physics_sim import PhysicsSim
        _PhysicsSim = PhysicsSim
    return _PhysicsSim


def _lazy_simwrapper():
    global _SimWrapper
    if _SimWrapper is None:
        from generate_demos import SimWrapper
        _SimWrapper = SimWrapper
    return _SimWrapper


# ────────────────────────────────────────────────────────
# NL Command Parser
# ────────────────────────────────────────────────────────

BIMANUAL_KEYWORDS = ["box", "both hands", "bimanual", "two hand", "green"]

SINGLE_ARM_COMMANDS = {
    "reach": "reach the red cube",
    "grasp": "grasp the red cube",
    "pick": "pick up the red cube",
    "place": "place the red cube on the blue plate",
}


def parse_task_command(text: str):
    """Parse NL command into (mode, task_label).

    Returns:
        mode: "single_arm" or "bimanual"
        task_label: canonical task label string
    """
    text_lower = text.strip().lower()

    # Check bimanual keywords
    if any(kw in text_lower for kw in BIMANUAL_KEYWORDS):
        return "bimanual", "pick up the green box with both hands"

    # Check exact matches
    if text_lower in TASK_LABELS:
        return "single_arm", text_lower

    # Check short aliases
    for alias, full in SINGLE_ARM_COMMANDS.items():
        if alias in text_lower:
            return "single_arm", full

    # Default to pick (most common task)
    return "single_arm", "pick up the red cube"


# ────────────────────────────────────────────────────────
# Inference engines
# ────────────────────────────────────────────────────────

def run_single_arm(model, task_label, status_cb, camera_cb,
                   device='cuda', max_steps=250, seed=None):
    """Run single-arm task with kinematic sim. Returns dict with result."""
    from generate_demos import RIGHT_ARM_CTRL
    import mujoco

    SimWrapper = _lazy_simwrapper()
    sim = SimWrapper()
    rng = np.random.default_rng(seed or int(time.time()) % 10000)
    sim.reset_with_noise(rng)

    task_id = task_to_id(task_label)
    chunk_size = model.chunk_size
    action_dim = model.action_dim

    is_composite = ("pick" in task_label or "place" in task_label)
    approach_task_id = task_to_id("grasp the red cube") if is_composite else None
    final_task_id = task_id
    active_task_id = approach_task_id if is_composite else final_task_id

    grasped = False
    grasp_step = -1
    released = False
    is_place = ("place" in task_label)
    place_target = sim.place_pos if is_place else None

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    for step in range(max_steps):
        # Re-plan every 5 steps
        if step % 5 == 0:
            image = sim.render_camera()
            pos, vel = sim.get_obs()
            state = np.concatenate([pos, vel])
            chunk = model.predict(image, state, active_task_id, device=device)
            for i in range(chunk_size):
                w = math.exp(-0.01 * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        sim.target_pos[RIGHT_ARM_CTRL] = action[RIGHT_ARM_CTRL]
        sim.step_frame()

        # Auto-grasp
        if not grasped and not released:
            d = np.linalg.norm(sim.hand_pos - sim.cube_pos)
            if d < 0.04:
                sim.set_weld(True)
                grasped = True
                grasp_step = step
                if is_composite:
                    active_task_id = final_task_id
                    action_sum[step + 1:] = 0
                    weight_sum[step + 1:] = 0

        # Auto-release (place task)
        if is_place and grasped and place_target is not None:
            steps_grasped = step - grasp_step
            cube_near = np.linalg.norm(sim.cube_pos[:2] - place_target[:2]) < 0.06
            hand_low = sim.hand_pos[2] < place_target[2] + 0.12
            if steps_grasped > 30 and cube_near and hand_low:
                sim.set_weld(False)
                grasped = False
                released = True
                sim.data.qpos[sim.cube_qpos_adr + 2] = 0.825
                mujoco.mj_forward(sim.model, sim.data)

        # Publish status every 10 steps
        if step % 10 == 0:
            progress = step / max_steps * 100
            state_str = "grasped" if grasped else ("released" if released else "executing")
            status_cb(step, progress, state_str)

        # Publish camera every 5 steps (~6Hz for status, keeps bandwidth low)
        if step % 5 == 0:
            camera_cb(sim.render_camera())

    # Evaluate success
    from evaluate import SUCCESS_FN
    check = SUCCESS_FN.get(task_label)
    success = check(sim, grasped) if check else False

    sim.renderer.close()
    return {
        "success": success,
        "task": task_label,
        "mode": "single_arm",
        "steps": max_steps,
        "grasped": grasped,
    }


def run_bimanual(model, task_label, status_cb, camera_cb,
                 device='cuda', max_steps=250, seed=None):
    """Run bimanual task with physics sim. Returns dict with result."""
    from physics_sim import LEFT_ARM_CTRL, RIGHT_ARM_CTRL

    PhysicsSim = _lazy_physics()
    sim = PhysicsSim()
    rng = np.random.default_rng(seed or int(time.time()) % 10000)
    sim.reset_with_noise(rng)
    box_init_z = sim.box_pos[2]

    chunk_size = model.chunk_size
    action_dim = model.action_dim

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    for step in range(max_steps):
        # Re-plan every 5 steps
        if step % 5 == 0:
            image = sim.render_camera()
            pos_all, vel_all = sim.get_obs()
            state = np.concatenate([
                pos_all[LEFT_ARM_CTRL], pos_all[RIGHT_ARM_CTRL],
                vel_all[LEFT_ARM_CTRL], vel_all[RIGHT_ARM_CTRL],
            ])
            chunk = model.predict(image, state, task_id=0, device=device)
            for i in range(chunk_size):
                w = math.exp(-0.01 * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        sim.target_pos[LEFT_ARM_CTRL] = action[:7]
        sim.target_pos[RIGHT_ARM_CTRL] = action[7:]
        sim.step_frame()

        # Publish status every 10 steps
        if step % 10 == 0:
            contacts = sim.get_palm_box_contacts()
            lift_cm = (sim.box_pos[2] - box_init_z) * 100
            progress = step / max_steps * 100
            state_str = f"lift={lift_cm:.1f}cm F=[{contacts['left_force']:.0f},{contacts['right_force']:.0f}]N"
            status_cb(step, progress, state_str)

        # Publish camera every 5 steps
        if step % 5 == 0:
            camera_cb(sim.render_camera())

    # Final evaluation
    contacts = sim.get_palm_box_contacts()
    lift_cm = (sim.box_pos[2] - box_init_z) * 100
    both_contact = contacts["left_contact"] and contacts["right_contact"]
    min_force = min(contacts["left_force"], contacts["right_force"])
    success = lift_cm >= 3.0 and both_contact and min_force >= 2.0

    sim.renderer.close()
    return {
        "success": success,
        "task": task_label,
        "mode": "bimanual",
        "steps": max_steps,
        "lift_cm": round(lift_cm, 1),
        "left_force": round(contacts["left_force"], 1),
        "right_force": round(contacts["right_force"], 1),
    }


# ────────────────────────────────────────────────────────
# ROS2 Node
# ────────────────────────────────────────────────────────

class TaskManagerNode(Node):
    """VLA Task Manager — accepts NL commands, runs ACT inference."""

    def __init__(self):
        super().__init__("vla_task_manager")
        cbg = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter("single_arm_checkpoint", "data/checkpoints/best.pt")
        self.declare_parameter("bimanual_checkpoint", "data/bimanual_checkpoints/best.pt")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")

        self._device = self.get_parameter("device").value
        self._sa_path = self.get_parameter("single_arm_checkpoint").value
        self._bm_path = self.get_parameter("bimanual_checkpoint").value

        # Load models
        self._sa_model = self._load_model(self._sa_path, "single-arm")
        self._bm_model = self._load_model(self._bm_path, "bimanual")

        # Publishers
        self.status_pub = self.create_publisher(String, "/vla/status", 10)
        self.camera_pub = self.create_publisher(Image, "/camera/image_raw", 1)

        # Subscriber
        self.create_subscription(
            String, "/vla/task_goal", self._on_task_goal, 10,
            callback_group=cbg)

        # State
        self._executing = False
        self._exec_lock = threading.Lock()

        self.get_logger().info(
            f"VLA Task Manager ready — listening on /vla/task_goal\n"
            f"  Single-arm model: {self._sa_path}\n"
            f"  Bimanual model:   {self._bm_path}\n"
            f"  Device: {self._device}")

    def _load_model(self, path, label):
        """Load an ACT checkpoint."""
        if not os.path.isfile(path):
            self.get_logger().warn(f"{label} checkpoint not found: {path}")
            return None

        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        config = ckpt['config']
        model = ACTPolicy(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            chunk_size=config['chunk_size'],
            hidden_dim=config['hidden_dim'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_tasks=config['num_tasks'],
        ).to(self._device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        self.get_logger().info(
            f"  Loaded {label} ACT [epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}]")
        return model

    def _publish_status(self, step, progress, status, result=None):
        """Publish JSON status message."""
        msg = String()
        data = {
            "step": step,
            "progress_pct": round(progress, 1),
            "status": status,
        }
        if result is not None:
            data["result"] = result
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)

    def _publish_camera(self, frame: np.ndarray):
        """Publish camera frame as ROS2 Image."""
        msg = Image()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "rgb8"
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_camera"
        self.camera_pub.publish(msg)

    def _on_task_goal(self, msg: String):
        """Handle incoming task command."""
        command = msg.data.strip()
        if not command:
            return

        with self._exec_lock:
            if self._executing:
                self.get_logger().warn(
                    f"Task already executing, ignoring: '{command}'")
                self._publish_status(0, 0, "busy",
                                     {"error": "task already executing"})
                return
            self._executing = True

        # Run in separate thread to not block the ROS2 executor
        thread = threading.Thread(
            target=self._execute_task, args=(command,), daemon=True)
        thread.start()

    def _execute_task(self, command: str):
        """Execute a task command (runs in separate thread)."""
        try:
            mode, task_label = parse_task_command(command)
            self.get_logger().info(
                f"Executing: '{command}' → mode={mode}, task='{task_label}'")

            self._publish_status(0, 0, "starting",
                                 {"command": command, "mode": mode,
                                  "task": task_label})

            if mode == "bimanual":
                if self._bm_model is None:
                    self._publish_status(0, 0, "error",
                                         {"error": "bimanual model not loaded"})
                    return
                result = run_bimanual(
                    self._bm_model, task_label,
                    status_cb=self._publish_status,
                    camera_cb=self._publish_camera,
                    device=self._device)
            else:
                if self._sa_model is None:
                    self._publish_status(0, 0, "error",
                                         {"error": "single-arm model not loaded"})
                    return
                result = run_single_arm(
                    self._sa_model, task_label,
                    status_cb=self._publish_status,
                    camera_cb=self._publish_camera,
                    device=self._device)

            status = "success" if result["success"] else "failed"
            self._publish_status(result["steps"], 100, status, result)

            self.get_logger().info(
                f"Task complete: {status} — {json.dumps(result)}")

        except Exception as e:
            self.get_logger().error(f"Task execution error: {e}")
            self._publish_status(0, 0, "error", {"error": str(e)})
        finally:
            with self._exec_lock:
                self._executing = False


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
