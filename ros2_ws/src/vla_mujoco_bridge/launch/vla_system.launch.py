"""
Launch file for the complete VLA system.

Starts:
  1. rosbridge_server — WebSocket bridge (port 9090) for RosClaw/OpenClaw
  2. task_manager_node — VLA inference node (accepts NL task commands)

Usage:
  source /opt/ros/jazzy/setup.bash
  source ~/projects/humanoid_vla/ros2_ws/install/setup.bash
  ros2 launch vla_mujoco_bridge vla_system.launch.py

  # Then from another terminal (or via rosbridge WebSocket):
  ros2 topic pub --once /vla/task_goal std_msgs/String "data: 'pick up the red cube'"
  ros2 topic pub --once /vla/task_goal std_msgs/String "data: 'pick up the green box with both hands'"
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Default checkpoint paths (relative to workspace root)
    ws_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

    sa_default = os.path.join(ws_root, "data", "checkpoints", "best.pt")
    bm_default = os.path.join(ws_root, "data", "bimanual_checkpoints", "best.pt")

    return LaunchDescription([
        # ── Arguments ──
        DeclareLaunchArgument(
            "single_arm_checkpoint",
            default_value=sa_default,
            description="Path to single-arm ACT model checkpoint",
        ),
        DeclareLaunchArgument(
            "bimanual_checkpoint",
            default_value=bm_default,
            description="Path to bimanual ACT model checkpoint",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Inference device: cuda or cpu",
        ),
        DeclareLaunchArgument(
            "rosbridge_port",
            default_value="9090",
            description="WebSocket port for rosbridge_server",
        ),

        # ── rosbridge_server (WebSocket for RosClaw) ──
        Node(
            package="rosbridge_server",
            executable="rosbridge_websocket_launch.py",
            name="rosbridge",
            parameters=[{
                "port": LaunchConfiguration("rosbridge_port"),
            }],
            output="screen",
        ),

        # ── VLA Task Manager ──
        Node(
            package="vla_mujoco_bridge",
            executable="task_manager_node",
            name="vla_task_manager",
            parameters=[{
                "single_arm_checkpoint": LaunchConfiguration("single_arm_checkpoint"),
                "bimanual_checkpoint": LaunchConfiguration("bimanual_checkpoint"),
                "device": LaunchConfiguration("device"),
            }],
            output="screen",
            # MUJOCO_GL=egl for headless rendering
            additional_env={"MUJOCO_GL": "egl"},
        ),
    ])
