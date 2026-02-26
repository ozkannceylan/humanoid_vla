# Phase A — Task Tracker

Last updated: 2026-02-26

---

## Milestone 1: Environment Setup ✅

- [x] Install ROS2 Humble (apt, Ubuntu 22.04)
- [x] Create Python venv with `--system-site-packages`
- [x] `pip install mujoco opencv-python numpy pynput` (also installed system-wide for ros2 run)
- [x] Clone `unitree_mujoco` and `mujoco_menagerie` into `repos/`
- [x] Verify: `python3 -c "import mujoco, rclpy, cv2; print('OK')"`

**Notes:**
- MuJoCo 3.5.0 installed
- ROS2 Humble (ros-humble-desktop) installed
- G1 model: 29 torque-controlled `<motor>` actuators (NOT position servos)
- zsh users must use `setup.zsh`, not `setup.bash`

---

## Milestone 2: G1 Model + Egocentric Camera ✅

- [x] Create `sim/models/g1_29dof.xml` (local copy with camera + absolute meshdir)
- [x] Create `sim/g1_with_camera.xml` (scene wrapper including local model)
- [x] Create `sim/test_g1.py` (standalone: viewer + camera preview + joint list)
- [x] Run `test_g1.py` — viewer opens, camera renders, 29 joints listed ✅ (user confirmed working)

**Notes:**
- Cannot augment body via second `<worldbody>` block — MuJoCo raises "repeated name"
- Camera added directly inside `torso_link` in `sim/models/g1_29dof.xml`
- meshdir must be an absolute path when using `<include>`

---

## Milestone 3: ROS2 Bridge Package ✅

- [x] `vla_mujoco_bridge` package created and built with colcon
- [x] `mujoco_sim.py` — PD controller + thread-safe state wrapper
- [x] `bridge_node.py` — `/joint_states` (100Hz), `/camera/image_raw` (30Hz), `/joint_commands`
- [x] `teleop_node.py` — pynput keyboard → position targets → `/joint_commands`
- [x] Runtime confirmed: user ran bridge + teleop, joints respond to keyboard ✅

---

## ✅ PHASE A COMPLETE

All three milestones done. The robot falls during teleoperation — this is **expected physics**,
not a bug. A torque-controlled humanoid with no balance controller will fall when joints are
moved independently. Balance/locomotion control is a separate research topic (see ozkan_todo.md).

---

## Environment Quick Reference

```bash
# Source everything (each new terminal — zsh)
source /opt/ros/humble/setup.zsh
source /media/orka/storage/robotics/ros2_ws/install/setup.zsh

# Standalone sim test (no ROS2)
MUJOCO_GL=egl python3 /media/orka/storage/robotics/sim/test_g1.py

# Bridge + teleop (two terminals)
ros2 run vla_mujoco_bridge bridge_node
ros2 run vla_mujoco_bridge teleop_node

# Monitor topics
ros2 topic hz /camera/image_raw     # expect ~30 Hz
ros2 topic hz /joint_states         # expect ~100 Hz
```

---

## Next: Phase B

See CLAUDE.md § Phase B — VLA Integration:
- Set up LeRobot / unitree_IL_lerobot
- Collect teleoperation demonstrations (20-50 per task)
- Train ACT model
- Evaluate VLA control loop: camera → model → joint actions → MuJoCo

**Prerequisite research first** (see `tasks/ozkan_todo.md`):
- Understand humanoid balance and locomotion control before Phase B data collection
- A robot that immediately falls cannot generate useful demonstration data
