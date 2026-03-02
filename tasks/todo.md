

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
- ROS2 Humble (ros-humble-desktop) on original desktop (Ubuntu 22.04)
- ROS2 Jazzy (ros-jazzy-desktop) on laptop (Ubuntu 24.04) — `install_ros2.sh` auto-detects
- G1 model: 29 torque-controlled `<motor>` actuators (NOT position servos)
- zsh users must use `setup.zsh`, not `setup.bash`
- Ubuntu 24.04: `pip3 install --break-system-packages` required (PEP 668)

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
# Source everything (each new terminal — bash, Ubuntu 24.04 Jazzy laptop)
source /opt/ros/jazzy/setup.bash
source /home/ozkan/projects/humanoid_vla/ros2_ws/install/setup.bash

# Standalone sim test (no ROS2)
MUJOCO_GL=egl python3 /home/ozkan/projects/humanoid_vla/sim/test_g1.py

# Bridge + teleop (two terminals)
ros2 run vla_mujoco_bridge bridge_node
ros2 run vla_mujoco_bridge teleop_node

# Monitor topics
ros2 topic hz /camera/image_raw     # expect ~30 Hz
ros2 topic hz /joint_states         # expect ~100 Hz
```

> **Laptop setup notes (Ubuntu 24.04):**
> - ROS2 Jazzy (not Humble) — runs on Noble
> - Python packages: `pip3 install --break-system-packages mujoco opencv-python numpy pynput`
> - Both `source` lines are in `~/.bashrc` automatically
> - meshdir: `/home/ozkan/projects/humanoid_vla/repos/unitree_mujoco/unitree_robots/g1/meshes`

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

---

# Phase B — VLA Integration

Last updated: 2026-03-02

**Strategy:** Fixed-base arm manipulation (standard ACT paper / LeRobot pipeline approach).
Robot pelvis is kinematically frozen at standing height (z=0.793m). Only arm joints are
teleoperated. Task: right hand reaches and touches the red cube on the table.

---

## Milestone B1: Robot Stability ✅

- [x] `mujoco_sim.py` — gravity compensation: `qfrc_bias[6:35]` added to PD torques
- [x] `mujoco_sim.py` — `fixed_base=True` kinematically freezes pelvis at initial height
- [x] `mujoco_sim.py` — `model_path`, `gravity_comp`, `fixed_base` constructor params
- [x] Verified: pelvis stays at z=0.793 with gravity_comp + fixed_base

---

## Milestone B2: Manipulation Scene ✅

- [x] `sim/g1_with_camera.xml` — red cube added (freejoint, 5cm, at table surface z=0.825)
- [x] `sim/g1_with_camera.xml` — `scene_camera` added (fixed overhead debug view)
- [x] `bridge_node.py` — ROS2 params: `model_path`, `gravity_comp`, `fixed_base`, `render_hz`, `physics_hz`
- [x] `pyproject.toml` added — fixes setuptools 68→80 regression on Ubuntu 24.04

---

## Milestone B3: Arm Teleoperation ✅

- [x] `arm_teleop_node.py` — 14-DOF bimanual arm keyboard control (joints 15-28)
- [x] Conservative joint limits per arm joint
- [ ] **USER ACTION:** Run arm_teleop in fixed_base bridge, confirm cube is reachable

---

## Milestone B4: Demo Recording ✅

- [x] `demo_recorder.py` — HDF5 recorder (obs/action pairs, gzip compressed)
- [x] `h5py` + `pandas` installed
- [x] `LeRobot 0.4.4` installed
- [ ] **USER ACTION:** Collect 20-50 reach demos (see Quick Ref below)

---

## Milestone B5: Convert + Train

- [ ] Run `python3 scripts/convert_to_lerobot.py`
- [ ] Run `python3 scripts/train_act.py` (~2-4h on RTX 4050)
- [ ] Monitor loss curves, eval on 10 rollouts

---

## Milestone B6: VLA Inference Loop (Phase B completion)

- [ ] Create `vla_node.py` — ACT inference: camera + state → joint commands @ 30Hz
- [ ] Integrate with bridge_node
- [ ] Evaluate: >50% reach success rate defines Phase B complete

---

## Phase B Quick Reference

```bash
# ─── Terminal 1: Fixed-base bridge (gravity comp ON, pelvis frozen) ───
source /opt/ros/jazzy/setup.bash
source /home/ozkan/projects/humanoid_vla/ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge bridge_node \
  --ros-args -p fixed_base:=true -p gravity_comp:=true

# ─── Terminal 2: Arm-only teleop ───
source /opt/ros/jazzy/setup.bash
source /home/ozkan/projects/humanoid_vla/ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge arm_teleop_node

# ─── Terminal 3: Demo recorder ───
source /opt/ros/jazzy/setup.bash
source /home/ozkan/projects/humanoid_vla/ros2_ws/install/setup.bash
cd /home/ozkan/projects/humanoid_vla
ros2 run vla_mujoco_bridge demo_recorder \
  --ros-args -p task:='reach red cube'
# Controls: Enter=start  Esc=save  d=discard  p=stats

# ─── After collecting demos ───
cd /home/ozkan/projects/humanoid_vla
python3 scripts/convert_to_lerobot.py
python3 scripts/train_act.py
```

### Arm Teleop Key Bindings
| Key | Joint | | Key | Joint |
|-----|-------|-|-----|-------|
| W/S | Right shoulder pitch ±| | I/K | Left shoulder pitch ± |
| A/D | Right shoulder roll ± | | J/L | Left shoulder roll ± |
| Q/E | Right shoulder yaw ±  | | U/O | Left shoulder yaw ± |
| R/F | Right elbow ±         | | 7/8 | Left elbow ± |
| Z/X | Right wrist pitch ±   | | 9/0 | Left wrist pitch ± |
| **space** | Reset all arms to zero | | **Esc** | Quit |
