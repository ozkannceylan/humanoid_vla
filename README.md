# Humanoid VLA: Vision-Language-Action Controlled Robot

**Autonomous humanoid robot controlled by a Vision-Language-Action model, commandable via natural language**

![Status](https://img.shields.io/badge/Phase-A%20Complete-green)
![Simulator](https://img.shields.io/badge/Simulator-MuJoCo%203.5.0-blue)
![Robot](https://img.shields.io/badge/Robot-Unitree%20G1-orange)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)

---

## 🎯 Project Vision

A simulated humanoid robot that:
1. **Sees** the world through egocentric cameras
2. **Understands** natural language task commands ("pick up the red cup")
3. **Acts** by generating joint-level motor commands through a VLA model
4. **Adapts** via fine-tuning on task-specific demonstrations
5. **Is controllable** remotely through Telegram (RosClaw + OpenClaw integration)

**Scope:** Simulation only (no physical hardware)
**Author:** Ozkan Ceylan
**Hardware:** RTX 4050 (6GB VRAM), Windows 11 + WSL Ubuntu 22.04

---

## 🏗️ System Architecture

```
User on Telegram
    │  "Pick up the red cup"
    ↓
OpenClaw (Charlie) — AI Agent
    │  Parses intent, dispatches task goal
    ↓
RosClaw — ROS2 Bridge
    │  Sends high-level task via action interface
    ↓
VLA Task Manager (ROS2 Node)
    │  Feeds camera + language instruction to VLA
    ↓
VLA Model (ACT / GR00T N1)
    │  Camera image + "pick cup" → joint actions (30Hz)
    ↓
Simulated Humanoid (MuJoCo)
    │  Executes joint commands, physics simulation
    ↓
Task Complete → Status reported to Telegram
```

---

## 📋 Current Status: Phase A Complete ✅

### What's Working

✅ **Simulation:** Unitree G1 (29 DOF, torque-controlled) in MuJoCo 3.5.0
✅ **Camera:** Egocentric 640×480 RGB camera @ 30Hz on torso_link
✅ **ROS2 Bridge:** Real-time joint states (100Hz) and camera feed (30Hz)
✅ **Teleoperation:** Keyboard control for manual demonstration collection
✅ **Performance:** Verified on RTX 4050 with VRAM headroom for VLA models

**Comprehensive verification completed:** See [Phase A Testing Guide](docs/phase_a_testing_guide.md)

### Known Behaviors

⚠️ **Robot falls during teleoperation** - This is **expected physics**, not a bug. A torque-controlled humanoid without a balance controller will fall when joints move. Balance control is a Phase B prerequisite.

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Ubuntu 22.04 (native or WSL2)
- **ROS2:** Humble
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with 4GB+ VRAM (for Phase B VLA training)

### Installation

```bash
# 1. Install ROS2 Humble
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions

# 2. Clone repository
cd ~/projects
git clone https://github.com/YOUR_USERNAME/humanoid_vla.git
cd humanoid_vla

# 3. Install Python dependencies
pip install mujoco opencv-python numpy pynput

# 4. Build ROS2 workspace
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build
source install/setup.bash

# 5. Verify installation
python3 sim/test_g1.py
```

---

## 🎮 Usage

### Standalone Simulation (No ROS2)

Test MuJoCo simulation with viewer and camera preview:

```bash
python3 sim/test_g1.py
```

**Controls:**
- **Viewer window:** Mouse to rotate view
- **Camera preview:** Press 'q' to quit
- **Terminal:** Ctrl+C to exit

---

### ROS2 Bridge + Teleoperation

Run the full system with ROS2 integration:

**Terminal 1:** Launch MuJoCo-ROS2 bridge
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge bridge_node
```

**Terminal 2:** Run keyboard teleoperation
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge teleop_node
```

**Keyboard controls** (focus Terminal 2):
- `w` / `s` — Left hip pitch +/-
- `a` / `d` — Left hip roll +/-
- `q` / `e` — Waist yaw +/-
- `i` / `k` — Right shoulder pitch +/-
- `j` / `l` — Right shoulder yaw +/-
- `r` — Reset all joints to zero
- `ESC` — Quit

**Terminal 3 (optional):** View camera feed
```bash
ros2 run rqt_image_view rqt_image_view
# Select /camera/image_raw from dropdown
```

---

### Monitor ROS2 Topics

```bash
# Check topic rates
ros2 topic hz /joint_states        # Expected: ~100 Hz
ros2 topic hz /camera/image_raw    # Expected: ~30 Hz

# View topic data
ros2 topic echo /joint_states --once
ros2 topic echo /joint_commands --once

# List all topics
ros2 topic list
```

---

## 🧪 Testing & Verification

### Automated Tests

```bash
# Level 0: Standalone simulation (60 seconds)
python3 tests/test_l0_standalone.py

# Extended test with VRAM profiling (5 minutes, headless)
python3 tests/test_l0_standalone.py --duration 300 --headless
```

### Comprehensive Manual Testing

See [Phase A Testing Guide](docs/phase_a_testing_guide.md) for:
- L0: Standalone simulation verification
- L1: ROS2 component verification
- L2: End-to-end integration testing
- L3: RTX 4050 performance profiling
- L4: Robustness & stability testing

**Performance baseline:** [Phase A Baseline Metrics](docs/phase_a_baseline.md)

---

## 📐 Technical Specifications

### Robot Model

- **Platform:** Unitree G1 (29 DOF humanoid)
- **Actuators:** Torque-controlled `<motor>` (not position servos)
- **Control:** PD controller with tuned gains (Kp: 3-70 Nm/rad, Kd: 0.1*Kp)
- **Hands:** Dex3-1 dexterous hands (future: manipulation tasks)

**Joint groups:**
- Legs: 6 DOF × 2 (hip pitch/roll/yaw, knee, ankle pitch/roll)
- Waist: 3 DOF (yaw, roll, pitch)
- Arms: 7 DOF × 2 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)

### Simulation

- **Physics:** MuJoCo 3.5.0
- **Timestep:** 500 Hz (2ms)
- **Integrator:** Implicit (stable for stiff systems)
- **Rendering:** EGL offscreen (headless compatible)
- **Camera:** 640×480 RGB @ 30Hz (egocentric, attached to torso_link)

### ROS2 Interface

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 100 Hz | Current joint positions/velocities |
| `/camera/image_raw` | `sensor_msgs/Image` | 30 Hz | Egocentric camera feed (BGR8) |
| `/joint_commands` | `sensor_msgs/JointState` | On demand | Desired joint positions for PD controller |

**Future (Phase B):**
- `/vla/task_goal` — Action interface for natural language tasks
- `/vla/feedback` — Task progress for RosClaw
- `/vla/status` — Human-readable status

---

## 🗺️ Development Roadmap

### Phase A: MuJoCo + ROS2 Setup ✅ COMPLETE

**Duration:** 2 weeks
**Status:** All milestones achieved, verified on RTX 4050

- [x] Environment setup (ROS2 Humble, MuJoCo 3.5.0)
- [x] Unitree G1 model with egocentric camera
- [x] ROS2 bridge node (joint states, camera, commands)
- [x] Keyboard teleoperation
- [x] Comprehensive testing (L0-L4 verification)

**Deliverables:**
- Verified simulation with real-time performance
- Documentation: testing guide, baseline metrics
- VRAM headroom confirmed for Phase B

---

### Phase B: VLA Integration (NEXT)

**Duration:** 4-6 weeks
**Status:** Blocked on gravity compensation (Phase B.0)

**B.0: Balance Controller (Prerequisite)**
- [ ] Implement gravity compensation in PD controller
- [ ] Train RL standing policy OR use whole-body control
- [ ] Verify robot can maintain upright posture

**B.1: Demonstration Collection**
- [ ] Set up LeRobot / unitree_IL_lerobot
- [ ] Collect teleoperation demos (20-50 episodes per task)
- [ ] Convert to HDF5 format with language annotations

**B.2: VLA Model Training**
- [ ] Train ACT model on pick/place task
- [ ] Evaluate success rate (>80% target)
- [ ] Fine-tune on additional tasks

**B.3: VLA Control Loop**
- [ ] Implement VLA inference node (camera → actions @ 30Hz)
- [ ] Integrate with ROS2 bridge
- [ ] End-to-end test: language command → task completion

---

### Phase C: Multi-Task & Complexity

**Duration:** 2 weeks

- [ ] Train on 3+ tasks (pick, place, transfer)
- [ ] Hierarchical control: locomotion + manipulation
- [ ] Experiment with GR00T N1 (if VRAM permits)

---

### Phase D: RosClaw/OpenClaw Integration

**Duration:** 1-2 weeks

- [ ] Set up RosClaw plugin (rosbridge WebSocket)
- [ ] Connect OpenClaw (Charlie) to Telegram
- [ ] End-to-end: Telegram → task dispatch → VLA execution
- [ ] Camera snapshots sent back to user

---

### Phase E: Polish & Documentation

**Duration:** 1 week

- [ ] Multi-task demo video
- [ ] Documentation cleanup
- [ ] GitHub repo public release

**Total timeline:** ~8-10 weeks at 1-2 hours/day

---

## 📚 Key Resources

### Repositories

- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation models
- [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot) — Imitation learning pipeline
- [lerobot](https://github.com/huggingface/lerobot) — Robot learning framework (HuggingFace)
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — High-quality robot models

### Papers

- **ACT:** "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
- **GR00T N1:** "An Open Foundation Model for Humanoid Robots" (NVIDIA, 2025)
- **OpenVLA:** "Open Vision-Language-Action Models" (Kim et al., 2024)

### Documentation

- [CLAUDE.md](CLAUDE.md) — Full project specification & workflow
- [Phase A Testing Guide](docs/phase_a_testing_guide.md) — Comprehensive test procedures
- [Phase A Baseline](docs/phase_a_baseline.md) — Performance metrics template
- [ozkan_todo.md](tasks/ozkan_todo.md) — Research prerequisites for Phase B

---

## 🛠️ Troubleshooting

### Common Issues

**Issue:** `Camera 'ego_camera' not found in model`
- **Fix:** Ensure `sim/g1_with_camera.xml` includes camera in torso_link

**Issue:** `pynput` keyboard error in WSL
- **Fix:** Requires X server (WSLg or VcXsrv). Run `echo $DISPLAY` to verify.

**Issue:** ROS2 nodes don't see topics
- **Fix:** Source workspace: `source ros2_ws/install/setup.bash`

**Issue:** `colcon build` fails
- **Fix:** Source ROS2 first: `source /opt/ros/humble/setup.bash`

**Issue:** Robot falls immediately
- **Expected behavior** - Not a bug. See [Known Behaviors](#known-behaviors)

### Performance Issues

**Low FPS (< 25 Hz camera):**
- Check GPU load with `nvidia-smi`
- Reduce resolution in `mujoco_sim.py` (640×480 → 320×240)
- Run in headless mode for VLA training

**High VRAM usage:**
- Use headless mode (`launch_viewer=False` in bridge_node.py)
- Close other GPU-intensive apps
- See [Phase A Testing Guide](docs/phase_a_testing_guide.md) for profiling

**ROS2 topic rate drops:**
- Check CPU load with `htop`
- Verify real-time kernel (optional): `uname -a`
- Reduce physics Hz (500 → 250) if needed

---

## 🤝 Contributing

This is a personal research project. Issues and suggestions welcome!

**To report bugs or suggest features:**
1. Check [Phase A Testing Guide](docs/phase_a_testing_guide.md) for known issues
2. Open an issue with:
   - Hardware specs (GPU, RAM)
   - Steps to reproduce
   - Logs/screenshots

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

**Third-party assets:**
- Unitree G1 model: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (BSD License)
- MuJoCo: Apache 2.0 License

---

## 🙏 Acknowledgments

- **NVIDIA** for GR00T foundation models
- **Unitree Robotics** for open-source G1 model and IL tooling
- **HuggingFace** for LeRobot framework
- **Google DeepMind** for MuJoCo physics engine

---

## 📧 Contact

**Author:** Ozkan Ceylan
**Project:** Humanoid VLA (Vision-Language-Action Robot)
**Status:** Phase A Complete, Phase B In Planning

For questions or collaboration: [Open an issue](https://github.com/YOUR_USERNAME/humanoid_vla/issues)

---

**Last updated:** 2026-02-26
**Phase:** A (Setup & Verification Complete)
**Next milestone:** Phase B.0 (Gravity Compensation)
