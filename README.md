# Humanoid VLA — Vision-Language-Action Controlled Humanoid Robot

A simulated **Unitree G1 humanoid robot** controlled by a Vision-Language-Action (VLA) model, commandable via natural language. The robot sees through an egocentric camera, understands task commands like *"pick up the red cube"*, and generates joint-level motor commands through a trained **ACT (Action Chunking with Transformers)** model.

> **Status:** Simulation complete — Phases A–F (Domain Randomization & Generalization)
> **Author:** Ozkan Ceylan
> **Full Report:** [PROJECT_REPORT.md](PROJECT_REPORT.md)

---

## Demo Videos

### Single-Arm Manipulation (4 Tasks)

<table>
<tr>
<td align="center"><b>Reach</b><br><video src="media/reach.mp4" width="300"></video></td>
<td align="center"><b>Grasp</b><br><video src="media/grasp.mp4" width="300"></video></td>
</tr>
<tr>
<td align="center"><b>Pick Up</b><br><video src="media/pick.mp4" width="300"></video></td>
<td align="center"><b>Place</b><br><video src="media/place.mp4" width="300"></video></td>
</tr>
</table>

### Bimanual Physics-Based Grasping

<table>
<tr>
<td align="center"><b>Bimanual Box Lift</b> — Both hands squeeze box via friction only (no weld constraints), full <code>mj_step</code> dynamics<br><video src="media/bimanual.mp4" width="600"></video></td>
</tr>
</table>

> Videos show side-by-side overview camera (left) and robot's egocentric view (right). The ACT model receives only the ego camera image as visual input.

---

## Architecture

```
User: "Pick up the red cube"
  │
  ├─── Telegram → OpenClaw (Charlie) → RosClaw ───┐
  │         (natural language interface)           │
  │                                                ▼
  │                              ┌─────────────────────────────┐
  │                              │  rosbridge (WebSocket:9090) │
  │                              └────────────┬────────────────┘
  │                                           ▼
  │              ┌────────────────────────────────────────────┐
  └─────────────►│  VLA Task Manager (ROS2 Node)              │
                 │                                            │
                 │  NL Parser: "pick up..." → single_arm,     │
                 │             task_id=2                       │
                 │                                            │
                 │  30Hz Control Loop:                        │
                 │    Camera (480×640 RGB) ──► ACT Model ──►  │
                 │    Joint State (58-d)   ──►  (15.6M)  ──►  │
                 │    Task Embedding       ──►           ──►  │
                 │                         20 joint actions    │
                 └────────────────┬───────────────────────────┘
                                  │ Joint commands
                 ┌────────────────▼───────────────────────────┐
                 │  MuJoCo Simulation                         │
                 │  Unitree G1 (29 DOF) + table + objects     │
                 │  Egocentric camera → 480×640 RGB           │
                 │  Physics: 500Hz (mj_step) / Kinematic      │
                 └────────────────────────────────────────────┘
```

**Key insight:** The VLA model runs in a tight 30Hz control loop (camera → action). RosClaw/OpenClaw operates at the **task dispatch level** — it sends the command once and monitors completion.

---

## Quick Start

### Prerequisites

- **Ubuntu 24.04** (tested), or Ubuntu 22.04
- **NVIDIA GPU** with CUDA (RTX 4050 6GB VRAM is sufficient)
- **Python 3.12+**, **ROS2 Jazzy** (or Humble)

### Installation

```bash
# 1. Clone
git clone https://github.com/ozkanceylan/humanoid_vla.git
cd humanoid_vla

# 2. Install ROS2
chmod +x install_ros2.sh && ./install_ros2.sh

# 3. Python dependencies
pip3 install --break-system-packages -r requirements.txt

# 4. Robot models
cd repos
git clone https://github.com/unitreerobotics/unitree_mujoco
git clone https://github.com/google-deepmind/mujoco_menagerie
cd ..

# 5. Build ROS2 workspace
source /opt/ros/jazzy/setup.bash
cd ros2_ws && colcon build --symlink-install && cd ..
source ros2_ws/install/setup.bash
```

### Full Pipeline (Train → Evaluate → Demo)

```bash
# 1. Generate training data (80 single-arm + 30 bimanual demos)
MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20
MUJOCO_GL=egl python3 scripts/generate_bimanual_demos.py --episodes 30

# 2. Train ACT models (~2.5 hours total on RTX 4050)
python3 scripts/train_act.py --demos data/demos --epochs 300 --batch-size 32
python3 scripts/train_bimanual.py --epochs 300

# 3. Evaluate
MUJOCO_GL=egl python3 scripts/evaluate.py --checkpoint data/checkpoints/best.pt --episodes 20
MUJOCO_GL=egl python3 scripts/evaluate_bimanual.py --checkpoint data/bimanual_checkpoints/best.pt --episodes 20

# 4. Interactive demos (opens MuJoCo viewer)
python3 scripts/live_demo.py --checkpoint data/checkpoints/best.pt
python3 scripts/live_bimanual.py --checkpoint data/bimanual_checkpoints/best.pt

# 5. Record demo videos
MUJOCO_GL=egl python3 scripts/record_demo_videos.py
```

### Run via ROS2 (Natural Language Interface)

```bash
# Terminal 1: Launch full VLA system (task manager + rosbridge)
ros2 launch vla_mujoco_bridge vla_system.launch.py

# Terminal 2: Send natural language commands
ros2 topic pub --once /vla/task_goal std_msgs/String "data: 'pick up the red cube'"
ros2 topic pub --once /vla/task_goal std_msgs/String "data: 'pick up the green box with both hands'"

# Terminal 3: Monitor status (JSON)
ros2 topic echo /vla/status
```

---

## Evaluation Results

### Single-Arm Manipulation (Phase C)

Trained for 300 epochs (93 min on RTX 4050, final loss: 0.000009). Evaluated with temporal ensembling and hierarchical task decomposition:

| Task | Success | Rate |
|------|---------|------|
| **Reach** the red cube | 20/20 | **100%** |
| **Grasp** the red cube | 18/20 | **90%** |
| **Pick up** the red cube | 18/20 | **90%** |
| **Place** the red cube on blue plate | 13/20 | **65%** |
| **Overall** | **69/80** | **86.2%** |

### Bimanual Physics-Based Grasping (Phase C2)

Both hands squeeze a 20×15×15cm box using friction only — no weld constraints, full `mj_step` dynamics, PD torque control + gravity compensation:

| Metric | Value |
|--------|-------|
| **Success rate** | **20/20 (100%)** |
| **Lift** | mean=8.5cm, min=6.5cm, max=10.4cm |
| **Contact force** | L=13.6N, R=13.3N (bilateral) |
| **Physics** | `mj_step` at 500Hz, control at 30Hz |
| **Training** | 300 epochs, 52 min, loss=0.000009 |

### Combined: 5 Tasks, 89/100 (89%)

Key inference techniques:
1. **Temporal ensembling** — overlapping action chunks with exponential decay weighting
2. **Hierarchical task decomposition** — composite tasks switch task embedding at grasp trigger
3. **Re-grasp prevention** — `released` flag prevents re-triggering after intentional release

### Generalization (Phase F — Domain Randomization)

Bimanual model trained with position randomization (+/-5cm), random arm starts, and visual domain randomization. Evaluated on out-of-distribution conditions never seen during training:

| Test Distribution | Success | Description |
|-------------------|---------|-------------|
| **In-distribution** | **90%** | Same ranges as training, different seeds |
| **OOD Position** (1.5x) | **80%** | Wider object positions |
| **OOD Visual** | **70%** | Novel table/light colors |
| **OOD Posture** (1.5x) | **60%** | Wider starting arm configs |
| **OOD Combined** | **55%** | All OOD factors simultaneously |

Graceful degradation from 90% to 55% demonstrates real generalization — the model handles unseen conditions rather than catastrophically failing.

---

## ROS2 Integration (Phase D)

The VLA Task Manager accepts natural language commands via ROS2 topics and runs ACT inference in a closed-loop MuJoCo simulation.

### ROS2 Interfaces

| Direction | Topic | Type | Purpose |
|-----------|-------|------|---------|
| Subscribe | `/vla/task_goal` | `std_msgs/String` | Natural language command |
| Publish | `/vla/status` | `std_msgs/String` | JSON: step, progress, result |
| Publish | `/camera/image_raw` | `sensor_msgs/Image` | Ego camera during execution |

### NL Command Examples

| Input | Mode | Task |
|-------|------|------|
| "pick up the red cube" | single_arm | pick up the red cube |
| "reach" | single_arm | reach the red cube |
| "lift the box" | bimanual | pick up the green box with both hands |
| "bimanual grasp" | bimanual | pick up the green box with both hands |

### rosbridge (WebSocket for External Systems)

The launch file co-starts rosbridge_server on port 9090, enabling any WebSocket client (RosClaw, JavaScript, Python) to send commands:

```python
import websocket, json
ws = websocket.create_connection("ws://localhost:9090")
ws.send(json.dumps({
    "op": "publish",
    "topic": "/vla/task_goal",
    "msg": {"data": "pick up the red cube"}
}))
```

---

## The Robot: Unitree G1

| Property | Value |
|----------|-------|
| DOF | 29 torque-controlled joints |
| Control | PD: $\tau = K_p(q_{des} - q) - K_d\dot{q} + \tau_{gravity}$ |
| Camera | Egocentric RGB, 480×640, torso-mounted |
| Fixed base | Pelvis frozen at z=0.793m |
| Right arm | 7 DOF (shoulder pitch/roll/yaw, elbow, wrist p/r/y) |
| Left arm | 7 DOF (mirror configuration) |

---

## Tasks

| ID | Task | Description | Success Criterion |
|----|------|-------------|-------------------|
| 0 | **Reach** | Move hand to the red cube | Hand within 6cm of cube |
| 1 | **Grasp** | Close hand around the cube | Auto-grasp triggered (hand < 4cm) |
| 2 | **Pick** | Lift the cube off the table | Cube z > 0.90m while grasped |
| 3 | **Place** | Move cube to the blue plate | Cube within 6cm of target, released |
| 4 | **Bimanual Lift** | Lift green box with both hands | Box ≥3cm, dual contact, force ≥2N |

---

## ACT Model Architecture

```
Image (480×640×3) ──► ResNet18 (frozen 0-6) ──► AvgPool ──► 512-d ──► Proj ──► 256-d ─┐
                                                                                         │
State (pos + vel) ──► MLP (→256→256) ──────────────────────────────────────────────────│──► Memory
                                                                                         │    (3 tokens)
Task ("pick up..") ──► Embedding ──► 256-d ────────────────────────────────────────────┘
                                                                              │
                                                                ┌─────────────▼──────────────┐
                                                                │  Transformer Decoder        │
                                                                │  4 layers, 4 heads, d=256   │
                                                                │  20 learnable query tokens   │
                                                                └─────────────┬──────────────┘
                                                                              │
                                                                Action chunk: (20, action_dim)
```

| Variant | Params | Trainable | State | Actions | Tasks |
|---------|--------|-----------|-------|---------|-------|
| Single-arm | 15.6M | 12.8M | 58 (29+29) | 29 | 4 |
| Bimanual | 15.6M | 12.8M | 28 (14+14) | 14 | 1 |

Chunk size: 20 timesteps (~0.67s). Training: AdamW (lr=1e-4), CosineAnnealing, MSE loss. VRAM: ~1.5GB.

---

## Project Structure

```
humanoid_vla/
├── README.md                          # This file
├── CLAUDE.md                          # Project vision & phase plan
│
├── sim/                               # MuJoCo simulation
│   ├── g1_with_camera.xml             # Scene: G1 + table + objects + cameras
│   ├── models/g1_29dof.xml            # Robot model (29 torque-actuated DOF)
│   └── test_g1.py                     # Standalone sim test
│
├── scripts/                           # Training & evaluation pipeline
│   ├── act_model.py                   # ACT policy architecture + dataset
│   ├── train_act.py                   # Single-arm training
│   ├── train_bimanual.py              # Bimanual training
│   ├── evaluate.py                    # Single-arm evaluation
│   ├── evaluate_bimanual.py           # Bimanual evaluation (contact + lift)
│   ├── generate_demos.py              # Single-arm scripted expert (IK + weld)
│   ├── generate_bimanual_demos.py     # Bimanual demo generator (friction)
│   ├── physics_sim.py                 # Physics wrapper (mj_step, PD, contacts)
│   ├── domain_randomization.py        # Runtime visual domain randomization
│   ├── eval_generalization.py         # OOD generalization evaluation
│   ├── visualize_configs.py           # Render randomization grid
│   ├── visualize_perception_action.py # Trajectory strip visualization
│   ├── live_demo.py                   # Interactive viewer (single-arm)
│   ├── live_bimanual.py               # Interactive viewer (bimanual)
│   ├── record_demo_videos.py          # Generate demo clips for README
│   ├── visualize_demo.py              # Render videos from HDF5 demos
│   └── convert_to_lerobot.py          # LeRobot format converter
│
├── ros2_ws/src/vla_mujoco_bridge/     # ROS2 package
│   ├── vla_mujoco_bridge/
│   │   ├── task_manager_node.py       # VLA Task Manager (NL → ACT → MuJoCo)
│   │   ├── bridge_node.py             # Low-level MuJoCo ↔ ROS2 bridge
│   │   ├── mujoco_sim.py              # Physics engine wrapper
│   │   ├── teleop_node.py             # Full-body keyboard teleop
│   │   ├── arm_teleop_node.py         # Arm-only keyboard teleop
│   │   └── demo_recorder.py           # HDF5 demonstration recorder
│   └── launch/
│       └── vla_system.launch.py       # Launch: rosbridge + task manager
│
├── media/                             # Demo videos (committed to repo)
│   ├── reach.mp4, grasp.mp4           # Individual task demos
│   ├── pick.mp4, place.mp4            # Pick and place demos
│   ├── bimanual.mp4                   # Bimanual box lift demo
│   └── all_tasks.mp4                  # Combined montage
│
├── data/                              # Generated data (gitignored)
│   ├── demos/                         # Single-arm HDF5 episodes
│   ├── checkpoints/                   # Single-arm model weights
│   ├── bimanual_demos/                # Bimanual HDF5 episodes
│   └── bimanual_checkpoints/          # Bimanual model weights
│
├── study/                             # Deep-dive study documents
│   ├── 01_project_deep_dive.md        # MuJoCo, G1, ROS2, camera pipeline
│   ├── 02_scripted_expert_demo_generation.md  # IK, kinematic playback
│   ├── 03_act_training_and_evaluation.md      # ACT training, debugging
│   ├── 04_bimanual_physics_grasping.md        # Physics, PD, friction grasp
│   └── 05_system_integration.md       # Task Manager, rosbridge, NL parsing
│
├── tasks/                             # Project management
│   ├── todo.md                        # Phase tracker with milestones
│   └── lessons.md                     # Engineering lessons (L001-L050)
│
└── logs/                              # Training logs
    └── act_training_300ep.log
```

---

## Documentation

### Study Documents (Deep Dives)

| # | Document | Topics |
|---|----------|--------|
| 01 | [Project Deep Dive](study/01_project_deep_dive.md) | MuJoCo fundamentals, G1 robot, MJCF XML, PD control, gravity comp, ROS2 bridge, threading, camera pipeline, teleoperation, HDF5 format |
| 02 | [Scripted Expert Demos](study/02_scripted_expert_demo_generation.md) | Inverse kinematics (iterative Jacobian), kinematic playback, weld constraint, trajectory design |
| 03 | [ACT Training & Evaluation](study/03_act_training_and_evaluation.md) | ACT architecture, action chunking, ResNet18 encoder, task embedding, Transformer decoder, training, evaluation debugging |
| 04 | [Bimanual Physics Grasping](study/04_bimanual_physics_grasping.md) | mj_step vs mj_forward, PD torque control, contact physics, friction cones, compliance grasping, bimanual coordination |
| 05 | [System Integration](study/05_system_integration.md) | ROS2 Task Manager, NL parsing, rosbridge WebSocket, thread-safe execution, temporal ensembling, full data flow |
| 06 | [Domain Randomization](study/06_domain_randomization_and_generalization.md) | Visual augmentation, position noise, posture variation, generalization evaluation, ablation study |

### Engineering Lessons

[tasks/lessons.md](tasks/lessons.md) — 50 concise lessons learned:
- L001–L008: Environment setup (torque actuators, meshdir, ROS2 Jazzy)
- L009–L012: Phase B infrastructure (gravity comp, setuptools, cv_bridge)
- L013–L016: Demo generation (ctrlrange, arm reach, kinematic IK, weld)
- L017–L023: ACT training (standalone, action chunking, frozen ResNet, auto-grasp)
- L024–L027: Evaluation (temporal ensembling, hierarchical decomposition, re-grasp)
- L028–L033: Bimanual physics (leg drift, palm pad, IK, compliance grasping)
- L034–L040: ROS2 integration (String+JSON, daemon threads, launch files)
- L041–L050: Domain randomization (memorization, IK validation, progressive difficulty)

---

## Development Phases

| Phase | Status | Duration | Summary |
|-------|--------|----------|---------|
| **A** — Sim + ROS2 | ✅ | 2 weeks | MuJoCo + G1 + camera + ROS2 bridge + teleop |
| **B** — Demo Generation | ✅ | 1 week | Scripted expert demos, IK pipeline, 80 episodes |
| **C** — ACT Training | ✅ | 2 weeks | 4-task ACT model, 86.2% success rate |
| **C2** — Bimanual | ✅ | 2 weeks | Physics-based bimanual grasping, 100% success |
| **D** — Integration | ✅ | 1 week | ROS2 Task Manager, NL commands, rosbridge |
| **E** — Polish | ✅ | 1 week | Demo videos, documentation, study docs |
| **F** — Generalization | ✅ | 1 week | Domain randomization, OOD evaluation, 90% in-dist / 55% OOD |

---

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| GPU | NVIDIA with CUDA, 4GB+ VRAM | RTX 4050 Laptop (6GB) |
| RAM | 16 GB | 33 GB |
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 24.04 |
| CUDA | 12.x | 12.8 |
| ROS2 | Humble or Jazzy | Jazzy |

---

## Key References

### Papers
- **ACT:** Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023
- **GR00T N1:** NVIDIA, "An Open Foundation Model for Humanoid Robots", 2025

### Repositories
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Robot models
- [lerobot](https://github.com/huggingface/lerobot) — Robot learning framework

---

## License

MIT
