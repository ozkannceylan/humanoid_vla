# Humanoid VLA — Vision-Language-Action Controlled Humanoid Robot

A simulated **Unitree G1 humanoid robot** controlled by a Vision-Language-Action (VLA) model, commandable via natural language. The robot sees through an egocentric camera, understands task commands like *"pick up the red cube"*, and generates joint-level motor commands through a trained **ACT (Action Chunking with Transformers)** model.

> **Status:** Simulation-only (MuJoCo) — Phases A–C complete  
> **Author:** Özkan Ceylan

---

## Architecture

```
User: "Pick up the red cube and place it on the blue plate"
  │
  ▼
┌─────────────────────────────────────────────┐
│  VLA Task Manager                           │
│  Camera image + language → ACT model → 29   │
│  joint actions @ 30Hz                       │
└──────────────────────┬──────────────────────┘
                       │ Joint commands
┌──────────────────────▼──────────────────────┐
│  MuJoCo Simulation                          │
│  Unitree G1 (29 DOF) + table + objects      │
│  Egocentric camera → 480×640 RGB            │
└─────────────────────────────────────────────┘
```

The ACT model runs in a tight control loop: each frame, it receives a camera image + joint state + task instruction, and predicts the next 20 joint configurations (action chunking). Only the first action is executed, then the loop repeats.

---

## Quick Start

### Prerequisites

- **Ubuntu 24.04** (tested), or Ubuntu 22.04
- **NVIDIA GPU** with CUDA (RTX 4050 6GB VRAM is sufficient)
- **Python 3.12+**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ozkanceylan/humanoid_vla.git
cd humanoid_vla

# 2. Install ROS2 (Jazzy on 24.04, Humble on 22.04)
chmod +x install_ros2.sh && ./install_ros2.sh

# 3. Install Python dependencies
pip3 install --break-system-packages \
  mujoco opencv-python numpy h5py torch torchvision pynput

# 4. Clone robot model repositories
cd repos
git clone https://github.com/unitreerobotics/unitree_mujoco
git clone https://github.com/google-deepmind/mujoco_menagerie
cd ..

# 5. Build ROS2 workspace
source /opt/ros/jazzy/setup.bash   # or humble
cd ros2_ws && colcon build --symlink-install && cd ..
source ros2_ws/install/setup.bash
```

### Generate Training Data

```bash
# Generate 80 scripted expert demos (20 per task: reach, grasp, pick, place)
MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20
```

### Train the ACT Model

```bash
# Train for 300 epochs (~84 min on RTX 4050)
python3 scripts/train_act.py --demos data/demos --epochs 300 --batch-size 32
```

### Evaluate

```bash
# Run evaluation (20 episodes per task)
MUJOCO_GL=egl python3 scripts/evaluate.py --checkpoint data/checkpoints/best.pt --episodes 20
```

---

## Evaluation Results

Trained for 300 epochs (93 min on RTX 4050, final loss: 0.000009). Evaluated with temporal ensembling and hierarchical task decomposition:

| Task | Success | Rate |
|------|---------|------|
| **Reach** the red cube | 20/20 | **100%** |
| **Grasp** the red cube | 18/20 | **90%** |
| **Pick up** the red cube | 18/20 | **90%** |
| **Place** the red cube on the blue plate | 13/20 | **65%** |
| **Overall** | **69/80** | **86.2%** |

Key inference techniques that bridge the train→eval gap:
1. **Temporal ensembling** — re-plan every 5 steps, exponentially-weighted average of overlapping action chunks
2. **Hierarchical task decomposition** — composite tasks use "grasp" embedding for approach, then switch to task-specific embedding
3. **Re-grasp prevention** — `released` flag prevents auto-grasp from re-triggering after intentional release

> Training loss is meaningless for closed-loop evaluation. See [study/03](study/03_act_training_and_evaluation.md) §9 for the full debugging story.

---

## Project Structure

```
humanoid_vla/
├── README.md                          # ← You are here
├── CLAUDE.md                          # Project vision & phase plan
│
├── sim/                               # MuJoCo simulation
│   ├── g1_with_camera.xml             # Scene: G1 + table + cube + place marker
│   ├── models/g1_29dof.xml            # Robot model (29 torque-actuated DOF)
│   └── test_g1.py                     # Standalone sim test (viewer + camera)
│
├── scripts/                           # Training & evaluation pipeline
│   ├── act_model.py                   # ACT policy architecture + dataset
│   ├── train_act.py                   # Standalone training loop
│   ├── evaluate.py                    # Simulation evaluation with success detection
│   ├── generate_demos.py              # Scripted expert demo generator (IK-based)
│   └── convert_to_lerobot.py          # LeRobot format converter (optional)
│
├── ros2_ws/src/vla_mujoco_bridge/     # ROS2 package
│   └── vla_mujoco_bridge/
│       ├── mujoco_sim.py              # Physics engine wrapper + PD controller
│       ├── bridge_node.py             # ROS2 ↔ MuJoCo bridge (topics/services)
│       ├── teleop_node.py             # Full-body keyboard teleop
│       ├── arm_teleop_node.py         # Arm-only keyboard teleop
│       └── demo_recorder.py           # HDF5 demonstration recorder
│
├── data/                              # Generated data (gitignored)
│   ├── demos/                         # HDF5 episodes (episode_0000.hdf5, ...)
│   └── checkpoints/                   # Model weights (best.pt, latest.pt)
│
├── study/                             # Deep-dive study documents
│   ├── 01_project_deep_dive.md        # Phase A+B architecture & concepts
│   ├── 02_scripted_expert_demo_generation.md  # IK pipeline & kinematic playback
│   └── 03_act_training_and_evaluation.md      # ACT model, training, Phase C
│
├── tasks/                             # Project management
│   ├── todo.md                        # Phase tracker with milestones
│   ├── lessons.md                     # Engineering lessons learned (L001–L023)
│   └── ozkan_todo.md                  # Personal research notes
│
└── logs/                              # Training logs (for documentation)
    └── act_training_300ep.log         # Terminal output from training run
```

---

## The Robot: Unitree G1

| Property | Value |
|----------|-------|
| DOF | 29 torque-controlled joints |
| Actuators | `<motor>` elements (torque input, not position servo) |
| Control | PD controller: $\tau = K_p(q_{des} - q) - K_d\dot{q} + \tau_{gravity}$ |
| Camera | Egocentric RGB, 480×640, mounted on torso |
| Fixed-base | Pelvis frozen at z=0.793m (no balance needed) |
| Right arm | 7 DOF (shoulder pitch/roll/yaw, elbow, wrist pitch/roll/yaw) |

---

## Tasks

The system supports 4 manipulation tasks, each with a natural language label:

| ID | Task | Description | Success Criterion |
|----|------|-------------|-------------------|
| 0 | **Reach** | Move hand to the red cube | Hand within 6cm of cube |
| 1 | **Grasp** | Close hand around the cube | Auto-grasp triggered (hand < 4cm) |
| 2 | **Pick** | Lift the cube off the table | Cube z > 0.90m while grasped |
| 3 | **Place** | Move cube to the blue plate | Cube within 6cm of target, released |

---

## ACT Model Architecture

```
Image (480×640×3)──→ ResNet18 (frozen layers 0-6) ──→ AvgPool ──→ 512-d ──→ Proj ──→ 256-d ─┐
                                                                                                │
State (29 pos + 29 vel) ──→ MLP (58→256→256) ─────────────────────────────────────────────────│──→ Memory (3 tokens)
                                                                                                │
Task label ("pick up...") ──→ Embedding lookup ──→ 256-d ────────────────────────────────────┘
                                                                                  │
                                                                    ┌─────────────▼──────────────┐
                                                                    │  Transformer Decoder        │
                                                                    │  4 layers, 4 heads, d=256   │
                                                                    │  20 learnable query tokens   │
                                                                    └─────────────┬──────────────┘
                                                                                  │
                                                                    Linear(256, 29) × 20 steps
                                                                                  │
                                                                    Action chunk: (20, 29) joint positions
```

| Component | Detail |
|-----------|--------|
| Total params | 15.6M |
| Trainable params | 12.8M (ResNet frozen except layer4) |
| Chunk size | 20 timesteps (~0.67s lookahead) |
| Training | AdamW (lr=1e-4), CosineAnnealing, MSE loss |
| VRAM usage | ~1.5 GB at batch_size=32 |

→ **[Detailed study: ACT Training & Evaluation](study/03_act_training_and_evaluation.md)**

---

## Documentation

### Study Documents (Deep Dives)

These are detailed study guides covering every concept, with code walkthroughs, math derivations, and engineering decisions:

| # | Document | Topics Covered |
|---|----------|----------------|
| 01 | [Project Deep Dive](study/01_project_deep_dive.md) | MuJoCo fundamentals, G1 robot, MJCF XML, PD control, gravity compensation, ROS2 bridge, threading, camera pipeline, teleoperation, HDF5 format |
| 02 | [Scripted Expert Demos](study/02_scripted_expert_demo_generation.md) | Inverse kinematics (iterative Jacobian), kinematic playback, weld constraint enforcement, reach/grasp/pick trajectory design |
| 03 | [ACT Training & Evaluation](study/03_act_training_and_evaluation.md) | ACT architecture, action chunking, ResNet18 visual encoder, task embedding, Transformer decoder, training pipeline, loss curves, evaluation with auto-grasp/release, success metrics |

### Engineering Lessons

[tasks/lessons.md](tasks/lessons.md) — 27 concise lessons learned during development:
- L001–L008: Environment setup (torque actuators, meshdir, ROS2 Jazzy, pip on 24.04)
- L009–L012: Phase B infrastructure (gravity comp, setuptools regression, cv_bridge crash)
- L013–L016: Demo generation (ctrlrange vs jnt_range, arm reach, kinematic IK, manual weld)
- L017–L023: ACT training & Phase C (standalone training, action chunking, frozen ResNet, auto-grasp eval)
- L024–L027: Evaluation (temporal ensembling, hierarchical decomposition, re-grasp prevention, kinematic gravity)

### Task Tracker

[tasks/todo.md](tasks/todo.md) — Detailed milestone tracker for all phases (A through C).

---

## Development Phases

| Phase | Status | Summary |
|-------|--------|---------|
| **A** — Sim + ROS2 | ✅ Complete | MuJoCo + G1 + camera + ROS2 bridge + teleop |
| **B** — VLA Training | ✅ Complete | Fixed-base manipulation, scripted demos, ACT training |
| **C** — Multi-step | ✅ Complete | Place task, 4-task training, evaluation |
| **D** — RosClaw | 🔲 Planned | Telegram integration via RosClaw + OpenClaw |
| **E** — Polish | 🔲 Planned | Multi-task demo, documentation, video |

---

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| GPU | NVIDIA with CUDA, 4GB+ VRAM | RTX 4050 Laptop (6GB) |
| RAM | 16 GB | 33 GB |
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 24.04 |
| CUDA | 12.x | 12.8 |

---

## Key References

### Papers
- **ACT:** Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023
- **GR00T N1:** NVIDIA, "An Open Foundation Model for Humanoid Robots", 2025
- **ACG:** "Action Coherence Guidance for VLA Models", ICRA 2026

### Repositories
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Robot models
- [lerobot](https://github.com/huggingface/lerobot) — Robot learning framework

---

## License

MIT
