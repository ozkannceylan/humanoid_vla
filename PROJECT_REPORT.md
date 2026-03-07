# Project Report: VLA-Controlled Humanoid Robot

**Vision-Language-Action Model for Autonomous Humanoid Manipulation in Simulation**

**Author:** Ozkan Ceylan
**Date:** March 2026
**Hardware:** NVIDIA RTX 4050 (6GB VRAM), Ubuntu 24.04

---

## 1. Introduction

This project implements a complete **Vision-Language-Action (VLA)** pipeline for a simulated humanoid robot. The system takes natural language commands (e.g., *"pick up the red cube"*) and autonomously executes them through a closed-loop neural controller that processes egocentric camera images in real time.

The core idea is simple: **a single neural network sees what the robot sees, understands what the user wants, and directly outputs joint motor commands at 30Hz.** No hand-crafted motion planning, no separate perception module, no explicit state estimation. The model learns end-to-end from demonstration data.

### What Makes This a VLA System

| Component | Implementation | Role |
|-----------|---------------|------|
| **Vision** | ResNet18 backbone processing 480x640 RGB from ego camera | The robot's only perception of the world |
| **Language** | Task embedding layer (4 tasks + bimanual) | Maps NL commands to behavior-conditioning vectors |
| **Action** | Transformer decoder outputting 20-step joint position chunks | Direct motor control at 30Hz |

The ego camera image is the **sole visual input** — the model has no access to object positions, joint states beyond proprioception, or scene geometry. It must learn to visually locate objects, plan approach trajectories, and execute grasps entirely from pixel observations.

---

## 2. Technology Stack

### 2.1 Simulation

| Technology | Purpose |
|------------|---------|
| **MuJoCo 3.x** | Physics engine — contact dynamics, friction, gravity at 500Hz |
| **Unitree G1** | 29-DOF humanoid robot model (MJCF format) |
| **Custom scene** | Table, objects, ego camera, scene camera (`sim/g1_with_camera.xml`) |

The G1 robot is mounted with a fixed base (pelvis frozen at z=0.793m) for tabletop manipulation. This is standard practice in imitation learning research — locomotion and manipulation are separate problems.

**Physics modes:**
- **Kinematic** (`mj_forward`): Used for single-arm tasks where contact forces don't matter (weld-based grasping)
- **Dynamic** (`mj_step`): Used for bimanual tasks where real friction-based grasping requires accurate contact simulation

### 2.2 Neural Network

| Technology | Purpose |
|------------|---------|
| **PyTorch 2.x** | Training framework |
| **ACT (Action Chunking with Transformers)** | VLA policy architecture (Zhao et al., RSS 2023) |
| **ResNet18** | Pretrained visual encoder (ImageNet, layers 0-6 frozen) |
| **Transformer Decoder** | Action prediction (4 layers, 4 heads, d=256) |
| **torchvision** | Training-time image augmentation |

### 2.3 Robotics Infrastructure

| Technology | Purpose |
|------------|---------|
| **ROS2 Jazzy** | Communication middleware between nodes |
| **rosbridge_server** | WebSocket interface (port 9090) for external systems |
| **Custom Task Manager** | NL parsing + VLA inference orchestration |

### 2.4 Data Pipeline

| Technology | Purpose |
|------------|---------|
| **HDF5 (h5py)** | Episode storage (observations, actions, metadata) |
| **OpenCV** | Image resizing for training (480x640 -> 224x224) |
| **NumPy** | Trajectory generation, IK computation |

---

## 3. System Architecture

### 3.1 End-to-End Data Flow

```
                    Natural Language Command
                    "pick up the red cube"
                              |
                              v
                 +------------------------+
                 |    NL Task Parser      |
                 |  Keywords -> task_id   |
                 |  "pick" -> task_id=2   |
                 +------------------------+
                              |
              +===============v================+
              ||     30 Hz VLA Control Loop   ||
              ||                              ||
              ||  +--------+    +---------+   ||
              ||  |  Ego   | -> | ResNet18| --+|
              ||  | Camera |    | Encoder |   ||
              ||  |480x640 |    |  512-d  |   ||
              ||  +--------+    +---------+   ||
              ||                     |        ||
              ||  +--------+    +----v----+   ||
              ||  | Joint  | -> |  State  | --+|
              ||  | State  |    |   MLP   |   ||   +---> Memory
              ||  |28/58-d |    |  256-d  |   ||   |     Tokens
              ||  +--------+    +---------+   ||   |
              ||                              ||   |
              ||  +--------+    +---------+   ||   |
              ||  |  Task  | -> |Embedding| --+----+
              ||  |  "pick"|    |  256-d  |   ||
              ||  +--------+    +---------+   ||
              ||                              ||
              ||         +------------+       ||
              ||         | Transformer|       ||
              ||         |  Decoder   |       ||
              ||         | 4L, 4H    |       ||
              ||         | 20 queries |       ||
              ||         +------+-----+       ||
              ||                |             ||
              ||     Action Chunk (20, 14/29) ||
              ||         |                    ||
              ||    Temporal Ensembling       ||
              ||    (overlapping chunks)      ||
              ||         |                    ||
              +========= | ==================+
                         v
              +------------------------+
              |  MuJoCo Physics Engine |
              |  PD Control -> Torques |
              |  500 Hz substeps       |
              +------------------------+
              |  Robot executes action |
              |  Camera renders next   |
              |  frame -> loop back    |
              +------------------------+
```

### 3.2 The Perception-Action Loop

This is the critical insight of VLA: **every 33ms (30Hz), the robot captures what it sees, feeds it through the neural network, and gets back motor commands.** There is no separate "perception" step followed by a "planning" step — it's a single forward pass.

**What the model receives:**
1. **Image** (3, 224, 224): Ego camera view, ImageNet-normalized. This is what the robot "sees"
2. **State** (28 or 58): Joint positions + velocities. This is proprioception — the robot "feeling" its own body
3. **Task ID** (integer): Which behavior to execute. This is the language conditioning

**What the model outputs:**
- **Action chunk** (20, 14 or 29): Joint position targets for the next 20 timesteps (0.67 seconds)

**Temporal ensembling** then blends overlapping predictions from consecutive forward passes, executing only 5 steps before re-querying the model. This creates smooth, corrective behavior — the model constantly adjusts based on new visual observations.

### 3.3 Why Ego Camera Matters

The ego camera is mounted on the robot's torso, providing a first-person perspective. This means:

- **The visual input changes as the robot moves** — arms entering/leaving the field of view, object growing larger during approach
- **No privileged information** — the model doesn't know the 3D position of objects, only pixel patterns
- **Generalization requires visual understanding** — the model must learn that "red blob at coordinates (x,y) in the image" corresponds to "cube at distance d"

This is fundamentally different from state-based control where the model receives `(cube_x, cube_y, cube_z)` as numbers.

---

## 4. Robot Model: Unitree G1

### 4.1 Physical Configuration

| Property | Value |
|----------|-------|
| Total DOF | 29 torque-controlled hinge joints |
| Left arm | 7 DOF: shoulder (3) + elbow (1) + wrist (3) |
| Right arm | 7 DOF: mirror configuration |
| Legs | 12 DOF (frozen for fixed-base manipulation) |
| Waist | 3 DOF: pitch/roll/yaw (frozen) |
| Reach | ~0.51m from shoulder (workspace <0.40m) |
| Camera | Egocentric RGB, 480x640, torso-mounted |

### 4.2 Control Method

The G1 uses **torque actuators** (not position servos). All control goes through a PD controller:

```
tau = Kp * (q_desired - q_current) - Kd * q_velocity + gravity_compensation
```

- **Kp** = 40 (shoulders), 10 (wrists) — stiffer shoulders for trajectory tracking, compliant wrists for gentle object interaction
- **Kd** = 4 (shoulders), 1 (wrists) — damping for stability
- **Gravity compensation** = `data.qfrc_bias[6:35]` — counteracts gravity on all 29 joints

### 4.3 Grasping Mechanisms

**Single-arm (kinematic):** Weld constraints attach the object to the hand when proximity < 4cm. Simple and deterministic.

**Bimanual (physics):** No weld constraints. Both hands have 8x6x2cm rubber palm pads (friction=1.5, condim=4). The PD controller targets positions *inside* the box surface, creating steady bilateral squeeze force of ~13N per palm. The object is held purely by friction — a more realistic and challenging approach.

---

## 5. ACT Model Architecture

### 5.1 Overview

ACT (Action Chunking with Transformers) predicts **chunks of future actions** rather than single-step outputs. This provides temporal coherence and reduces compounding errors in closed-loop execution.

```
                     Input Modalities
                    /       |        \
              Image(3,224,224)  State(58)  Task("pick")
                |           |           |
           ResNet18      2-layer MLP   Embedding
           (frozen 0-6)  (58->256->256) lookup
                |           |           |
           Linear(512->256) |           |
                \           |          /
                 +-----+----+----+----+
                       |
                  3 Memory Tokens
                       |
              Transformer Decoder
              (4 layers, 4 heads)
              (20 learnable queries)
                       |
                  Linear Head
                  (256 -> action_dim)
                       |
              Action Chunk (20, action_dim)
```

### 5.2 Parameters

| Component | Params | Trainable |
|-----------|--------|-----------|
| ResNet18 (layers 0-6) | 3.8M | Frozen |
| ResNet18 (layer 4) | 4.2M | Yes |
| Image projection | 131K | Yes |
| State MLP | 66K | Yes |
| Task embedding | 1K | Yes |
| Transformer decoder | 3.4M | Yes |
| Action head | 5.1K | Yes |
| Query embeddings | 5.1K | Yes |
| **Total** | **15.6M** | **12.8M trainable** |

### 5.3 Why Freeze ResNet Layers 0-6?

With only ~9,000 training samples (single-arm) or ~5,000 (bimanual), fine-tuning all of ResNet18 would overfit on the low-level visual features. Layers 0-6 detect edges, textures, and colors — these transfer perfectly from ImageNet. Only layer 4 (high-level spatial features) is fine-tuned to adapt to MuJoCo's rendering style.

### 5.4 No CVAE

The original ACT paper uses a Conditional VAE to handle multimodal action distributions (different humans demonstrate the same task differently). Our scripted expert demos are deterministic — there's only one trajectory per initial state — so the CVAE is unnecessary complexity. We use a simple deterministic decoder with MSE loss.

---

## 6. Training Pipeline

### 6.1 Demo Generation

Demonstrations are generated by a scripted expert (not human teleoperation):

**Single-arm:** Iterative Jacobian IK computes joint waypoints for each task phase (approach, grasp, lift, place). Trajectories are played back in kinematic mode with `mj_forward`.

**Bimanual:** IK plans synchronized left/right arm waypoints. Trajectories are executed in full physics (`mj_step`) with PD torque control. The expert must produce enough squeeze force for friction-based grasping.

**Data format (HDF5 per episode):**
```
episode_0042.hdf5
  obs/joint_positions:   (T, 14)  float32  — both arms
  obs/joint_velocities:  (T, 14)  float32
  obs/camera_frames:     (T, 480, 640, 3)  uint8  — ego camera
  action:                (T, 14)  float32  — joint position targets
  [attrs] success:       bool
  [attrs] task_description: str
  [attrs] lift_cm:       float
```

### 6.2 Training Configuration

| Setting | Single-Arm | Bimanual (Phase F) |
|---------|------------|-------------------|
| Demos | 80 (20/task) | 108 (filtered from 120) |
| Samples | 9,000 | 18,108 |
| Epochs | 300 | 300 |
| Batch size | 32 | 32 |
| Learning rate | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Optimizer | AdamW (wd=1e-4) | AdamW (wd=1e-4) |
| Augmentation | ColorJitter, Blur, Crop | ColorJitter, Blur, Crop |
| GPU time | 93 min | 423 min |
| Final loss | 0.000009 | 0.000011 |

### 6.3 Image Augmentation (Training-Side)

Applied to every sample during training at zero simulation cost:

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| ColorJitter | brightness=0.3, contrast=0.3, sat=0.2, hue=0.05 | Robustness to lighting/color changes |
| GaussianBlur | kernel=5, sigma=(0.1, 2.0), p=0.3 | Robustness to focus/blur |
| RandomResizedCrop | scale=(0.85, 1.0), p=0.5 | Robustness to slight camera shifts |

### 6.4 Loss Curve

```
Loss
0.015 |*
      |
0.010 | *
      |
0.005 |  *
      |   *
0.001 |    **
      |      ****
      |          ********
0.000 |                  ********************************
      +-------------------------------------------------> Epoch
      0    30   60   90  120  150  180  210  240  270  300
```

The loss drops 1,000x over 300 epochs. Most learning happens in the first 100 epochs; the cosine LR schedule allows fine convergence in the final 100.

---

## 7. Inference Techniques

Training a good model is only half the story. Deploying it in a closed-loop requires three critical inference techniques:

### 7.1 Temporal Ensembling

The model predicts 20 future actions, but we only execute 5 before re-querying. This means at any timestep, we have predictions from multiple overlapping chunks. We blend them with exponential decay:

```
weight[j] = exp(-0.01 * j)    // j = offset into chunk
action[t] = weighted_average(all chunks covering timestep t)
```

Newer predictions get higher weight. This produces smoother trajectories and corrects drift from earlier predictions. **Without temporal ensembling: 0% success. With it: 86-100%.**

### 7.2 Hierarchical Task Decomposition

Composite tasks (pick, place) have distinct phases: approach vs. manipulation. A single task embedding averages training samples from both phases, producing conflicted behavior (e.g., moving up and forward simultaneously).

**Solution:** Split execution into phases. Phase 1 uses the "grasp" embedding (approach only). When auto-grasp triggers (hand < 4cm from object), switch to the "pick" embedding (lift behavior). Reset ensembling buffers at the transition.

### 7.3 Auto-Grasp with Re-Grasp Prevention

The model predicts only joint positions — no explicit grasp command. Grasping is triggered by proximity:
- **Grasp:** When hand < 4cm from cube, activate weld constraint
- **Release:** For place tasks, release weld when conditions met (near target + delay)
- **Re-grasp prevention:** A `released` flag prevents the proximity check from re-triggering after intentional release

---

## 8. Domain Randomization (Phase F)

### 8.1 The Memorization Problem

The initial system (Phases A-E) achieved 86.2% single-arm and 100% bimanual success. But these numbers were misleading — the model **memorized trajectories** rather than learning visual-motor policies because:

1. Box position varied only +/-3cm — always at the same pixel location in ego camera
2. Robot always started at the same joint configuration
3. No visual diversity — same table color, same lighting, same empty scene

### 8.2 Domain Randomization Strategy

**Goal:** Force the model to actually look at the image to find the object, rather than memorizing pixel coordinates.

| Randomization | Range | Effect on Ego Camera |
|---------------|-------|---------------------|
| Box position (x, y) | +/-5cm, +/-4cm | Object appears at different pixel locations |
| Arm starting posture | spread=0.15 | Arms start in different configurations |
| Table color | Brown/grey/white | Background changes |
| Floor color | +/-20% brightness | Scene context varies |
| Lighting direction | +/-45 degrees | Shadows shift |
| Lighting intensity | +/-30% | Brightness changes |
| Object color | +/-30% brightness | Target appearance varies |
| Distractor objects | 3 geoms show/hide | Scene complexity |
| Camera pose | +/-2cm pos, +/-3 deg rot | Viewpoint jitter |

### 8.3 Expert Robustness Improvements

Wider randomization initially crashed the scripted expert's success rate to ~50%. Three fixes brought it to **90%**:

1. **IK validation:** Check return values — if any waypoint IK fails, skip and retry instead of silently using the bad solution
2. **Reachability pre-check:** `random_arm_start` now rejects configs where hand-to-box distance > 0.38m
3. **Moderate randomization:** Start with +/-5cm (not +/-8cm) — within the expert's reliable range

### 8.4 Generalization Results

Model trained on 108 successful demos with moderate randomization + domain rand:

| Test Distribution | Success Rate | Description |
|-------------------|-------------|-------------|
| **In-distribution** | **90%** (18/20) | Same noise range as training |
| **OOD Position** | **80%** (16/20) | 1.5x wider box positions |
| **OOD Visual** | **70%** (14/20) | Novel colors/lighting never seen in training |
| **OOD Posture** | **60%** (12/20) | 1.5x wider starting arm configs |
| **OOD Combined** | **55%** (11/20) | All OOD factors simultaneously |

The graceful degradation from 90% to 55% shows the model is **genuinely generalizing** — it handles unseen conditions at reduced but meaningful success rates, rather than catastrophically failing on anything outside the training distribution.

---

## 9. ROS2 Integration

### 9.1 System Topology

```
+-------------------+     WebSocket:9090     +------------------+
|  External Client  | <-------------------> | rosbridge_server |
|  (Telegram/Web)   |                       +--------+---------+
+-------------------+                                |
                                                     | ROS2 topics
                                            +--------v---------+
                                            | VLA Task Manager |
                                            |                  |
                                            | 1. Parse NL cmd  |
                                            | 2. Load ACT model|
                                            | 3. Run 30Hz loop |
                                            | 4. Report status |
                                            +--------+---------+
                                                     |
                                            +--------v---------+
                                            | MuJoCo Simulation|
                                            | (embedded in     |
                                            |  Task Manager)   |
                                            +------------------+
```

### 9.2 Natural Language Interface

The Task Manager parses natural language commands into (mode, task_label) pairs:

| User says | Parsed as | Model used |
|-----------|-----------|------------|
| "pick up the red cube" | single_arm, task_id=2 | Single-arm ACT |
| "reach the cube" | single_arm, task_id=0 | Single-arm ACT |
| "place it on the plate" | single_arm, task_id=3 | Single-arm ACT |
| "lift the green box" | bimanual, task_id=0 | Bimanual ACT |
| "pick up with both hands" | bimanual, task_id=0 | Bimanual ACT |

### 9.3 ROS2 Topics

| Topic | Type | Hz | Purpose |
|-------|------|-----|---------|
| `/vla/task_goal` | String | on-demand | NL command input |
| `/vla/status` | String (JSON) | 30 | Step count, progress, result |
| `/camera/image_raw` | Image | 30 | Ego camera feed during execution |

---

## 10. Evaluation Results Summary

### 10.1 Single-Arm Manipulation

| Task | Success | Rate | Key Challenge |
|------|---------|------|---------------|
| Reach | 20/20 | **100%** | Pure trajectory following |
| Grasp | 18/20 | **90%** | Precise approach + proximity trigger |
| Pick up | 18/20 | **90%** | Hierarchical: approach then lift |
| Place | 13/20 | **65%** | Lateral precision + release timing |
| **Overall** | **69/80** | **86.2%** | |

### 10.2 Bimanual Grasping (Baseline)

| Metric | Value |
|--------|-------|
| Success rate | **20/20 (100%)** |
| Mean lift | 8.5cm |
| Contact force | L=13.6N, R=13.3N |
| Training | 30 demos, 300 epochs, 52 min |

### 10.3 Bimanual Generalization (Phase F)

| Distribution | Success | Avg Lift |
|-------------|---------|----------|
| In-distribution | 90% | 9.0cm |
| OOD Position (1.5x) | 80% | 7.8cm |
| OOD Visual | 70% | 8.4cm |
| OOD Posture (1.5x) | 60% | 5.7cm |
| OOD Combined | 55% | -4.4cm |

### 10.4 Combined: All Tasks

**Total: 89/100 tasks (89%)** across single-arm + bimanual (baseline evaluation).

---

## 11. Code Structure

```
humanoid_vla/
|
+-- sim/                              # Simulation environment
|   +-- g1_with_camera.xml            # MuJoCo scene (table, objects, cameras)
|   +-- models/g1_29dof.xml           # G1 robot (29 DOF + palm pads + ego cam)
|
+-- scripts/                          # Core ML pipeline
|   +-- act_model.py                  # ACT architecture + dataset classes
|   +-- train_act.py                  # Single-arm training loop
|   +-- train_bimanual.py             # Bimanual training (--filter-success)
|   +-- evaluate.py                   # Single-arm eval (temporal ensembling)
|   +-- evaluate_bimanual.py          # Bimanual eval (contact + lift criteria)
|   +-- eval_generalization.py        # OOD generalization framework
|   +-- generate_demos.py             # Single-arm scripted expert (IK + weld)
|   +-- generate_bimanual_demos.py    # Bimanual expert (IK validation + retry)
|   +-- physics_sim.py                # MuJoCo wrapper (PD control, IK, contacts)
|   +-- domain_randomization.py       # Runtime visual randomization
|   +-- live_demo.py                  # Interactive MuJoCo viewer (single-arm)
|   +-- live_bimanual.py              # Interactive MuJoCo viewer (bimanual)
|   +-- record_demo_videos.py         # MP4 generation for demos
|   +-- visualize_configs.py          # Render domain randomization grid
|   +-- visualize_perception_action.py # Trajectory strip visualization
|
+-- ros2_ws/src/vla_mujoco_bridge/    # ROS2 package
|   +-- task_manager_node.py          # NL -> ACT inference -> MuJoCo
|   +-- bridge_node.py                # Low-level MuJoCo <-> ROS2
|   +-- mujoco_sim.py                 # Physics engine wrapper
|   +-- teleop_node.py                # Full-body keyboard control
|   +-- arm_teleop_node.py            # Arm-only keyboard control
|   +-- demo_recorder.py              # HDF5 recording node
|   +-- launch/vla_system.launch.py   # rosbridge + task manager
|
+-- data/                             # Generated (gitignored)
|   +-- demos/                        # 80 single-arm HDF5 episodes
|   +-- checkpoints/                  # Single-arm ACT weights
|   +-- bimanual_demos_phase_f2/      # 120 bimanual episodes (108 success)
|   +-- bimanual_checkpoints_phase_f2/# Bimanual ACT weights (Phase F)
|
+-- media/                            # Demo videos (6 MP4 clips)
+-- study/                            # 6 technical deep-dive documents
+-- tasks/                            # todo.md + lessons.md (50 lessons)
```

**Total:** ~11,500 lines of custom code across 54 files.

---

## 12. How to Reproduce

### Full Pipeline

```bash
# 1. Generate demos
MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20
MUJOCO_GL=egl python3 scripts/generate_bimanual_demos.py \
  --episodes 120 --noise-x 0.05 --noise-y 0.04 \
  --random-start 0.15 --domain-rand

# 2. Train
python3 scripts/train_act.py --demos data/demos --epochs 300
python3 scripts/train_bimanual.py \
  --demos data/bimanual_demos_phase_f2 --epochs 300 --filter-success

# 3. Evaluate
MUJOCO_GL=egl python3 scripts/evaluate.py \
  --checkpoint data/checkpoints/best.pt --episodes 20
MUJOCO_GL=egl python3 scripts/eval_generalization.py \
  --checkpoint data/bimanual_checkpoints_phase_f2/best.pt \
  --mode bimanual --episodes 20 \
  --train-noise-x 0.05 --train-noise-y 0.04 --train-random-start 0.15

# 4. Run via natural language (ROS2)
ros2 launch vla_mujoco_bridge vla_system.launch.py
ros2 topic pub --once /vla/task_goal std_msgs/String \
  "data: 'pick up the red cube'"
```

### Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA, 4GB+ VRAM | RTX 4050 (6GB) |
| RAM | 16 GB | 33 GB |
| OS | Ubuntu 22.04/24.04 | Ubuntu 24.04 |
| ROS2 | Humble or Jazzy | Jazzy |

---

## 13. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **ACT over OpenVLA/GR00T** | 15.6M params fits RTX 4050; OpenVLA (7B) too large; GR00T needs cloud GPU |
| **Standalone training over LeRobot** | More control, easier debugging, no format-conversion surprises |
| **Fixed base (no locomotion)** | Standard in imitation learning; balance control is orthogonal to manipulation |
| **Deterministic decoder (no CVAE)** | Scripted demos are deterministic; CVAE adds unnecessary complexity |
| **Kinematic mode for single-arm** | Faster, deterministic; physics not needed when using weld constraints |
| **Physics mode for bimanual** | Friction-based grasping requires real contact forces |
| **Frozen ResNet 0-6** | Prevents overfitting with <10K samples; ImageNet features transfer well |
| **Moderate randomization (+/-5cm)** | Sweet spot: enough diversity for learning, high enough expert success (~90%) |
| **String+JSON for ROS2** | Avoids custom interfaces; compatible with rosbridge |

---

## 14. Lessons Learned

50 engineering lessons documented in `tasks/lessons.md`. Key highlights:

1. **Temporal ensembling is non-negotiable** (L024) — Without it, 0% success even with loss=0.000009. Low training loss means nothing for closed-loop rollout.

2. **Never train on failed demos** (L046) — 46% garbage data made the model perform WORSE than baseline. Always filter by success attribute.

3. **IK failures cascade** (L047) — Ignoring solver return values causes error propagation through waypoints. Always validate and retry.

4. **Start randomization small** (L050) — Jumping from +/-2cm to +/-8cm crashed expert success to 50%. Incremental increases let you find the sweet spot.

5. **Compliance grasping via PD penetration** (L033) — Targeting inside the object surface creates steady squeeze force. No force controller needed.

---

## 15. Future Work

| Direction | Description | Requirement |
|-----------|-------------|-------------|
| **GR00T N1 integration** | Purpose-built VLA for humanoids, diffusion-based action generation | Cloud GPU for fine-tuning |
| **Wrist camera** | Dual-view (torso + wrist) for close-up manipulation | Architecture change (2 ResNet inputs) |
| **Progressive randomization** | Push from +/-5cm to +/-10cm as model improves | Iterative train-eval cycles |
| **More tasks** | Pour, stack, sort — expand to 10+ manipulation skills | More expert demos per task |
| **Locomotion** | Whole-body balance for mobile manipulation | Separate RL policy + coordination |

---

## 16. References

### Papers
- **ACT:** Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023
- **GR00T N1:** NVIDIA, "An Open Foundation Model for Humanoid Robots", 2025
- **Domain Randomization:** Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017

### Repositories
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Robot models
- [lerobot](https://github.com/huggingface/lerobot) — Robot learning framework

---

*This project demonstrates a complete VLA pipeline — from egocentric vision to natural language understanding to motor control — running entirely in simulation on a consumer GPU.*
