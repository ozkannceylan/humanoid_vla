# Final Goal: VLA-Controlled Humanoid Robot with Natural Language Interface

> **Project:** Autonomous humanoid robot controlled by a Vision-Language-Action model, commandable via natural language through RosClaw/OpenClaw  
> **Scope:** Simulation only (no physical hardware)  
> **Author:** Ozkan Ceylan  
> **Status:** Planning

---

## 1. Vision

A simulated humanoid robot that:
1. **Sees** the world through egocentric cameras
2. **Understands** natural language task commands ("pick up the red cup from the table")
3. **Acts** by generating joint-level motor commands through a VLA model
4. **Adapts** via fine-tuning on task-specific demonstrations in simulation
5. **Is controllable** remotely through Telegram via RosClaw + OpenClaw (Charlie)

```
User on Telegram
    │  "Pick up the red cup and place it on the shelf"
    ↓
OpenClaw (Charlie) — AI Agent
    │  Parses intent, dispatches task goal
    ↓
RosClaw — ROS2 Bridge
    │  Sends high-level task goal via action interface
    ↓
VLA Task Manager (ROS2 Node)
    │  Feeds camera frames + language instruction to VLA
    ↓
VLA Model (GR00T N1 / OpenVLA)
    │  Camera image + "pick up red cup" → joint actions (30Hz)
    ↓
Simulated Humanoid (Isaac Lab / MuJoCo)
    │  Executes joint commands, physics simulation
    ↓
Task Complete → Status reported back to Telegram
```

**Critical architectural insight:** The VLA model runs in a tight 10-30Hz control loop (camera → action), while RosClaw/OpenClaw operates at the **task dispatch level** — it sends "pick up the red cup" once and monitors completion. It does NOT participate in the per-frame control loop.

---

## 2. Simulator Selection

### 2.1 Comparison

| Criteria | MuJoCo | Isaac Sim / Isaac Lab | Gazebo Harmonic |
|----------|--------|----------------------|-----------------|
| **Humanoid physics** | Excellent — designed for articulated bodies, fast contact dynamics | Excellent — PhysX GPU-accelerated, parallel envs | Limited — ODE/Bullet, not optimized for humanoids |
| **Speed (single env)** | Very fast (~1μs/step for humanoid) | Slower per-env (~20x overhead vs MuJoCo) but massively parallel | Moderate |
| **Parallel envs** | MuJoCo-XLA (MJX): 1000s of envs on GPU | Native: 4096+ envs on single GPU | Not supported |
| **Camera rendering** | Basic (offscreen rendering, not photorealistic) | Photorealistic (RTX ray tracing, Omniverse) | Basic |
| **VLA ecosystem** | OpenVLA, Octo, ACT — all support MuJoCo (via robosuite) | GR00T N1/N1.6 — native, purpose-built | No VLA support |
| **Humanoid models** | MuJoCo Menagerie: Unitree H1, G1, Atlas, etc. | SimReady: Unitree, Fourier GR-1, 1X Neo | Limited URDF-only |
| **ROS2 integration** | Requires custom bridge (mujoco_ros2) | Native ROS2 support via Isaac Sim | Native |
| **RL training** | Standard for RL research (Gymnasium, dm_control) | Isaac Lab (built on Isaac Sim) — GPU RL | Not designed for RL |
| **Fine-tuning data** | Robosuite, dm_control tasks | Isaac Lab tasks, GR00T-Mimic synthetic data | Not suitable |
| **GPU requirement** | CPU-friendly, GPU optional (MJX for parallel) | RTX 3070+ minimum, RTX 4070+ recommended | Minimal |
| **Your hardware (RTX 4050)** | ✅ Runs perfectly | ⚠️ Tight — Isaac Sim needs ~8GB VRAM, 4050 has 6GB | ✅ Runs fine |
| **Learning curve** | Moderate (MJCF format, Python API) | Steep (Omniverse, USD, Isaac Lab framework) | Already familiar |
| **Open source** | ✅ Apache 2.0 (Google DeepMind) | ✅ Apache 2.0 (but Omniverse deps are closed) | ✅ Open source |
| **RL background fit** | ✅ Perfect — your MSc thesis used similar envs | 🟡 Different paradigm | ❌ Not RL-friendly |

### 2.2 Recommendation

**Primary: MuJoCo** — for the following reasons:

1. **Hardware fit** — Your RTX 4050 (6GB VRAM) is tight for Isaac Sim but perfect for MuJoCo. Isaac Sim recommends 8GB+ VRAM minimum for comfortable use.
2. **RL background** — Your master's thesis was in RL for mobile robotics. MuJoCo is the standard RL simulation platform. The tools, APIs, and paradigms will feel natural.
3. **VLA support** — OpenVLA, Octo, and ACT all use MuJoCo (via robosuite/dm_control) as their primary simulation backend. GR00T N1 also supports MuJoCo evaluation.
4. **Speed** — For single-env development and debugging (which is what you'll do most), MuJoCo is 10-20x faster than Isaac Sim.
5. **Humanoid models** — MuJoCo Menagerie (by Google DeepMind) includes high-quality Unitree H1, G1, and other humanoid MJCF models ready to use.

**Secondary: Isaac Lab** — explore later when:
- You need photorealistic rendering for camera-based VLA training
- You want to use GR00T N1/N1.6 natively (it's the GR00T platform's foundation)
- You get access to a more powerful GPU (cloud or desktop RTX 4090)
- You want to scale to 1000s of parallel environments for RL training

**Not recommended: Gazebo Harmonic** — while you already know it from Zero to Hero, it lacks humanoid physics quality, has no VLA ecosystem, and is not designed for the RL/imitation learning workflows this project requires. Your Gazebo skills are still valuable for ROS2 integration and will transfer to the ROS2 bridge layer.

### 2.3 The Newton Engine (Future)

NVIDIA, Google DeepMind, and Disney Research are jointly developing **Newton** — a new open-source GPU physics engine built on NVIDIA Warp and OpenUSD. It aims to combine MuJoCo's accuracy with Isaac Sim's GPU parallelism. Early benchmarks show MuJoCo-Warp achieving 70-150x speedups for humanoid simulation. This is in beta and will likely become the standard. Worth monitoring but not ready for production use yet.

---

## 3. Humanoid Model Selection

### 3.1 Comparison

| Robot | DOF | Height | Hands | MJCF | URDF | VLA Support | Ecosystem |
|-------|-----|--------|-------|------|------|-------------|-----------|
| **Unitree G1** | 23-37 (config dependent) | 1.27m | Optional dexterous (Dex3) | ✅ Official | ✅ Official | ✅ GR00T N1, LeRobot | Very active — unitree_mujoco, unitree_IL_lerobot, unitree_sim_isaaclab |
| **Unitree H1** | 19 | 1.8m | No dexterous hands | ✅ MuJoCo Menagerie | ✅ Official | ✅ GR00T N1, Humanoid-Gym | Active — locomotion focus, teleoperation demos |
| **Unitree H1-2** | 19+ | 1.8m | Preview | ✅ MJCF available | ✅ Simplified URDF | 🟡 Newer, less ecosystem | Growing |
| **Fourier GR-1** | 40+ | 1.65m | Dexterous | ❌ Not public | ❌ Not public | ✅ GR00T N1 demo robot | Closed ecosystem |
| **1X Neo** | 20+ | 1.6m | Yes | ❌ Not public | ❌ Not public | ✅ GR00T N1 demo robot | Closed ecosystem |
| **MuJoCo Humanoid** | 21 | Generic | Simple grippers | ✅ Built-in | ❌ | ✅ Standard benchmark | Universal — every RL paper |

### 3.2 Recommendation

**Primary: Unitree G1** — for the following reasons:

1. **Best VLA ecosystem** — Unitree provides `unitree_IL_lerobot`, an open-source imitation learning framework based on HuggingFace LeRobot, specifically adapted for G1 with dexterous hands. This is exactly the fine-tuning pipeline you need.
2. **Dexterous manipulation** — G1 with Dex3 hands can perform the manipulation tasks (pick, place, grasp) that your VLA needs to learn. H1 lacks dexterous hands.
3. **Complete toolchain** — Official MuJoCo integration (`unitree_mujoco`), ROS2 support (`unitree_ros2`), Isaac Lab support (`unitree_sim_isaaclab`), and imitation learning (`unitree_IL_lerobot`). Everything is open-source under BSD-3.
4. **Compact size** — At 1.27m, G1 is easier to simulate (faster physics, simpler collision geometry) than the 1.8m H1.
5. **Active community** — Most humanoid RL/VLA research in 2024-2025 uses Unitree robots due to their open-source approach.
6. **GR00T N1 compatible** — When you're ready to try NVIDIA's VLA, G1 is a supported embodiment.

**Secondary: MuJoCo built-in humanoid** — for initial prototyping and learning the MuJoCo API before moving to the more complex G1 model.

**Future: Unitree H1** — if you want to explore locomotion-focused tasks (walking, running, terrain traversal), H1 has better support via Humanoid-Gym.

### 3.3 Available Models and Formats

**Unitree G1 MJCF (for MuJoCo):**
```bash
# Option 1: From unitree_mujoco (official)
git clone https://github.com/unitreerobotics/unitree_mujoco
# Models in: unitree_robots/g1/

# Option 2: From unitree_ros (URDF + MJCF)
git clone https://github.com/unitreerobotics/unitree_ros
# Models in: robots/g1_description/
```

**Unitree H1 MJCF (from MuJoCo Menagerie by Google DeepMind):**
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
# Model in: unitree_h1/
```

---

## 4. VLA Model Selection

### 4.1 Comparison

| Model | Parameters | Architecture | Training Data | Fine-tuning | Humanoid Support | Open Source |
|-------|-----------|-------------|---------------|-------------|-----------------|------------|
| **GR00T N1.6** | 2B+ | VLM (Cosmos-Reason-2B) + DiT (32 layers) | Real robot + synthetic (Isaac Sim) + internet video | ✅ LeRobot-compatible, LoRA | ✅ Purpose-built for humanoids | ✅ Apache 2.0 |
| **OpenVLA** | 7B | Llama 2 + ViT | Open X-Embodiment | ✅ LoRA fine-tuning | 🟡 Mainly manipulators | ✅ MIT |
| **Octo** | 93M | Transformer | Open X-Embodiment | ✅ Fine-tuning API | 🟡 Mainly manipulators | ✅ MIT |
| **π-0** | 3B | PaLI-based VLM + flow matching | Physical Intelligence data | ❌ Not released | ✅ Humanoid demos shown | ❌ Closed |
| **ACT** | ~50M | CVAE + Transformer | Task-specific demos | ✅ Train from scratch | 🟡 Bimanual manipulation | ✅ MIT |

### 4.2 Recommendation

**Start with: ACT (Action Chunking with Transformers)** — for learning the pipeline:
- Smallest model (~50M params), trains on your RTX 4050
- Well-documented, simple architecture
- Proven for bimanual manipulation tasks
- Available in LeRobot framework (same as Unitree's IL pipeline)
- Learn the full cycle: collect demos → train → evaluate → iterate

**Then: GR00T N1.6** — for the final system:
- Purpose-built for humanoid robots
- Dual-system architecture (System 1: fast action, System 2: reasoning)
- Best generalization across tasks
- LeRobot-compatible fine-tuning
- But: needs more GPU (inference: RTX 4050 possible with optimization; training: likely needs cloud GPU)

**Skip: OpenVLA** — 7B parameters is too large for your GPU, and it's designed for tabletop manipulators, not humanoids.

### 4.3 GR00T N1.6 Architecture Deep Dive

```
                    Language Instruction
                    "Pick up the red cup"
                            ↓
                    ┌───────────────────┐
                    │  System 2 (Slow)   │
                    │  Cosmos-Reason-2B  │  ← Vision-Language Model
                    │  VLM               │     Reasons about scene
                    │                    │     Plans action sequence
                    └────────┬──────────┘
                             ↓
              Action plan + visual features
                             ↓
                    ┌───────────────────┐
                    │  System 1 (Fast)   │
                    │  Diffusion         │  ← Diffusion Transformer
                    │  Transformer       │     Generates continuous
                    │  (32 DiT layers)   │     joint actions at 30Hz
                    └────────┬──────────┘
                             ↓
              Joint positions/velocities/torques
                             ↓
                    Humanoid Robot Controller
```

**Dual-system design rationale:**
- System 2 runs at ~1-5Hz (reasoning is slow but strategic)
- System 1 runs at 30Hz (fast reflexive actions, smooth motion)
- System 2 sets the "what" and "why", System 1 handles the "how"

---

## 5. System Architecture

### 5.1 Full Stack

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                  │
│                                                         │
│  Telegram ←→ OpenClaw (Charlie) ←→ RosClaw Plugin       │
│                                                         │
│  Natural language in, status/camera frames out           │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (rosbridge)
┌──────────────────────▼──────────────────────────────────┐
│                    ROS2 ORCHESTRATION LAYER              │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Task Manager │  │   Camera     │  │   Status     │  │
│  │ (Action      │  │   Bridge     │  │   Reporter   │  │
│  │  Server)     │  │ (MuJoCo →   │  │ (joint       │  │
│  │              │  │  sensor_msgs)│  │  states,     │  │
│  │ Receives     │  │              │  │  task        │  │
│  │ NL goals,    │  │ Publishes    │  │  progress)   │  │
│  │ runs VLA     │  │ Image msgs   │  │              │  │
│  │ control loop │  │ at 30Hz      │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                              │
│  ┌──────▼─────────────────▼──────┐                      │
│  │         VLA Inference          │                      │
│  │  Camera frame + language       │                      │
│  │  → GR00T N1 / ACT             │                      │
│  │  → joint action commands       │                      │
│  │  (runs at 10-30Hz)             │                      │
│  └──────────────┬────────────────┘                      │
└─────────────────┼───────────────────────────────────────┘
                  │ Joint commands
┌─────────────────▼───────────────────────────────────────┐
│                    SIMULATION LAYER                      │
│                                                         │
│  MuJoCo Physics Engine                                  │
│  ├── Unitree G1 humanoid model (MJCF)                   │
│  ├── Environment (table, objects, room)                  │
│  ├── Camera sensors (egocentric)                        │
│  ├── Joint state feedback                               │
│  └── Contact/force simulation                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
MuJoCo Camera (30Hz) → Camera Bridge Node → /camera/image_raw
                                                    ↓
Language Goal ("pick up red cup") ──────→ Task Manager Node
                                                    ↓
                                          VLA Model Inference
                                          (camera + language → actions)
                                                    ↓
                                          /joint_commands (30Hz)
                                                    ↓
                                          MuJoCo Joint Controller
                                                    ↓
                                          Physics Step → New Camera Frame
                                          (loop continues)
```

### 5.3 ROS2 Topics & Interfaces

| Topic / Action | Type | Direction | Purpose |
|----------------|------|-----------|---------|
| `/camera/image_raw` | `sensor_msgs/Image` | MuJoCo → ROS2 | Egocentric camera feed |
| `/joint_states` | `sensor_msgs/JointState` | MuJoCo → ROS2 | Current joint positions/velocities |
| `/joint_commands` | `sensor_msgs/JointState` | ROS2 → MuJoCo | Commanded joint positions |
| `/vla/task_goal` | Custom Action | RosClaw → Task Manager | NL task command |
| `/vla/feedback` | Custom Feedback | Task Manager → RosClaw | Task progress |
| `/vla/status` | `std_msgs/String` | Task Manager → Status | Human-readable status |

---

## 6. Fine-Tuning Pipeline

### 6.1 Data Collection Strategy

```
Step 1: Teleoperation in MuJoCo
├── Use keyboard/joystick to control G1 joints
├── Record: camera frames + joint states + actions
├── Save in LeRobot HDF5 format
└── Target: 20-50 demonstrations per task

Step 2: Convert to LeRobot Schema
├── (video, state, action) triplets
├── Language annotations per episode
└── Compatible with GR00T N1 and ACT

Step 3: Fine-tune VLA
├── ACT: Train from scratch (~2-4 hours on RTX 4050)
├── GR00T N1: LoRA fine-tune (~needs cloud GPU for training)
└── Evaluate in simulation

Step 4: Iterate
├── Identify failure modes
├── Collect more demos for hard cases
├── Re-train and evaluate
└── Repeat until task success > 80%
```

### 6.2 Unitree IL LeRobot Framework

Unitree provides an open-source imitation learning framework specifically for G1:

```bash
git clone https://github.com/unitreerobotics/unitree_IL_lerobot
```

This supports:
- **Data collection** — teleoperation with G1 in simulation
- **ACT (Action Chunking with Transformers)** — bimanual manipulation
- **Diffusion Policy (DP)** — alternative to ACT
- **Training** — standard PyTorch training loop
- **Evaluation** — automated rollout in simulation
- **Deployment** — tested on real G1 hardware (for future use)

---

## 7. Development Phases

### Phase A: MuJoCo + G1 Fundamentals (2 weeks)
- Set up MuJoCo environment
- Load and simulate Unitree G1
- Implement camera rendering pipeline
- Build ROS2 bridge (MuJoCo ↔ ROS2 topics)
- Teleoperate G1 in simulation

### Phase B: VLA Integration (2-3 weeks)
- Set up LeRobot / unitree_IL_lerobot
- Collect teleoperation demonstrations (20-50 per task)
- Train ACT model on collected data
- Evaluate: camera → VLA → joint actions → MuJoCo
- Iterate on simple tasks (reach, grasp, pick)

### Phase C: Task Complexity (2 weeks)
- Multi-step tasks (pick and place, object transfer)
- Fine-tune for each new task category
- Experiment with GR00T N1 (zero-shot + fine-tuned)
- Implement task success detection

### Phase D: RosClaw Integration (1-2 weeks)
- Set up rosbridge_server
- Connect OpenClaw (Charlie) via RosClaw plugin
- Implement Task Manager action server
- Test end-to-end: Telegram → task → VLA → simulation → result → Telegram
- Add camera snapshot capability (send scene to user)

### Phase E: Polish & Demo (1 week)
- Error handling and recovery behaviors
- Multi-task demonstration
- Record demo video
- Documentation

**Total estimated time: 8-10 weeks** (at 1-2 hours/day)

---

## 8. Hardware Requirements & Constraints

### Your Setup
- **GPU:** NVIDIA RTX 4050 Laptop (6GB VRAM)
- **RAM:** Assume 16-32GB
- **OS:** Windows + WSL2 + Docker

### What Fits
| Component | RTX 4050 Feasibility |
|-----------|---------------------|
| MuJoCo simulation | ✅ Perfect — CPU-primary, GPU optional |
| MuJoCo camera rendering | ✅ Fine for single env |
| ACT training | ✅ ~50M params, fits in 6GB |
| ACT inference | ✅ Real-time |
| GR00T N1 inference | ⚠️ Tight — 2B params, may need quantization (INT8) or offloading |
| GR00T N1 fine-tuning | ❌ Needs cloud GPU (A100/H100 or at minimum RTX 4090) |
| Isaac Sim | ⚠️ Marginally possible but uncomfortable |

### Cloud GPU Strategy
For GR00T N1 fine-tuning, use on-demand cloud:
- **Lambda Labs** — A100 80GB at ~$1.10/hr
- **RunPod** — RTX 4090 at ~$0.40/hr (sufficient for LoRA)
- **Google Colab Pro** — A100 for ~$10/month
- **Vast.ai** — Cheapest on-demand GPUs

Estimated cloud cost: ~$20-50 for initial fine-tuning experiments.

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| RTX 4050 insufficient for GR00T N1 inference | High | Start with ACT (50M params), use INT8 quantization for GR00T, or offload to cloud |
| MuJoCo ↔ ROS2 bridge complexity | Medium | Use existing `mujoco_ros2` packages, or build minimal custom bridge |
| VLA fine-tuning needs too much data | Medium | Start with simple tasks (reach, grasp), use Unitree's LeRobot framework |
| Camera rendering quality affects VLA | Medium | MuJoCo's rendering is basic but sufficient for VLA — photorealism not required for most tasks |
| RosClaw latency for task dispatch | Low | RosClaw is only for high-level commands, not in control loop |
| Sim-to-real gap (if ever going physical) | Out of scope | Addressed later via domain randomization and Isaac Sim |

---

## 10. Success Criteria

### Minimum Viable Demo
- [ ] G1 humanoid simulated in MuJoCo with camera feed
- [ ] ACT model trained on teleoperation data for 1 task (e.g., pick up object)
- [ ] VLA control loop working: camera → model → joint actions → physics
- [ ] Task completes successfully >50% of the time

### Full Demo
- [ ] 3+ different tasks working (pick, place, transfer)
- [ ] GR00T N1 or ACT fine-tuned with >80% success rate
- [ ] RosClaw connected — task commands from Telegram
- [ ] Camera snapshots sent back to user via Telegram
- [ ] Demo video recorded

### Stretch Goals
- [ ] Multi-step tasks with language reasoning ("find the red cup, then put it next to the blue plate")
- [ ] VLS/ACG integration for out-of-distribution generalization
- [ ] Isaac Lab migration for photorealistic rendering
- [ ] Locomotion + manipulation combined tasks

---

## 11. Key Resources

### Repositories
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation
- [Unitree IL LeRobot](https://github.com/unitreerobotics/unitree_IL_lerobot) — Imitation learning framework
- [Unitree ROS2](https://github.com/unitreerobotics/unitree_ros2) — ROS2 interface
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — High-quality robot models
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — VLA foundation model
- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) — RL for humanoid locomotion
- [RosClaw](https://github.com/PlaiPin/rosclaw) — NL robot control
- [Awesome Humanoid Learning](https://github.com/jonyzhang2023/awesome-humanoid-learning) — Curated resource list

### Papers
- GR00T N1: "An Open Foundation Model for Humanoid Robots" (NVIDIA, 2025)
- ACT: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
- VLS: "Vision-Language Steering" (2024)
- ACG: "Action Coherence Guidance for VLA Models" (ICRA 2026)
- Humanoid-Gym: "RL for Humanoid Robot with Zero-Shot Sim2Real Transfer" (2024)

### Platforms
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t) — Complete humanoid development platform
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — Robot learning framework
- [MuJoCo](https://mujoco.org/) — Physics engine