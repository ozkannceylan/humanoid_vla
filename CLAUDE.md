# Workflow Orchestration

## 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

## 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

## 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

## 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

## 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

# Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

# Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

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

### 2.1 Recommendation

**Primary: MuJoCo** — RTX 4050 (6GB VRAM) is tight for Isaac Sim but perfect for MuJoCo. RL background from MSc thesis maps naturally. OpenVLA/Octo/ACT all use MuJoCo. MuJoCo Menagerie includes Unitree H1/G1 MJCF models.

**Secondary: Isaac Lab** — explore later for photorealistic rendering (GR00T N1 native), when better GPU is available, or for 1000s of parallel RL envs.

**Not recommended: Gazebo Harmonic** — lacks humanoid physics quality, no VLA ecosystem, not designed for RL/imitation learning.

---

## 3. Humanoid Model Selection

### 3.1 Recommendation

**Primary: Unitree G1** — Best VLA ecosystem (`unitree_IL_lerobot`), dexterous Dex3 hands for manipulation, complete open-source toolchain (MuJoCo + ROS2 + Isaac Lab + LeRobot IL), GR00T N1 compatible.

**Secondary: MuJoCo built-in humanoid** — for initial prototyping before moving to G1.

```bash
# G1 MJCF (official)
git clone https://github.com/unitreerobotics/unitree_mujoco
# Models in: unitree_robots/g1/

# H1 MJCF (MuJoCo Menagerie by Google DeepMind)
git clone https://github.com/google-deepmind/mujoco_menagerie
# Model in: unitree_h1/
```

---

## 4. VLA Model Selection

### 4.1 Recommendation

**Start with: ACT (Action Chunking with Transformers)** — ~50M params, trains on RTX 4050, proven for bimanual manipulation, available in LeRobot framework.

**Then: GR00T N1.6** — purpose-built for humanoids, dual-system (System 1: 30Hz diffusion action, System 2: 1-5Hz VLM reasoning), LeRobot-compatible fine-tuning. Inference on RTX 4050 possible with INT8 quantization; fine-tuning needs cloud GPU.

**Skip: OpenVLA** — 7B params, too large for RTX 4050, designed for tabletop manipulators.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  USER INTERFACE LAYER                                    │
│  Telegram ←→ OpenClaw (Charlie) ←→ RosClaw Plugin       │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (rosbridge)
┌──────────────────────▼──────────────────────────────────┐
│  ROS2 ORCHESTRATION LAYER                                │
│  Task Manager (Action Server) | Camera Bridge | Status   │
│  VLA Inference: camera + language → joint actions 30Hz   │
└──────────────────────┬──────────────────────────────────┘
                       │ Joint commands
┌──────────────────────▼──────────────────────────────────┐
│  SIMULATION LAYER (MuJoCo)                               │
│  Unitree G1 MJCF | Environment | Egocentric cameras      │
└─────────────────────────────────────────────────────────┘
```

### 5.1 ROS2 Topics & Interfaces

| Topic / Action | Type | Purpose |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Egocentric camera feed |
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions/velocities |
| `/joint_commands` | `sensor_msgs/JointState` | Commanded joint positions |
| `/vla/task_goal` | Custom Action | NL task command from RosClaw |
| `/vla/feedback` | Custom Feedback | Task progress to RosClaw |
| `/vla/status` | `std_msgs/String` | Human-readable status |

---

## 6. Fine-Tuning Pipeline

1. **Teleoperate** G1 in MuJoCo → record (camera frames + joint states + actions)
2. **Convert** to LeRobot HDF5 format with language annotations (20-50 demos/task)
3. **Fine-tune**: ACT from scratch (~2-4 hrs on RTX 4050); GR00T N1 LoRA (cloud GPU)
4. **Evaluate** → identify failure modes → collect more demos → iterate until >80% success

```bash
git clone https://github.com/unitreerobotics/unitree_IL_lerobot
```

---

## 7. Development Phases

| Phase | Goal | Duration |
|---|---|---|
| **A** | MuJoCo + G1 setup, camera pipeline, ROS2 bridge, teleoperation | 2 weeks |
| **B** | Collect demos, train ACT, evaluate VLA control loop | 2-3 weeks |
| **C** | Multi-step tasks, GR00T N1 experiments, success detection | 2 weeks |
| **D** | RosClaw integration, Telegram end-to-end, camera snapshots | 1-2 weeks |
| **E** | Polish, multi-task demo, documentation | 1 week |

**Total: ~8-10 weeks** at 1-2 hours/day

---

## 8. Hardware Constraints (RTX 4050 / 6GB VRAM)

| Component | Feasibility |
|---|---|
| MuJoCo simulation | ✅ Perfect |
| ACT training/inference | ✅ ~50M params fits |
| GR00T N1 inference | ⚠️ Needs INT8 quantization |
| GR00T N1 fine-tuning | ❌ Cloud GPU needed (RunPod RTX 4090 ~$0.40/hr) |
| Isaac Sim | ⚠️ Marginally possible |

---

## 9. Success Criteria

**Minimum Viable Demo:**
- [ ] G1 humanoid simulated in MuJoCo with camera feed
- [ ] ACT model trained for 1 task (pick up object)
- [ ] VLA control loop working end-to-end
- [ ] Task completes >50% of the time

**Full Demo:**
- [ ] 3+ tasks working (pick, place, transfer) with >80% success
- [ ] RosClaw connected — task commands from Telegram
- [ ] Camera snapshots sent back to user via Telegram
- [ ] Demo video recorded

---

## 10. Key Resources

### Repositories
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — G1/H1 simulation
- [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot) — Imitation learning
- [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) — ROS2 interface
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — High-quality robot models
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — VLA foundation model
- [lerobot](https://github.com/huggingface/lerobot) — Robot learning framework

### Papers
- GR00T N1: "An Open Foundation Model for Humanoid Robots" (NVIDIA, 2025)
- ACT: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
- ACG: "Action Coherence Guidance for VLA Models" (ICRA 2026)
