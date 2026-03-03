

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
- [x] **Auto-generated scripted demos** — scripted expert replaces manual teleoperation

---

## Milestone B4.5: Scripted Expert Demo Generation ✅

- [x] `scripts/generate_demos.py` — scripted expert auto-generates reach/grasp/pick demos
- [x] IK solver: iterative Jacobian (step=0.02, damping=0.05, tol=0.01) — converges in <20 iter
- [x] Architecture: kinematic IK → joint interpolation → kinematic playback (no mj_step)
- [x] Weld constraint enforced manually in kinematic mode (cube follows hand during lift)
- [x] `sim/g1_with_camera.xml` — table + cube repositioned to (0.3, -0.1) within arm reach
- [x] `sim/models/g1_29dof.xml` — right_hand_site added
- [x] `mujoco_sim.py` — get_site_xpos, get_body_xpos, get_site_jacp, set_grasp methods
- [x] `bridge_node.py` — /grasp service, /hand_pos + /cube_pos publishers
- [x] 60 episodes generated: 20 reach × 20 grasp × 20 pick — **100% convergence**
- [x] HDF5 output with task_description attribute for language-conditioned training

**Key lessons:**
- ctrlrange = torque limits, NOT position limits → use jnt_range
- G1 arm reach = 0.51m from shoulder → keep targets within 0.40m
- Pure kinematic mode (mj_forward) >> PD tracking for demo generation
- mj_forward ignores equality constraints → enforce weld manually

---

## Milestone B5: ACT Training Pipeline ✅

- [x] `scripts/act_model.py` — standalone ACT policy (ResNet18 + Transformer decoder)
  - ~6M trainable params, ~1.5 GB VRAM at batch_size=32
  - Language conditioning via learned task embedding (4 tasks)
  - Action chunking: predicts 20 future timesteps at once
- [x] `scripts/train_act.py` — standalone training loop (no LeRobot dependency)
  - Reads HDF5 demos directly, cosine LR schedule, gradient clipping
  - Checkpoint saving with model config for standalone loading
- [x] `DemoDataset` class — preloads + resizes images to 224×224 at init

---

## Milestone B6: VLA Evaluation ✅

- [x] `scripts/evaluate.py` — evaluation in MuJoCo simulation
  - Per-task success detection (hand proximity, grasp, lift height, placement)
  - Auto-grasp: hand within 4cm of cube triggers weld
  - Auto-release: for place task, cube near target + delay triggers release
  - Reports per-task success rate

---

# Phase C — Multi-Step Tasks & Evaluation

Last updated: 2026-03-03

**Strategy:** Extend to multi-step "place" task. Train single ACT model on all 4 tasks.
Evaluate per-task success rates.

---

## Milestone C1: Scene Extension ✅

- [x] `sim/g1_with_camera.xml` — blue place marker + place_site on table
  - World position (0.2, 0.0, 0.825) — 14cm diagonal from cube
  - contype="0" conaffinity="0" — purely visual, no physics effect
- [x] `scripts/generate_demos.py` — place_site_id + place_pos property in SimWrapper

---

## Milestone C2: Place Task ✅

- [x] `generate_place()` — 5-waypoint trajectory:
  approach(8cm above) → descend(2cm above) → [GRASP] → lift(12cm) → lateral move → lower → [RELEASE]
- [x] `release_after_wp` parameter in `_kinematic_record` — deactivates weld after specified waypoint
- [x] Verified: 3/3 episodes converged, cube-to-target distance = 0.029m

---

## Milestone C3: Expanded Dataset ✅

- [x] 80 episodes generated (20×4 tasks): reach, grasp, pick, place
- [x] 9000 total samples, 100% convergence

---

## Milestone C4: ACT Training on All Tasks ✅

- [x] 300-epoch training run: 93.3 min on RTX 4050 (6GB VRAM)
- [x] Final loss: 0.000009 (from 0.010948 at epoch 0)
- [x] Best checkpoint saved: `data/checkpoints/best.pt`
- [x] Training log: `logs/act_training_300ep.log`

---

## Milestone C5: Evaluation ✅

- [x] Run `evaluate.py` on trained model (20 episodes per task)
- [x] Implemented temporal ensembling (chunk_exec=5, ensemble_k=0.01)
- [x] Implemented hierarchical task decomposition (grasp→task phase switching)
- [x] Fixed auto-grasp re-triggering after release (`released` flag)
- [x] Added gravity simulation for kinematic mode

**Results:**

| Task | Success | Rate |
|------|---------|------|
| Reach | 20/20 | **100%** |
| Grasp | 18/20 | **90%** |
| Pick | 18/20 | **90%** |
| Place | 13/20 | **65%** |
| **Overall** | **69/80** | **86.2%** |

- [x] ✅ Phase B criterion met: reach ≥ 50% (achieved 100%)
- [x] ✅ Full demo criterion: 3+ tasks > 80% (reach, grasp, pick)

---

## ✅ PHASES A–C COMPLETE

All milestones delivered. Key metrics:
- 80 scripted expert demos (20×4 tasks, 100% convergence)
- ACT model: 12.8M trainable params, trained in 93 min
- Final evaluation: 86.2% overall success rate
- 4 inference fixes discovered via systematic debugging

---

# Phase C2 — Bimanual Box Manipulation (Full Physics)

Last updated: 2026-03-04

**Goal:** Two-hand squeeze grasp of a larger box (20×15×15 cm), using real physics
simulation (mj_step with contact forces and friction). Separate ACT model for bimanual tasks.

**Key architectural change:** Transition from kinematic mode (mj_forward) to full physics
(mj_step) with PD controller, gravity compensation, and contact-based friction grasping.
No weld constraints — the box is held purely by friction from both hands squeezing.

**RESULT:** 100% success rate (20/20), mean lift 8.5cm, bilateral forces 13-14N.

---

## Milestone C2.0: Model Preparation ✅

- [x] Add `left_hand_site` to `g1_29dof.xml` (mirror of right_hand_site)
- [x] Add collision geoms ("palm pads") to both hands for contact
  - Box-shaped geoms (~8×6×2 cm) approximating palm surface
  - `friction="1.5 0.005 0.0001"` (high sliding friction for rubber-on-cardboard)
  - `contype="1" conaffinity="1"` (enable contact detection)
  - Must not interfere with existing single-arm tasks (cube is too small to touch palms)
- [x] Verify: existing kinematic pipeline still works (regression test)

---

## Milestone C2.1: Scene Extension ✅

- [x] Add big box to `g1_with_camera.xml`:
  - Size: 10×7.5×7.5 cm half-extents (= 20×15×15 cm full)
  - Freejoint, mass ~0.3 kg, high friction
  - Position: centered in front of robot, reachable by both arms
  - Color: green (distinct from red cube and blue plate)
  - Proper contact parameters: solimp, solref, condim=4 (torsional friction)
- [x] Add `box_site` for position tracking
- [x] Verify: box stays on table under gravity with mj_step

---

## Milestone C2.2: Physics Sim Wrapper ✅

- [x] `scripts/physics_sim.py` — new sim wrapper using mj_step
  - PD controller: τ = Kp*(q_des - q) - Kd*q̇ + gravity_comp
  - Fixed base (pelvis frozen kinematically each step)
  - Both arms controllable: LEFT_ARM_CTRL + RIGHT_ARM_CTRL
  - Bimanual IK solver (iterative Jacobian for both arms independently)
  - Camera rendering (offscreen EGL)
  - Step = 16 mj_step calls per control frame (500Hz physics / 30Hz control)
- [x] Contact monitoring: detect hand-box contact pairs
- [x] PD gain tuning: Kp=40/10, Kd=4/1 (shoulders/wrists)
- [x] Freeze ALL non-arm joints (legs + waist) every substep (L028)
- [x] Verify: hands stable at targets, squeeze forces 11-14N bilateral

---

## Milestone C2.3: Bimanual Trajectory Planning ✅

- [x] Trajectory design (6 waypoints, synchronized left/right):
  1. HOME: rest position
  2. PRE-APPROACH: above + sides of box (collision-free path)
  3. APPROACH: 5cm outside box surface at box height
  4. SQUEEZE: 3cm inside box surface (compliance grasping)
  5. LIFT: squeeze pos + 15cm up
  6. HOLD: stabilize at lifted height
- [x] Bimanual IK: solve left arm and right arm independently
- [x] ±10% timing variation for training diversity
- [x] Verify: trajectory + PD → box lifts 6-9cm, held by friction

---

## Milestone C2.4: Demo Generation ✅

- [x] `scripts/generate_bimanual_demos.py` — physics-based demo generator
  - PD tracking of planned trajectories via mj_step
  - Records: camera (480×640×3) + joint state/vel (14D) + actions (14D)
  - 30 Hz control, 500 Hz physics, ±2cm box position noise
- [x] Generate 30 episodes: 100% success, lifts 4.4-9.5cm, forces 13-14N
- [x] Verify: smooth trajectories, no oscillation, box doesn't slip

---

## Milestone C2.5: Bimanual ACT Model + Training ✅

- [x] Bimanual ACT: state_dim=28 (14 pos + 14 vel), action_dim=14 (both arms)
  - Same architecture: ResNet18 + MLP + TaskEmbed → TransformerDecoder
  - Task label: "pick up the green box with both hands"
  - 15.6M total params, 12.8M trainable
- [x] Train: 300 epochs, loss 0.0177 → 0.000009, 52.5 min on RTX 4050
- [x] Checkpoints: data/bimanual_checkpoints/{best.pt, latest.pt}

---

## Milestone C2.6: Evaluation ✅

- [x] Evaluate bimanual model with full physics (mj_step)
- [x] Success: box lifted ≥3cm + BOTH palms in contact + force ≥2N per palm
- [x] 20 episodes: **100% success rate (20/20)**
- [x] Lift: mean=8.5cm, min=6.5cm, max=10.4cm
- [x] Force: L_mean=13.6N, R_mean=13.3N
- [x] Target was >50% — achieved 100%

---

## Milestone C2.7: Live Demo + Documentation ✅

- [x] Live viewer with scripted expert demo (`scripts/live_bimanual.py`)
- [x] Live viewer with ACT model inference (`--checkpoint` flag)
- [x] Study document: `study/04_bimanual_physics_grasping.md`
- [x] Update lessons.md (L028-L033), README.md
- [x] Git commit

---

# Phase D — RosClaw Integration & Task Manager

Last updated: 2026-03-03

**Goal:** Natural language task dispatch via ROS2. A Task Manager node accepts
NL commands, runs the appropriate ACT model in MuJoCo, and reports results.
rosbridge_server enables WebSocket access for RosClaw/OpenClaw.

---

## Milestone D.1: VLA Task Manager Node ✅

- [x] `task_manager_node.py` — ROS2 node that:
  - Subscribes to `/vla/task_goal` (std_msgs/String) for NL task commands
  - Maps NL commands to appropriate model (single-arm ACT or bimanual ACT)
  - Runs VLA inference loop in MuJoCo (temporal ensembling)
  - Publishes `/vla/status` (std_msgs/String) — JSON progress/result
  - Publishes `/camera/image_raw` (sensor_msgs/Image) during execution
  - Auto-detects task type: single-arm (reach/grasp/pick/place) vs bimanual
- [x] Register in setup.py + update package.xml

---

## Milestone D.2: rosbridge_server ✅

- [x] Install ros-jazzy-rosbridge-server
- [x] Verify: launch rosbridge, connect via WebSocket

---

## Milestone D.3: Launch File ✅

- [x] `launch/vla_system.launch.py` — launches:
  - rosbridge_server (WebSocket port 9090)
  - task_manager_node (with model paths)

---

## Milestone D.4: Test End-to-End ✅

- [x] Launch system via launch file
- [x] Send task via `ros2 topic pub /vla/task_goal std_msgs/String`
- [x] Verify: VLA runs, status published, success reported

---

# Phase E — Polish & Demo

Last updated: 2026-03-03

---

## Milestone E.1: Demo Videos ✅

- [x] Record individual task videos (scene + ego camera) for all 5 tasks
- [x] Store in media/ folder (committed, not gitignored)
- [x] Keep files small (<5MB each)

---

## Milestone E.2: Study Document 05 ✅

- [x] `study/05_system_integration.md` — covers:
  - Task Manager architecture
  - NL command parsing and model routing
  - ROS2 ↔ VLA inference loop
  - rosbridge WebSocket interface
  - Full system data flow

---

## Milestone E.3: Finalize README ✅

- [x] Add video embeds for all tasks
- [x] Document Phase D architecture
- [x] Update project structure
- [x] Final success metrics table

---

## Milestone E.4: Final Commit ✅

- [x] Update all docs (lessons.md, todo.md)
- [x] Git commit + push

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
