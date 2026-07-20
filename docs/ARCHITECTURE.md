# Architecture — Current System and Target System

Companion to the [Codebase Review](CODEBASE_REVIEW.md) and [Upgrade Plan](UPGRADE_PLAN.md).
Part 1 documents the system **as it exists today** (including the shortcuts, so the
diagram matches reality). Part 2 specifies the **target architecture** the upgrade plan
converges to.

---

## Part 1 — Current Architecture (as-built, July 2026)

### 1.1 Component map

```
                        ┌────────────────────────────────────────────────┐
                        │                REPOSITORY LAYOUT               │
                        │  scripts/        loose modules, sys.path hacks │
                        │  ros2_ws/…/vla_mujoco_bridge/  ROS 2 package   │
                        │  sim/            MJCF scene + G1 model         │
                        │  data/ (ignored) HDF5 demos + checkpoints      │
                        └────────────────────────────────────────────────┘

  User text ─► Telegram/rosbridge(ws:9090) ─► /vla/task_goal (std_msgs/String)
                                                     │
                                     ┌───────────────▼───────────────────┐
                                     │ task_manager_node (ROS 2, rclpy)  │
                                     │  keyword parser → (mode, task_id) │
                                     │  daemon thread per goal           │
                                     │  imports scripts/ via ../../../.. │
                                     │  re-implements ensembling loop    │
                                     └───────────────┬───────────────────┘
                                                     │ 30 Hz
        ┌────────────────────────────────────────────▼─────────────────────────────┐
        │                       INFERENCE LOOP (5 copies exist:                    │
        │   evaluate.py · evaluate_bimanual.py · live_demo.py · live_bimanual.py · │
        │   task_manager_node.py)                                                  │
        │                                                                          │
        │   ego RGB 480×640 → resize 224² → ImageNet norm ─┐                       │
        │   joint pos+vel (58-d single / 28-d bimanual) ───┼─► ACTPolicy ─► chunk  │
        │   task_id (int 0–4) ─────────────────────────────┘    (20, act_dim)      │
        │                                                                          │
        │   temporal ensembling: exec 5 of 20, exp-decay blend of overlapping      │
        │   chunks; hierarchical switch: grasp-embedding → task-embedding at       │
        │   auto-grasp trigger (hand–cube < 4 cm)                                  │
        └───────────────┬─────────────────────────────────┬────────────────────────┘
                        │ single-arm                      │ bimanual
        ┌───────────────▼────────────────┐   ┌────────────▼─────────────────────┐
        │ KINEMATIC MODE (mj_forward)    │   │ PHYSICS MODE (mj_step 500 Hz)    │
        │  qpos set directly, qvel ≈ 0   │   │  PD: τ=Kp·e−Kd·q̇+qfrc_bias      │
        │  weld constraint = "grasp"     │   │  friction-only bimanual squeeze  │
        │  cube teleported to hand/table │   │  contact-force success criterion │
        └───────────────┬────────────────┘   └────────────┬─────────────────────┘
                        └────────────┬────────────────────┘
                        ┌────────────▼────────────────────┐
                        │ MuJoCo: sim/g1_with_camera.xml  │
                        │  G1 29-DOF (torque actuators)   │
                        │  fixed base = state overwrite   │
                        │  every substep (legs/waist/     │
                        │  pelvis re-frozen)              │
                        │  ego cam (torso, fovy 87)       │
                        │  meshes from repos/ (broken     │
                        │  gitlinks — manual clone req.)  │
                        └─────────────────────────────────┘
```

### 1.2 Policy: `ACTPolicy` (`scripts/act_model.py`)

A simplified, decoder-only variant of ACT (Zhao et al., RSS 2023):

| Aspect | Implementation | vs. paper |
|---|---|---|
| Vision | ResNet18 → global avgpool → 1 token (256-d) | Paper keeps the 15×20 feature map as a token sequence |
| State | MLP 58→256→256 → 1 token | similar |
| Language | `nn.Embedding(num_tasks, 256)` over integer task-id → 1 token | No language model; not in the paper either (ALOHA is single-task) |
| Memory | **3 tokens total** | hundreds |
| Decoder | `nn.TransformerDecoder` 4L/4H/d256, 20 learned queries | Paper is encoder-decoder |
| CVAE | none (deterministic) | Paper's core component |
| Loss | MSE on raw radians (no normalization) | L1 + KL |
| Params | 15.6M total / 12.8M trainable (layer4 8.4M is the bulk) | ~80M |

Two model instances exist: single-arm (state 58, action 29, 4 tasks) and bimanual
(state 28, action 14, 1 task).

### 1.3 Data & training pipeline (current)

```
scripted expert (IK waypoints)                      training
  single-arm: kinematic playback + weld    HDF5      argparse config, no seed,
  bimanual:   mj_step + PD squeeze      ──►per-ep──► no val split, best.pt = min
  DR: colors/light/camera jitter          episode    TRAIN loss, print() logging,
  (runtime, bimanual only)                           no AMP, num_workers=0,
                                                     full dataset preloaded to RAM
```

- HDF5 schema: `obs/joint_positions`, `obs/joint_velocities`, `obs/camera_frames`
  (480×640×3 uint8), `action`, plus success/force/lift attrs on bimanual.
- Augmentation (ColorJitter/Blur/Crop) runs synchronously on the main process.
- A LeRobot v2.0 converter exists but nothing consumes it (orphaned).

### 1.4 Evaluation (current)

- 10–20 episodes/condition, one RNG consumed sequentially, no CIs.
- Single-arm success blends learned reaching with scripted state changes (auto-weld
  "grasp", teleported "place"). Bimanual success is fully physical (lift + bilateral
  ≥ 2 N contact).
- OOD suite (in-dist / position / visual / posture / combined); known bug: single-arm
  posture randomization is overwritten by `run_episode`'s internal reset.

### 1.5 ROS 2 layer (current)

- `task_manager_node`: `std_msgs/String` in, JSON-in-String status out; keyword NL
  parser; manual busy-flag concurrency; checkpoints `torch.load`ed in the constructor;
  spins single-threaded while declaring a reentrant callback group.
- `bridge_node` + `mujoco_sim`: the cleaner half — physics thread owns EGL, ROS
  executor on a background thread, locked snapshot accessors.
- Teleop nodes, HDF5 demo recorder, one launch file (rosbridge + task manager).
- No custom interfaces, no QoS profiles, no lifecycle, template-only tests.

---

## Part 2 — Target Architecture (post-upgrade)

The end-state after Phases 0–7 (Phase 8 adds the whole-body layer). Numbers in
brackets reference upgrade-plan phases.

### 2.1 Repository as a product

```
humanoid_vla/
├── pyproject.toml                  # installable package, pinned deps, ruff config [0]
├── .github/workflows/ci.yml       # lint + unit tests + EGL smoke rollout [0]
├── Makefile / justfile            # setup / fetch-assets / test / eval one-liners [0]
├── src/humanoid_vla/
│   ├── env/                       # ONE parameterized MuJoCo env [3]
│   │   ├── g1_env.py              #   single|bimanual × kinematic|physics
│   │   ├── control.py             #   PD + gravity comp (order-asserted)
│   │   ├── ik.py                  #   single DLS solver, side-parameterized
│   │   └── randomization.py
│   ├── policies/                  # bake-off heads behind one interface [4]
│   │   ├── act.py                 #   ACT (+CVAE option), L1, feature-map tokens
│   │   ├── diffusion.py           #   Diffusion Policy (UNet/transformer)
│   │   ├── flow.py                #   flow-matching action expert
│   │   └── runner.py              #   PolicyRunner: chunking + temporal ensembling
│   │                              #   + async inference — THE single copy [7]
│   ├── language/                  # frozen SigLIP/CLIP text encoder + projection [2]
│   ├── data/                      # LeRobotDataset v3 read/write, HF Hub push [5]
│   │   └── mimicgen.py            #   segment/transform/replay/verify augmentation [5]
│   ├── eval/                      # paired-seed suites, Wilson CIs, paraphrase
│   │   └── tables.py              #   suites, OOD suite, README table generator [1,2]
│   └── configs/                   # typed dataclasses (draccus-style) [1]
├── sim/                           # MJCF: <option timestep>, keyframe home pose,
│   └── …                          #   gripper/Dex3-1 variants, true fixed base [3]
├── ros2_ws/src/
│   ├── vla_interfaces/            # ExecuteManipulation.action + msgs [7]
│   └── vla_mujoco_bridge/         # lifecycle task manager (action server),
│                                  #   SensorDataQoS, imports humanoid_vla pkg [7]
├── deploy/                        # ONNX/TensorRT export + latency bench [7]
└── docs/                          # this folder; results all auto-generated
```

### 2.2 Target inference dataflow

```
 "put the blue cube on the plate"           ┌ paraphrase-robust: trained on
        │                                   │ instruction templates, evaluated on
        ▼                                   │ held-out paraphrases + task swaps [2]
 frozen text encoder (SigLIP text) ─► lang tokens ─┐
 ego RGB (+ optional depth→point cloud) ─► vision  ├─► policy head (best of bake-off:
        feature-map token sequence ────────────────┤    ACT-CVAE | diffusion | flow) 
 proprioception (norm. by dataset stats) ─► state ─┘         │
                                                             ▼
                                                   action chunk (normalized)
                                                             │
                                            PolicyRunner: temporal ensembling,
                                            async predict-ahead, ONNX-exportable
                                                             │
                                                             ▼
                                     PD torque control + gravity compensation
                                                             │
                                                             ▼
                                MuJoCo mj_step physics — ALL tasks, no welds:
                                actuated gripper closes under force control,
                                success = measured contact + object pose [3]
```

External VLAs (fine-tuned SmolVLA locally; GR00T N1.5 / π0.5 LoRA from cloud runs)
plug in as alternative policy backends behind the same `PolicyRunner` interface and are
compared on the same paired-seed eval suite [6].

### 2.3 Target ROS 2 topology

```
 client (CLI / rosbridge / Telegram)
        │  ExecuteManipulation.action  (goal: instruction; feedback: phase, step,
        ▼                               progress; result: success, metrics; CANCELABLE)
 vla_task_manager (LifecycleNode)
   on_configure: load checkpoints (safetensors/weights_only)
   on_activate:  action server up
   executor: MultiThreadedExecutor; SensorDataQoS on /camera & /joint_states
        │ uses
        ▼
 humanoid_vla.PolicyRunner ──► humanoid_vla.env (or hardware bridge later)
```

The String topic survives only as a thin compatibility shim for rosbridge demos.

### 2.4 Target evaluation contract

Every published number satisfies:

1. ≥ 50 episodes per condition, fixed per-episode seed lists shared across models;
2. Wilson 95 % CI reported alongside every success rate;
3. tables generated by `humanoid_vla.eval.tables`, never hand-typed;
4. learned-physics results never blended with scripted-assist results;
5. language results always include held-out-paraphrase and cross-task-swap axes;
6. at least one number on a public benchmark (BiGym or LIBERO suite) for external
   comparability.

### 2.5 Phase 8 extension (flagship)

```
            velocity / stance commands        arm+hand targets
 planner ────────────► RL balance policy ◄──── VLA manipulation policy
                        (unitree_rl_lab-trained,        (this repo)
                         sim2sim-validated in MuJoCo)
                                 │ leg torques + posture
                                 ▼
                    G1 free-floating base, full mj_step
                    (state-overwrite fixed base deleted)
```

Upper-body VLA and lower-body balance controller composed HumanPlus/ExBody2-style;
the bimanual lift executed while balancing is the flagship demo.

---

## Design decisions locked in by this document

| Decision | Rationale |
|---|---|
| Keep MuJoCo as the primary sim | Sim2sim validation target of the Isaac-trained ecosystem; team knowledge; MJX optional later |
| LeRobot conventions (dataset v3, dataclass configs) over custom formats | Ecosystem leverage: free policy zoo, HF Hub distribution, hiring-signal fluency |
| Frozen text encoder before any VLM backbone | Earns the "L" in VLA on 6 GB; VLM backbones arrive via SmolVLA/GR00T fine-tunes in Phase 6 |
| L1-regression chunked head stays as a first-class baseline | OpenVLA-OFT evidence that it rivals diffusion at a fraction of the cost |
| One env class, one PolicyRunner | Kills the 4-wrapper / 5-loop duplication that caused doc-code drift |
| Honest metric separation (learned vs. scripted) | Credibility is the portfolio's core asset |
| ROS 2 Actions for tasks | Progress + result + cancellation are the actual requirements; String+JSON was a prototype shortcut |
