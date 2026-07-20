# Codebase Review — Technical Assessment (July 2026)

A full-repository technical review of `humanoid_vla`, written as the basis for the
[Upgrade Plan](UPGRADE_PLAN.md) and the [Architecture document](ARCHITECTURE.md).
The review is deliberately candid: knowing exactly where the current system is
simplified, scripted, or below community norms is what makes the upgrade plan credible.

**Scope reviewed:** `scripts/` (ML pipeline), `ros2_ws/src/vla_mujoco_bridge/` (ROS 2
integration), `sim/` (MJCF assets), repo hygiene/packaging, documentation, and the
claims made in `README.md` / `PROJECT_REPORT.md`, checked against the state of the art
in VLA and humanoid manipulation as of mid-2026.

---

## 1. What Is Genuinely Strong

These are the parts of the project that hold up under expert scrutiny and should be
preserved and built on:

1. **The bimanual physics track is the real core.** Friction-only grasping under full
   `mj_step` dynamics, with success defined by measured bilateral contact forces
   (`evaluate_bimanual.py` — lift ≥ 3 cm AND both palms in contact AND ≥ 2 N per palm,
   read from `mj_contactForce`), no weld constraints. This is legitimate learned
   manipulation with a defensible success criterion.
2. **PD torque control with gravity compensation** (`physics_sim.py`):
   τ = Kp·(q_des − q) − Kd·q̇ + `qfrc_bias`, clipped to `ctrlrange`, is the correct
   MuJoCo idiom, with sensibly hand-tuned per-joint gains.
3. **Damped-least-squares Jacobian IK** (`physics_sim.py:309–351`) is textbook-correct
   (LM damping, step clamping, joint-limit clamping at 95 % range) and — importantly —
   callers check the convergence flag and retry/skip on failure
   (`generate_bimanual_demos.py:137–140`). Lesson L047 documents why.
4. **Action chunking + temporal ensembling** are implemented faithfully (exponential
   decay weights, re-plan every 5 steps) and the hierarchical task decomposition
   (embedding switch at grasp trigger, ensemble-buffer reset) matches the docs.
5. **Threading/EGL discipline in `mujoco_sim.py`**: renderer created inside the physics
   thread so EGL binds to its owning thread, all shared state behind a lock,
   snapshot-copy accessors. This is real systems competence.
6. **The domain-randomization module** (`domain_randomization.py`) is clean: saves and
   restores nominal values, uses a correct small-angle quaternion perturbation.
7. **Documentation and honesty culture.** Six deep-dive study docs, 50 engineering
   lessons, a 600-line project report, and a results table that reports *failures*
   (Place at 65 %, OOD degradation to 55 %) — a maturity signal reviewers notice.
8. **Coherent phase-structured history** (A→F) that tells a story.

---

## 2. Critical Findings (Blocking / Credibility-Damaging)

### 2.1 A fresh clone does not run

`repos/mujoco_menagerie` and `repos/unitree_mujoco` are committed as **gitlinks with no
`.gitmodules`**, so `git submodule update --init` cannot resolve them. The robot model
references `meshdir="../repos/unitree_mujoco/unitree_robots/g1/meshes"`
(`sim/models/g1_29dof.xml:2`), which resolves to nothing after a clean clone. The README
does document a manual `git clone` step, but the broken gitlinks contradict it and the
first thing a reviewer does — "clone and run" — fails. **This is the single most
damaging issue in the repo.**

### 2.2 Git history bloat and committed artifacts

- `git count-objects -vH` → **~125 MiB pack**, dominated by a `venv/` committed in the
  initial commit and deleted later (largest blob: 29 MB `cv2.abi3.so`). The blobs are
  permanent until history is rewritten.
- **4 stale `.pyc` files are tracked** under
  `ros2_ws/src/vla_mujoco_bridge/vla_mujoco_bridge/__pycache__/` despite `.gitignore`
  rules — and they no longer even match the current module set.
- `media/` carries **~22 MB of MP4/GIF directly in git** (no LFS).

### 2.3 The "VLA / natural language" framing overstates the implementation

There is **no language model anywhere in the system**. "Language" is:

- a 4-entry Python list of task strings mapped by `list.index()` to an integer
  (`act_model.py:48–61`), fed to `nn.Embedding(num_tasks, 256)`;
- a keyword substring matcher in `task_manager_node.py:105–128` with concrete bugs:
  `"green"` and `"box"` route *any* sentence containing them to bimanual, and **any
  unrecognized command silently falls through to "pick up the red cube"**.

An interviewer familiar with OpenVLA / π0 / GR00T will identify this within minutes as a
task-conditioned behavior-cloning policy, not a VLA. The gap between the marketing and
`parse_task_command` is the project's biggest interview risk. (The PROJECT_REPORT is
more honest about this than the README.)

### 2.4 Single-arm "manipulation" is largely scripted, not learned

- Demos are generated in **pure kinematic mode**: `mj_forward` playback, the cube is
  teleported to follow the hand once welded (`generate_demos.py:228–244`), and recorded
  joint velocities are ~0 (29 of 58 state dims are dead inputs).
- At evaluation, "grasp" success is the **scripted auto-weld trigger** at hand–cube
  distance < 4 cm (`evaluate.py:151–163`); `check_grasp` just returns that flag. "Pick"
  is trivially satisfied once welded. For "place", the cube is **teleported to table
  height on release** (`evaluate.py:175–178`) because kinematic mode has no gravity.
- Net effect: the headline "89 % across 5 tasks" blends one genuinely physical task
  (bimanual, 100 %) with four tasks that are mostly *reach metrics wrapped in scripted
  state changes*. Reported separately and honestly this is still a fine result — blended
  under a "manipulation" banner it invites a credibility hit.

### 2.5 Scientific rigor below 2026 community norms

- **No train/val split** — `best.pt` is selected on *training* loss
  (`train_act.py:208–211`); train loss of 9e-6 with a "0 % success without temporal
  ensembling" observation is a textbook train/rollout-mismatch signature.
- **No seeding at all** in training (no `torch.manual_seed` / `np.random.seed` /
  cuDNN determinism) — runs are unreproducible.
- **No experiment tracking** (no wandb/TensorBoard/CSV) — only `print()`.
- **Evaluation**: 10–20 episodes per condition, single seed, no confidence intervals.
  At n = 20 each reported point carries roughly ±11 pp binomial noise; the marquee
  90→80→70→60→55 OOD curve has no error bars. Modern practice (and an easy win in sim)
  is ≥ 50 episodes/condition with Wilson 95 % CIs.
- **Concrete evaluation bug:** in `eval_generalization.py`, the suite applies posture
  randomization and then calls `run_episode`, which itself calls `reset_with_noise`
  (`evaluate.py:104`) — for single-arm, the suite's posture randomization is
  **immediately overwritten**, so the single-arm OOD-posture condition likely never
  tested what it claims.
- Thresholds are uncalibrated magic numbers (reach success at 6 cm while auto-grasp
  requires 4 cm; z > 0.90; table z = 0.825 inline).

### 2.6 Documentation/code mismatches

- `act_model.py:211` and `train_act.py:15` claim "~6M trainable params"; the actual
  count is **~12.8M trainable / 15.6M total** (the README's totals are right, the code
  docstrings are wrong).
- `PROJECT_REPORT.md` §5.2 inverts the ResNet split: it lists layers 0–6 as 3.8M frozen
  and layer4 as 4.2M trainable; in reality **layer4 ≈ 8.4M** (≈ 75 % of the backbone)
  and frozen layers 0–6 ≈ 2.8M. The table doesn't sum to its own total, and the §5.3
  overfitting rationale is weakened by the correct numbers.
- `README.md` documents `CLAUDE.md` and `tasks/todo.md` in the project tree; both are
  **gitignored and absent** — dangling references for anyone who clones.
- `num_tasks=8` default (`act_model.py:217`) with only 4 tasks — 4 dead embedding rows.

---

## 3. Engineering-Quality Findings

### 3.1 ROS 2 package (`vla_mujoco_bridge`)

- **Stringly-typed interfaces.** Task goals and status are `std_msgs/String` (+
  hand-serialized JSON). A long-running task with progress feedback, a result, and the
  need for cancellation is the textbook use case for a **ROS 2 Action**; there is no
  custom interface package at all. Concurrency is a manual `_executing` bool + lock.
- **Fragile imports.** `task_manager_node.py:61–88` climbs `../../../../scripts` on
  `sys.path`. After `colcon install` the relative path no longer points at the repo, so
  installed and source runs diverge. `scripts/` is not an installable package.
- **Duplicated inference loop.** The temporal-ensembling loop is re-implemented in
  `task_manager_node.py`, `evaluate.py`, `evaluate_bimanual.py`, `live_demo.py`, and
  `live_bimanual.py` — five copies of the same logic that can silently diverge.
- **Threading inconsistency**: a `ReentrantCallbackGroup` is declared but the node spins
  single-threaded; actual concurrency comes from a raw daemon thread publishing from a
  non-executor thread. `bridge_node.py` shows the better pattern.
- **No QoS profiles** — camera and joint-state topics use default RELIABLE instead of
  `SensorDataQoS` best-effort.
- **Model loading in the constructor** (two `torch.load` calls, `weights_only=False`)
  blocks node bring-up; a lifecycle node's `on_configure` is idiomatic. `weights_only=False`
  is also an arbitrary-code-execution footgun on untrusted checkpoints.
- **Dead scaffolding / stale metadata**: references to a nonexistent `task_commander`
  node; `cv_bridge` declared in `package.xml` but deliberately unused everywhere;
  `demo_recorder.py` imports `threading` twice; `arm_teleop_node.py` publishes
  `JointState` without names; both `bridge_node` and `task_manager_node` publish
  `/camera/image_raw` (topic collision if co-launched).
- **Tests are unmodified ament templates** (copyright/flake8/pep257) that would
  currently *fail* (missing license headers), and there are **no functional tests** for
  `parse_task_command`, the PD controller, or the sim wrappers.

### 3.2 ML pipeline (`scripts/`)

- **Architecture vs. paper**: the implementation is a simplified decoder-only variant of
  ACT — global-avg-pooled ResNet18 output as a *single* image token (all spatial
  structure destroyed before the transformer), 3 memory tokens total, no encoder stack,
  no CVAE, MSE (not L1) loss. Legitimate simplifications for deterministic scripted
  demos, but they should be documented as such (§2.6).
- **No state/action normalization** — raw radians in, raw radians out. Works for O(1)
  arm joints; breaks on any differently-scaled action space and diverges from every
  modern policy implementation (LeRobot policies carry dataset-stats normalization).
- **Duplication**: four sim wrappers (`SimWrapper`, `PhysicsSim`, `LiveSim`,
  `LivePhysicsSim`) re-implement ID caching, arm addressing, resets, camera rendering,
  and IK; the two dataset classes are ~95 % identical; `_KP/_KD`, `count_params`,
  `save_checkpoint` are copy-pasted across training scripts.
- **Dead code**: `LivePhysicsSim` creates a throwaway `PhysicsSim.__new__` object that
  is never used (`live_bimanual.py:70–72`); `convert_to_lerobot.py` targets the
  outdated LeRobot v2.0 format, hard-codes 29 joints (mislabels bimanual data), and
  nothing consumes its output.
- **Efficiency**: no mixed precision on a 6 GB GPU; `num_workers=0` with synchronous
  CPU-side augmentation in `__getitem__` is likely the actual training bottleneck; the
  whole dataset is preloaded to RAM (fine at ~9 k frames, unscalable beyond).
- **Config management** is argparse-only with structural hyperparameters (`nhead`,
  state/action dims) buried in inline dicts.
- Left/right IK solvers duplicated verbatim rather than parameterized by site/DOF ids.

### 3.3 Simulation assets (`sim/`)

- Scene design is thoughtful (tuned `solimp/solref` on the cube, `condim=4` + high
  friction on the bimanual box, runtime weld equality, DR distractor geoms, ego +
  scene cameras).
- No `<option>` block — the 500 Hz timestep is an unenforced default the Python loop
  merely assumes; `SUBSTEPS = 500//30 = 16` makes effective control cadence 480 Hz, not
  the documented 500/30 split.
- Fixed base is enforced by **overwriting pelvis/leg/waist state every substep**
  (`physics_sim.py:297–305`) rather than modeling a fixed base — a hack that injects
  non-physical constraint handling; physics is only "real" for the 14 arm DOFs + box.
- Gravity compensation hard-codes actuator-order == DOF-order 6..34; a reordered MJCF
  silently breaks it. Four of five actuator default classes are identical (redundant).
- No keyframe/home pose defined.

### 3.4 Packaging, CI, hygiene

- `requirements.txt` uses only `>=` lower bounds (no lock, no torch CUDA index
  guidance) — non-reproducible installs.
- **No CI at all** (`.github/workflows` absent) despite a badge-heavy README — a green
  build/lint/test badge is conspicuous by absence.
- No repo-root `pyproject.toml`, no ruff/black/pre-commit config.
- `install_ros2.sh` is actually good (set -e, distro detection, idempotent bashrc
  append); minor nits: `apt upgrade -y` inside an installer, no checksum on the
  downloaded .deb.
- Committed training log is harmless; `study/` and `tasks/lessons.md` are assets.

---

## 4. Gap Analysis vs. Mid-2026 State of the Art

| Axis | This project today | Community SOTA / norm (mid-2026) |
|---|---|---|
| Language | 4-way integer task embedding + keyword parser | VLM backbones (Eagle/PaliGemma/Qwen-VL/SmolVLM); frozen text encoders at minimum; paraphrase-robustness eval (LIBERO-Para shows 22–52 pp drops — the field's measured sore spot) |
| Action head | Deterministic MSE chunk regression | Flow-matching experts are the 2026 default (π0/π0.5, GR00T N1.5+, SmolVLA, RDT2-FM); Diffusion Policy standard baseline; OpenVLA-OFT legitimizes L1-chunked regression (97.1 % LIBERO) |
| Vision | Single avg-pooled ResNet18 token | Feature-map token sequences; SigLIP/DINOv2; ego point clouds (iDP3) for viewpoint generalization |
| Data format | Ad-hoc HDF5 + orphaned v2.0 converter | **LeRobotDataset v3** on HF Hub (streaming, standard tooling); unlocks SmolVLA/π0.5/ACT/DP for free |
| Demos | ~100 scripted-expert episodes | MimicGen/DexMimicGen-style augmentation (60 human demos → 21 k); XR/phone teleop (Open-TeleVision, unitree xr_teleoperate) |
| Benchmarks | Self-defined 5 tasks | LIBERO (≈ saturated at 97 %+), RoboCasa, BiGym (mobile bimanual humanoid, closest match), HumanoidBench |
| Eval rigor | 10–20 eps, single seed, no CIs | ≥ 50 rollouts/condition; Wilson 95 % CIs; paraphrase suites; SimplerEnv-style variant aggregation |
| Humanoid scope | Fixed-base arms-only (recognized "stage 1") | Whole-body control via RL (HOVER, ExBody2 — validated on G1; unitree_rl_lab), loco-manipulation, dexterous hands (Dex3-1) |
| Training stack | argparse + print() | LeRobot-style typed dataclass configs (draccus), wandb, AMP, ONNX/TensorRT export, async inference |
| ROS 2 | String+JSON topics | Custom action/msg interfaces, lifecycle nodes, QoS profiles, ros2_control framing |
| Feasible SOTA fine-tunes (RTX 4050 6 GB) | — | **SmolVLA (450M)** locally; GR00T N1.5 LoRA (16 GB+), π0.5 LoRA (22.5 GB+), OpenVLA-OFT (24 GB) on short rented-GPU runs |

---

## 5. Priority Summary

Ordered by (credibility impact × effort):

| # | Finding | Severity | Fix cost |
|---|---|---|---|
| 1 | Clean clone cannot load the robot (missing `.gitmodules`/meshes) | Blocking | Hours |
| 2 | No CI, template tests that fail, no functional tests | High | Days |
| 3 | History bloat (125 MiB venv blobs), tracked `.pyc`, media in git | High | Hours (history rewrite) |
| 4 | "VLA/NL" framing vs. integer task-id + keyword parser | High (interview risk) | Weeks (real fix) / Hours (honest reframing) |
| 5 | Scripted grasp/teleport shortcuts blended into headline metrics | High | Hours (reporting) / Weeks (physics single-arm) |
| 6 | No val split, no seeding, no tracking, n=20 no-CI eval, OOD-posture bug | High | Days |
| 7 | Param-count doc errors; broken README refs | Medium | Hours |
| 8 | String+JSON ROS interfaces; path-hack imports; 5× duplicated inference loop | Medium | Days |
| 9 | No normalization, no AMP, argparse-only config, dataset duplication | Medium | Days |
| 10 | Fixed-base overwrite hack, timestep unenforced, gravity-comp order assumption | Low–Medium | Hours |

The [Upgrade Plan](UPGRADE_PLAN.md) turns this table into a phased roadmap with
acceptance criteria; the [Architecture document](ARCHITECTURE.md) describes the current
system as-is and the target system the plan converges to.
