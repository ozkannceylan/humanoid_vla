# Upgrade Plan — Roadmap to a SOTA-Credible Humanoid VLA Stack

This plan converts the findings of the [Codebase Review](CODEBASE_REVIEW.md) into a
phased, prioritized roadmap. Each phase has concrete tasks, acceptance criteria, an
effort estimate, and a hardware budget (local RTX 4050 6 GB vs. short rented-GPU runs).
The target end-state is described in [ARCHITECTURE.md](ARCHITECTURE.md).

**Design principles**

1. *Credibility before capability.* A reviewer's first 15 minutes are "clone → run →
   read the claims." Phases 0–1 make that experience flawless before any new ML lands.
2. *Honest metrics are a feature.* Separating scripted-assist results from learned
   manipulation, and adding confidence intervals, makes the numbers stronger, not weaker.
3. *Ride the ecosystem.* Migrating to the LeRobot data/training conventions unlocks
   SmolVLA, π0.5, Diffusion Policy, and ACT reference implementations for free and
   signals fluency with the tooling employers actually use.
4. *Humanoid-specific depth is the differentiator.* Plenty of portfolios fine-tune a
   VLA on a tabletop arm. Whole-body control, bimanual physics, and G1-specific work
   are what make this repo stand out.

**Phase overview**

| Phase | Theme | Effort | Hardware |
|---|---|---|---|
| 0 | Make it clone-and-run: repo repair, CI, packaging | ~1 week | none |
| 1 | Scientific rigor: seeding, val split, tracking, statistical eval | ~1 week | local |
| 2 | Real language conditioning + paraphrase robustness | 1–2 weeks | local |
| 3 | Physics-true single-arm manipulation (retire the weld) | 1–2 weeks | local |
| 4 | Modern policy heads: Diffusion Policy, flow matching, ACT-CVAE bake-off | 2 weeks | local |
| 5 | LeRobot v3 data platform + demo scaling (MimicGen-style) | 1–2 weeks | local |
| 6 | SOTA VLA fine-tuning: SmolVLA locally, GR00T N1.5 / π0.5 LoRA in cloud | 2 weeks | local + ~$50–150 cloud |
| 7 | ROS 2 professionalization: actions, lifecycle, deployment latency | 1–2 weeks | local |
| 8 | Whole-body humanoid: RL locomotion + loco-manipulation (stretch) | 4+ weeks | local + cloud |

Phases 0–2 are the highest ROI per hour. Phases 3–7 can be reordered or run in
parallel. Phase 8 is the flagship stretch goal.

---

## Phase 0 — Make It Clone-and-Run (blocking)

The repo must survive "git clone → follow README → see the robot move" on a clean
machine. Everything else builds on this.

### 0.1 Fix the missing-mesh problem
- Remove the orphan gitlinks for `repos/unitree_mujoco` and `repos/mujoco_menagerie`.
  Either add a proper `.gitmodules` (pinned SHAs) **or** replace with a
  `scripts/fetch_assets.sh` that shallow-clones the two repos at pinned commits —
  the script approach avoids submodule friction and is easier for reviewers.
- Add a `make setup` / `just setup` one-liner: fetch assets → install deps → run smoke test.

### 0.2 History and artifact cleanup
- `git filter-repo` to purge the leaked `venv/` blobs (~100 MiB reclaimed) and the four
  tracked `__pycache__/*.pyc` files. Coordinate the force-push (history rewrite) once,
  early, before more forks/clones exist.
- Move `media/*.mp4` (~22 MB) to Git LFS or GitHub release assets; keep only the two
  README GIFs in-tree.
- Delete stale ignore-listed references: either commit real `CLAUDE.md` / `tasks/todo.md`
  or remove them from the README project tree.

### 0.3 Packaging
- Create a proper installable package: move `scripts/*.py` logic into
  `src/humanoid_vla/` with a root `pyproject.toml` (`pip install -e .`), keeping thin
  CLI wrappers. Delete every `sys.path.insert` hack (five entry points + the ROS node's
  `../../../../scripts` climb).
- Pin dependencies: `requirements.txt` with `==` pins (or `uv lock`), plus a documented
  torch CUDA index URL.

### 0.4 CI + tests (first real ones)
- GitHub Actions workflow: ruff lint + format check, `pytest`, and a **headless smoke
  test** (load MJCF with fetched assets, construct the policy, run one forward pass, one
  10-step kinematic rollout under `MUJOCO_GL=egl`).
- Seed unit tests where bugs were found or logic is pure:
  `parse_task_command` (including the `"green"`/`"box"` misroute and silent-fallback
  bugs), PD torque computation, IK convergence on a reachable target, dataset
  shape/normalization round-trip, chunk padding.
- Fix or delete the failing ament template tests (add license headers or drop
  `test_copyright`).

### 0.5 Truth-in-docs pass
- Fix the "~6M trainable" docstrings (`act_model.py`, `train_act.py`) → 12.8M/15.6M.
- Fix the inverted ResNet split table in `PROJECT_REPORT.md` §5.2 (layer4 ≈ 8.4M
  trainable; layers 0–6 ≈ 2.8M frozen) and revisit the §5.3 rationale.
- README: split headline metrics into "learned, physics-based" (bimanual 100 %) vs.
  "learned reaching + scripted grasp assist" (single-arm), until Phase 3 retires the
  weld. This *increases* credibility.

**Acceptance criteria:** fresh clone on a clean Ubuntu 24.04 VM reaches a rendered
rollout in ≤ 10 minutes of scripted steps; CI green badge in README; repo pack size
< 30 MiB; zero dangling doc references.

---

## Phase 1 — Scientific Rigor

Make every number in the README defensible.

### 1.1 Reproducible training
- Global seeding (`torch`, `numpy`, `random`, cuDNN deterministic flag) with the seed
  recorded in the checkpoint and the run config.
- **Train/val split by episode** (e.g., 90/10), `best.pt` selected on validation loss;
  log the gap (the current train-loss-9e-6 / 0-%-without-ensembling combination is a
  documented overfitting signature — show it, then fix it).
- Typed dataclass configs (LeRobot-style draccus pattern — the community moved off
  Hydra) serialized alongside every checkpoint; retire the inline config dicts.
- `wandb` (or offline TensorBoard fallback) for loss/LR/grad-norm curves; kill the
  ASCII loss curve in the report in favor of real plots.
- AMP (`torch.autocast` + `GradScaler`) and `num_workers>0` with GPU-side or
  worker-side augmentation — free speed on a 6 GB card.
- Add state/action normalization from dataset statistics (stored in the checkpoint,
  applied symmetrically at inference). Switch loss to L1 (matches ACT paper and
  OpenVLA-OFT findings); keep MSE as a config option for comparison.

### 1.2 Statistical evaluation
- ≥ **50 episodes per task/condition** (cheap in sim), fixed per-episode seed lists
  shared across all models being compared (paired evaluation).
- Report **Wilson 95 % confidence intervals** on every success rate; a small
  `eval/stats.py` helper + a results table generator so the README tables are emitted
  by code, not typed by hand. (A 2026 audit found essentially no real-robot VLA papers
  reporting CIs — doing this in a portfolio repo is a differentiator.)
- **Fix the OOD-posture bug**: `run_episode` re-randomizes via `reset_with_noise`,
  overwriting the suite's posture randomization for single-arm — thread the episode
  configuration through explicitly, then re-run the whole OOD table.
- Calibrate thresholds (reach 6 cm vs. grasp 4 cm inconsistency) and hoist all magic
  numbers into a single `constants.py` / task-spec dataclass.
- Add a multi-seed (≥ 3 training seeds) run for the headline model; report mean ± CI.

**Acceptance criteria:** every README number regenerable by one command
(`python -m humanoid_vla.eval --suite full`); all tables carry n and CI; OOD table
re-published with the bug fixed (numbers may change — that's the point).

---

## Phase 2 — Real Language Conditioning (the "L" in VLA)

Replace the integer task-id with genuine language grounding — the single highest-leverage
ML upgrade, and it targets the field's measured sore spot (paraphrase robustness:
LIBERO-Para reports 22–52 pp drops for policies that shortcut language).

### 2.1 Language encoder
- Swap `nn.Embedding(num_tasks)` for a **frozen text encoder** — CLIP/SigLIP text tower
  (~60M params, runs comfortably on 6 GB) — projecting the instruction to the policy's
  token space. Keep the task-id path behind a config flag as the ablation baseline.
- Generate an **instruction-template corpus** per task (30–50 paraphrases each:
  synonym substitution for verbs *and* objects, word-order variation, distractor
  mentions), split into train/held-out sets.
- Retrain single-arm and bimanual policies with sampled paraphrases per episode.

### 2.2 Paraphrase-robustness evaluation (new marquee table)
- Eval axes: (a) training instructions, (b) held-out paraphrases, (c) cross-task
  instruction swaps — verify the policy performs the *instructed* task, not the most
  likely one (this is the test the integer embedding fails by construction).
- Report success + CI per axis; add to README as a first-class result.

### 2.3 Fix the NL front-end
- Replace the keyword parser in `task_manager_node.py` with the same text encoder →
  the policy consumes the raw instruction end-to-end. Delete the
  `"green"`/`"box"` keyword misrouting and the silent "pick up the red cube" fallback;
  unknown/low-confidence instructions must return an explicit error to the caller.
- Add multi-object scenes (red cube + blue cube + green box) so instructions are
  *discriminative* — "pick up the blue cube" must require reading the sentence.

**Acceptance criteria:** held-out-paraphrase success within ~10 pp of
training-instruction success; cross-task swap accuracy > 90 %; demo video of two
different-colored cubes selected purely by instruction.

---

## Phase 3 — Physics-True Single-Arm Manipulation

Retire the weld/teleport shortcuts so every headline number reflects learned, physical
manipulation (the bimanual track already meets this bar).

- Add an actuated end-effector to the G1 model: start with the **parallel two-finger
  gripper** MJCF (matches shipped G1 hardware; Unitree assets exist); optionally the
  **Dex3-1 three-finger hand** (7 DOF) later as a stretch.
- Move single-arm tasks to full `mj_step` physics with the existing PD + gravity-comp
  stack; the scripted expert closes the gripper via force/position control rather than
  a weld; contact-force-based grasp detection replaces the proximity weld trigger.
- Record joint velocities for real (currently ~0 in kinematic playback — 29 dead input
  dims), and drop the place-height teleport: success = object physically resting within
  the target zone.
- Re-generate demos, retrain, re-evaluate with Phase 1 statistics. Expect lower
  success rates than the scripted-assist numbers — publish both tables side by side
  with an honest note; that comparison is itself compelling content.
- Unify the four sim wrappers (`SimWrapper`, `PhysicsSim`, `LiveSim`,
  `LivePhysicsSim`) into one parameterized environment class (single-arm | bimanual ×
  kinematic | physics) while touching this code; parameterize the duplicated
  left/right IK into one function; delete `LivePhysicsSim._tmp` dead code.
- Model hygiene while in the MJCF: explicit `<option timestep="0.002">`, a home-pose
  `<keyframe>`, replace the every-substep pelvis/leg overwrite with a properly fixed
  base (attach via fixed joint or remove the freejoint in a manipulation-specific
  variant), assert actuator→DOF ordering for gravity comp at load time.

**Acceptance criteria:** zero calls to `set_weld` in any training/eval path; all five
tasks succeed under `mj_step` with measured contact forces; one environment class
serves demos, eval, live viewers, and the ROS node.

---

## Phase 4 — Modern Policy Heads (Bake-Off)

Turn the single ACT variant into a small, well-controlled architecture study — the kind
of table a research-engineer portfolio should contain. All of these train on 6 GB.

| Head | Why it's in the study |
|---|---|
| ACT (current, deterministic, L1) | Baseline; OpenVLA-OFT shows L1-chunked regression is competitive (97.1 % LIBERO) |
| **ACT + CVAE** | The actual RSS 2023 architecture; needed once demos become multimodal (teleop/MimicGen) |
| **Diffusion Policy** (U-Net & transformer variants, ~10–80M) | The standard 2025–26 baseline everyone expects |
| **Flow-matching head** (~100M expert, π0/SmolVLA-style) | The 2026 default action head; drop-in replacement for the decoder |

- Also upgrade the vision tokenization: replace the single avg-pooled token with the
  ResNet **feature-map token sequence** (+2D positional embeddings) per the ACT paper;
  ablate frozen-SigLIP vs. ResNet18 backbones.
- Shared training/eval harness (same data, same seeds, same eval suite from Phase 1)
  → one results table: success ± CI, params, VRAM, inference latency, actions/sec.
- Optional extension: an **iDP3-style egocentric point-cloud variant** using MuJoCo
  depth rendering — directly relevant to single-ego-camera humanoids and strong
  viewpoint-generalization story.

**Acceptance criteria:** ≥ 3 heads trained on identical data/seeds; published bake-off
table with CIs and latency; the best head becomes the default for later phases.

---

## Phase 5 — Data Platform: LeRobot v3 + Demo Scaling

- **Migrate HDF5 → LeRobotDataset v3.0** (Parquet shards + MP4 + metadata; streaming
  API) and **push datasets to the Hugging Face Hub** — replaces the orphaned v2.0
  converter (which also hard-codes 29 joints and mislabels bimanual data). Publishing a
  clean humanoid dataset is portfolio value in itself.
- **MimicGen-style augmentation**: the scripted experts are an asset here — segment
  source demos (approach/grasp/transport/place), then transform-and-replay across
  randomized object poses with physics verification and success filtering.
  Target: ~100 seed demos → 2–5 k verified episodes. Document acceptance-rate stats.
- Re-train the Phase 4 winner on the scaled dataset; quantify the demo-count →
  success-rate curve (100 vs. 500 vs. 2 k vs. 5 k) — a classic, persuasive plot.
- Optional: human-in-the-loop demos via phone/keyboard teleop of the MuJoCo G1
  (LeRobot teleop utilities; Open-TeleVision/xr_teleoperate if a headset is available)
  to introduce natural multimodality — which the CVAE/diffusion/flow heads from
  Phase 4 are built for.

**Acceptance criteria:** dataset on HF Hub loadable by stock `lerobot`; ≥ 2 k verified
episodes; scaling curve published.

---

## Phase 6 — SOTA VLA Fine-Tuning

With LeRobot-format data, fine-tune real VLAs and benchmark them against the in-house
policies on the same eval suite. Hardware reality (RTX 4050 6 GB) dictates the split:

| Model | Size / head | Where it runs |
|---|---|---|
| **SmolVLA** (450M, flow matching) | Fine-tunes on ~6–8 GB (bf16, grad-accum, frozen vision) — community-verified on laptop GPUs / free Colab | **Local** |
| **GR00T N1.5/N1.7** (3B, flow-matching DiT) | LoRA r=32 community recipes fit 16 GB | Rented 4090/L4, hours-scale (~$10–40) |
| **π0.5** (3.3B, flow + FAST, LeRobot port available) | LoRA ≥ 22.5 GB | Rented A100, hours-scale |
| OpenVLA-OFT (7B) | LoRA ≈ 24 GB | Optional; same tier |

- Deliverable: a **"VLA on a budget" comparison table** — in-house ACT/diffusion/flow
  policies vs. SmolVLA vs. one 3B-class LoRA fine-tune, all on the same 50-episode
  paired eval with CIs, plus a writeup on VRAM/cost/latency trade-offs. Few portfolios
  contain an apples-to-apples table like this.
- Wire the winning fine-tuned model into the ROS 2 task manager as an alternative
  policy backend (config-selectable), proving the integration layer is policy-agnostic.
- Optional external validation: submit the best in-house policy to a **LIBERO** or
  **BiGym** task suite (BiGym is MuJoCo bimanual-humanoid — the closest public
  benchmark to this project) so at least one number is comparable to published work.

**Acceptance criteria:** ≥ 2 external VLAs fine-tuned on project data; comparison table
in README; one benchmark number on a public suite.

---

## Phase 7 — ROS 2 Professionalization & Deployment

- **Custom interface package `vla_interfaces`**: an `ExecuteManipulation.action`
  (goal: instruction string; feedback: phase/step/progress; result: success, metrics)
  replacing String+JSON. Free wins: type safety, introspection, and **cancellation**.
  Keep a thin String bridge topic for rosbridge/Telegram compatibility.
- **Lifecycle node** for the task manager: checkpoints load in `on_configure`
  (with `weights_only=True` or safetensors), not the constructor.
- Proper **QoS**: `SensorDataQoS` for camera/joint streams; resolve the
  `/camera/image_raw` topic collision between bridge and task manager.
- Deduplicate: the temporal-ensembling inference loop exists in five files — extract a
  single `PolicyRunner` in the installable package; the ROS node imports it (the
  `sys.path` climb died in Phase 0).
- Fix small defects from review: double `import threading` (demo_recorder), missing
  `JointState.name` (arm_teleop), phantom `task_commander` references, stale
  `cv_bridge` dependency in `package.xml`.
- **Deployment story**: ONNX export of the policy (+ optional TensorRT), a latency
  benchmark (policy Hz vs. 30 Hz control budget, chunk length vs. re-plan rate), and
  SmolVLA-style **async inference** (act on chunk k while predicting k+1). Publish the
  latency table.

**Acceptance criteria:** `ros2 action send_goal` runs a task end-to-end with live
feedback and working cancel; one shared inference implementation; latency table in docs;
ament tests pass in CI.

---

## Phase 8 — Whole-Body Humanoid (Flagship Stretch)

The recognized "stage 2" for humanoid manipulation portfolios: remove the fixed base.

- **8.1 Balance/locomotion policy**: train a G1 RL locomotion/balance policy with
  **unitree_rl_lab** (Isaac Lab + PPO) or MuJoCo-native MJX; validate **sim2sim in this
  repo's MuJoCo stack** (the standard Isaac→MuJoCo validation flow). Cloud GPU for
  training; inference is tiny.
- **8.2 Loco-manipulation composition**: upper-body VLA policy + lower-body balance
  controller (HumanPlus/ExBody2-style decomposition — ExBody2 is validated on G1):
  stand at the table and manipulate under a live balance controller; then approach-
  then-pick with a velocity-command interface.
- **8.3 Optional dexterity**: swap grippers for Dex3-1 hands on one task;
  DexMimicGen-style augmentation for hand demos.
- Even partial completion (balance-while-bimanual-lift) is a standout demo video few
  candidate portfolios can show.

**Acceptance criteria:** G1 maintains balance under `mj_step` with no state overwrites
while completing the bimanual lift; demo video; writeup of the controller composition.

---

## Suggested Execution Order & Milestones

```
M1  (Week 1)      Phase 0 complete — clone-and-run + CI green          ← do first, non-negotiable
M2  (Week 2)      Phase 1 complete — all numbers regenerable w/ CIs
M3  (Weeks 3–4)   Phase 2 — language conditioning + paraphrase table   ← biggest ML credibility jump
M4  (Weeks 5–6)   Phase 3 — weld retired, physics-true single arm
M5  (Weeks 7–8)   Phase 4 — policy-head bake-off table
M6  (Weeks 9–10)  Phase 5 — LeRobot v3 on HF Hub + 2–5k demos
M7  (Weeks 11–12) Phase 6 — SmolVLA + GR00T/π0.5 LoRA comparison
M8  (Weeks 13–14) Phase 7 — ROS 2 actions + deployment latency story
M9  (Ongoing)     Phase 8 — whole-body flagship
```

**If time is short**, the minimum set that transforms the repo's reception is:
Phase 0 + Phase 1 + Phase 2 (≈ 4 weeks): a reproducible, CI-green, statistically
rigorous project with *real* language conditioning — at that point the "VLA" claim is
earned, and every subsequent phase only adds depth.
