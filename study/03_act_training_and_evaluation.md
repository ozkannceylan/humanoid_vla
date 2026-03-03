# 03 — ACT Training & Evaluation (Phase B5–B6, Phase C)

> **What this covers:** Building a standalone ACT (Action Chunking with Transformers) model,
> training it on scripted expert demonstrations, extending to a multi-step "place" task,
> and evaluating the trained policy in simulation.

---

## 1. Why Standalone ACT (Not LeRobot)?

We had LeRobot 0.4.4 installed — a great framework from HuggingFace for robot learning.
But wrapping its Hydra-based training CLI turned out to be fragile:

| Approach | Pros | Cons |
|----------|------|------|
| **LeRobot CLI** | Battle-tested, many models, HF Hub integration | Hydra config complexity, HF dataset format conversion, API changes between versions |
| **Standalone ACT** | Full control, simple debug, reads HDF5 directly | Must implement from scratch, no pretrained weights |

**Decision:** Standalone. For a research project with 9000 samples and one architecture,
the 200 lines of custom training code are far easier to debug than format conversion pipelines.

> **Lesson L017:** For small-scale research, standalone training loops beat framework wrappers.

---

## 2. The ACT Model — Architecture Deep Dive

### 2.1 What is Action Chunking?

Standard behavior cloning predicts **one action per observation**. This causes:
- **Compounding errors:** small prediction errors accumulate over time
- **Temporal jitter:** each frame predicts independently → jerky motion

**Action chunking** (from the ACT paper, Zhao et al. RSS 2023) instead predicts
a **chunk of K future actions** at once. At inference time, only the first action
is executed, but the model was trained to predict the full trajectory segment.

```
Standard:   obs_t → action_t         (one step)
Chunking:   obs_t → [a_t, a_{t+1}, ..., a_{t+K-1}]   (K steps, use a_t)
```

Benefits:
1. **Temporal coherence** — the model learns to plan multi-step trajectories
2. **Error correction** — later actions in the chunk "know about" earlier ones
3. **Smoother motion** — predictions are inherently consistent over K frames

We use `chunk_size=20` (at 30Hz, this is ~0.67 seconds of lookahead).

### 2.2 Why No CVAE?

The original ACT paper uses a **Conditional Variational Autoencoder** (CVAE) to model
multimodal behavior — when there are multiple valid ways to do a task, the latent variable
$z$ captures which "mode" to follow.

Our scripted expert demos are **deterministic** — there is exactly one correct trajectory
per initial state (plus small random noise). CVAE's multimodal capacity is unnecessary.

```
Original ACT:  Image + State + z (sampled) → Actions    (handles human variance)
Our ACT:       Image + State + Task → Actions            (deterministic, simpler)
```

> **Lesson L018:** Skip CVAE for deterministic demos. Add it when training on human data.

### 2.3 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ENCODER (3 tokens → memory)                                                 │
│                                                                              │
│  Image (480×640×3)                                                           │
│    ↓ resize to 224×224                                                       │
│    ↓ ResNet18 (layers 0-6 frozen, layer4 trainable)                         │
│    ↓ AvgPool → (512,)                                                        │
│    ↓ Linear(512, 256) → img_token (1, 256)                                  │
│                                                                              │
│  State (58 = 29 joint pos + 29 joint vel)                                   │
│    ↓ Linear(58, 256) → ReLU → Linear(256, 256) → state_token (1, 256)      │
│                                                                              │
│  Task ID (integer 0–3)                                                       │
│    ↓ nn.Embedding(8, 256) → task_token (1, 256)                             │
│                                                                              │
│  memory = [img_token, state_token, task_token]  → (3, 256)                  │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────────┐
│  DECODER (queries attend to memory → action chunk)                           │
│                                                                              │
│  queries = nn.Embedding(20, 256)   ← 20 learnable position embeddings       │
│                                                                              │
│  TransformerDecoder(                                                         │
│    num_layers=4,                                                             │
│    d_model=256, nhead=4,                                                     │
│    dim_feedforward=1024,                                                     │
│    dropout=0.1                                                               │
│  )                                                                           │
│                                                                              │
│  output = decoder(queries, memory)  → (20, 256)                             │
│  actions = Linear(256, 29)(output)  → (20, 29)                              │
│                                                                              │
│  Each output token = one future joint configuration                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Why Freeze Early ResNet Layers?

ResNet18 has ~11M parameters. Our dataset has only 9000 samples. Training the full
network would overfit on MuJoCo's rendered images.

| Layer | What it learns | Frozen? |
|-------|---------------|---------|
| layers 0–3 (conv1, bn1, relu, maxpool) | Edges, colors, textures | ✅ Frozen |
| layer1–layer3 (residual blocks) | Shapes, patterns | ✅ Frozen |
| layer4 (512 channels) | High-level features | 🔄 Trainable |
| img_proj (Linear) | Task-relevant projection | 🔄 Trainable |

This reduces trainable params from 15.6M → 12.8M. Layer4 adapts ImageNet features to
MuJoCo's rendering style (flat colors, simple geometry) without catastrophic forgetting.

> **Lesson L019:** For <50K training samples, freeze all but the last ResNet block.

### 2.5 Parameter Count Breakdown

| Component | Params | Trainable |
|-----------|--------|-----------|
| ResNet18 (full) | 11.2M | 2.4M (layer4 only) |
| img_proj | 131K | 131K |
| state_proj MLP | 81K | 81K |
| task_embed | 2K | 2K |
| query_embed | 5K | 5K |
| TransformerDecoder | 4.2M | 4.2M |
| action_head | 7.5K | 7.5K |
| **Total** | **15.6M** | **12.8M** |

---

## 3. The Dataset: DemoDataset

### 3.1 Data Format

Each episode is an HDF5 file:

```
episode_0042.hdf5
├── obs/
│   ├── joint_positions    (T, 29) float32
│   ├── joint_velocities   (T, 29) float32
│   └── camera_frames      (T, 480, 640, 3) uint8
├── action                 (T, 29) float32
└── attrs:
    └── task_description = "pick up the red cube"
```

### 3.2 Preloading Strategy

Loading images lazily from HDF5 per `__getitem__` call is too slow (gzip decompression +
disk I/O per batch). Instead:

1. **At init time:** Load all episodes, resize images from 480×640 → 224×224, store as uint8
2. **At getitem time:** Convert uint8 → float32, normalize with ImageNet stats

```python
# Init: preload + resize (~515 MB for 80 episodes)
images = np.empty((T, 224, 224, 3), dtype=np.uint8)
for i in range(T):
    images[i] = cv2.resize(images_raw[i], (224, 224), interpolation=cv2.INTER_AREA)

# Getitem: fast normalization
img = ep['images'][t].transpose(2, 0, 1).astype(np.float32) / 255.0
img = (img - IMG_MEAN) / IMG_STD
```

> **Lesson L020:** For datasets <1GB after resize, preload everything. Normalize lazily.

### 3.3 Action Chunk Construction

For each sample at time $t$ in an episode of length $T$:

$$\text{chunk}[i] = \begin{cases} \text{action}[t + i] & \text{if } t + i < T \\ \text{action}[T - 1] & \text{otherwise (padding)} \end{cases}$$

The padding with the last action is important — it tells the model "if you're near the end
of a task, just hold your current pose." Without this, the model would have no ground truth
for the tail of chunks near episode boundaries.

### 3.4 Dataset Statistics

| Property | Value |
|----------|-------|
| Episodes | 80 (20 reach + 20 grasp + 20 pick + 20 place) |
| Total samples | 9000 |
| Episode length | ~90–140 frames (task-dependent) |
| Image size | 224×224×3 uint8 |
| State dim | 58 (29 pos + 29 vel) |
| Action dim | 29 (joint positions) |
| Preloaded RAM | ~515 MB |

---

## 4. Training Pipeline

### 4.1 Training Loop

The training is a standard supervised learning loop with MSE loss:

```
for epoch in range(300):
    for images, states, task_ids, action_chunks in dataloader:
        pred = model(images, states, task_ids)      # (B, 20, 29)
        loss = MSE(pred, action_chunks)              # mean over all dims
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)     # stability
        optimizer.step()
    scheduler.step()                                  # cosine annealing
```

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for transformers, weight decay prevents overfitting |
| Learning rate | 1e-4 | Conservative; larger LRs destabilize ResNet fine-tuning |
| Weight decay | 1e-4 | Regularization for small dataset |
| Scheduler | CosineAnnealing(T_max=300, eta_min=1e-6) | Smooth decay, avoids sudden drops |
| Batch size | 32 | Fits in 6GB VRAM with ~1.5GB usage |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients in decoder attention |
| Epochs | 300 | Loss plateaus after ~200 epochs |

### 4.3 Loss Curve

The loss drops rapidly in the first 20 epochs, then continues decreasing gradually:

```
Epoch    0 — loss: 0.010948 — lr: 1.00e-04    (random predictions)
Epoch   20 — loss: 0.000192 — lr: 9.88e-05    (57× decrease, learned basic motion)
Epoch   40 — loss: 0.000090 — lr: 9.55e-05    (fine trajectory matching)
Epoch   60 — loss: 0.000060 — lr: 9.02e-05
Epoch  100 — loss: 0.000033 — lr: 7.48e-05
Epoch  140 — loss: 0.000021 — lr: 5.52e-05
Epoch  200 — loss: 0.000012 — lr: 2.53e-05    (converging)
Epoch  220 — loss: 0.000011 — lr: 1.70e-05    (plateau)
Epoch  300 — loss: ~0.000009                    (final, see training log)
```

The rapid initial drop (0.011 → 0.0002 in 20 epochs) indicates the model quickly learns
the "big picture" — which direction to move the arm. Subsequent improvement is about
precision: exact joint angles, smooth trajectories, task-specific behavior.

### 4.4 Checkpoint Strategy

| File | When Saved | Purpose |
|------|-----------|---------|
| `latest.pt` | Every save_freq epochs + final | Resume interrupted training |
| `best.pt` | When avg_loss improves | Deployment / evaluation |
| `checkpoint_NNNN.pt` | Every save_freq (100) epochs | Historical snapshots |

Each checkpoint includes:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,    # for resuming training
    'epoch': 299,
    'loss': 0.000009,
    'config': {                      # for standalone model loading
        'state_dim': 58,
        'action_dim': 29,
        'chunk_size': 20,
        'hidden_dim': 256,
        'nhead': 4,
        'num_layers': 4,
        'num_tasks': 8,
        'task_labels': [...]
    }
}
```

---

## 5. Phase C — Multi-Step Place Task

### 5.1 Scene Extension

Added a **blue place marker** (cylinder) and invisible site on the table:

```xml
<!-- Inside <body name="table"> -->
<geom name="place_marker" type="cylinder" size="0.04 0.002"
      pos="-0.1 0.1 0.802" rgba="0.2 0.4 0.9 0.7"
      contype="0" conaffinity="0"/>   <!-- no physics collision -->
<site name="place_site" pos="-0.1 0.1 0.825" size="0.001"/>
```

World coordinates:
- Table at (0.3, -0.1, 0) → place_marker at (0.2, 0.0, 0.825)
- Cube starts at (0.3, -0.1, 0.825) → 14cm diagonal from place target
- Both within the arm's 40cm comfortable reach

> **Lesson L022:** Always set `contype="0" conaffinity="0"` on visual-only geoms.

### 5.2 Place Trajectory Design

The place task is a 5-waypoint trajectory:

```
Waypoint 0: APPROACH      Waypoint 1: DESCEND       Waypoint 2: LIFT
  Hand 8cm above cube       Hand 2cm above cube       Hand 12cm up (cube attached)
  35 interpolation frames   25 frames                 25 frames
                                ↓ GRASP at wp1

Waypoint 3: LATERAL MOVE  Waypoint 4: LOWER
  Hand above place target    Hand 2cm above target
  30 frames                  25 frames
                                ↓ RELEASE after wp4
```

The grasp happens after waypoint 1 (hand is close to cube), and the release happens
after waypoint 4 (cube is above the target). This produces natural-looking pick-and-place
motions.

### 5.3 release_after_wp: Extending Kinematic Recording

The original `_kinematic_record` only supported `grasp_after_wp` (activate weld).
For place, we need two events:

```python
# In _kinematic_record():
# Grasp: activate weld constraint
if not grasp_done and grasp_after_wp >= 0 and t >= wp_frame_idx[grasp_after_wp] - 1:
    sim.set_weld(True)
    grasp_done = True

# Release: deactivate weld constraint (new!)
if grasp_done and release_after_wp >= 0 and t >= wp_frame_idx[release_after_wp] - 1:
    sim.set_weld(False)
    grasp_done = False
```

> **Lesson L023:** Keep kinematic recording extensible with per-waypoint event hooks.

### 5.4 Place Demo Verification

```
Place task test (3 episodes):
  Episode 0: cube→target distance = 0.029m  ✓
  Episode 1: cube→target distance = 0.031m  ✓
  Episode 2: cube→target distance = 0.027m  ✓
  All welds released correctly.
```

### 5.5 Expanded Dataset

Final dataset: 80 episodes (20 per task), 9000 total samples, all generated with
100% convergence from the scripted IK expert.

---

## 6. Evaluation Framework

### 6.1 Run Episode

The evaluation loop mirrors demo generation but uses the **trained model** instead
of the scripted expert:

```python
for step in range(max_steps):
    image = sim.render_camera()                          # 480×640×3
    pos, vel = sim.get_obs()
    state = np.concatenate([pos, vel])                   # (58,)
    
    actions = model.predict(image, state, task_id)       # (20, 29)
    action = actions[0]                                  # first action of chunk
    
    sim.target_pos[RIGHT_ARM_CTRL] = action[RIGHT_ARM_CTRL]
    sim.step_frame()                                     # kinematic: qpos → mj_forward
    
    # Auto-grasp: hand < 4cm from cube → activate weld
    # Auto-release: place task, cube near target → deactivate weld
```

### 6.2 Auto-Grasp / Auto-Release

The model outputs only joint positions (29-dim), not grasp commands.
We use proximity-based heuristics:

| Event | Trigger | Mechanism |
|-------|---------|-----------|
| **Grasp** | `‖hand − cube‖ < 4cm` | Activate weld constraint |
| **Release** | Place task + cube near target + hand low + delay > 30 steps | Deactivate weld |

> **Lesson L021:** Standard simplification in VLA research. Model learns trajectory;
> grasping is mechanical.

### 6.3 Success Criteria

| Task | Criterion | Threshold |
|------|-----------|-----------|
| **Reach** | `‖hand − cube‖₂ < 0.06` at end | 6 cm |
| **Grasp** | Auto-grasp triggered during episode | hand < 4cm |
| **Pick** | Grasped AND `cube.z > 0.90` | 10cm above table |
| **Place** | Released AND `‖cube_{xy} − target_{xy}‖ < 0.06` AND `cube.z < 0.87` | 6cm, near table |

### 6.4 Phase B Criterion

> **Reach success ≥ 50%** → Phase B complete

This is the minimum bar. The scripted expert achieves 100% on all tasks, so the question
is how well the ACT model can imitate those trajectories from visual+state observations.

---

## 7. Key Design Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Training framework | Standalone PyTorch | Debugging ease, HDF5 direct read, no format conversion |
| CVAE | Skipped | Deterministic demos, single mode per state |
| Image encoder | ResNet18, frozen 0–6 | Small dataset, transfer from ImageNet |
| Task conditioning | Learned embedding (not CLIP) | 4 tasks, no need for open-vocabulary |
| Action representation | Joint positions, chunk=20 | Temporal coherence, works with PD controller |
| Loss function | MSE on full chunk | Standard for behavior cloning |
| Auto-grasp | Proximity heuristic | Model doesn't predict discrete grasp action |

---

## 8. File Reference

### scripts/act_model.py
- `TASK_LABELS` — string→int task registry
- `DemoDataset` — preloads HDF5, resizes images, constructs action chunks
- `ACTPolicy` — ResNet18 + MLP + Embedding → Transformer decoder → action head
- `predict()` — single-step inference for evaluation

### scripts/train_act.py
- Standalone training loop (no LeRobot)
- AdamW + CosineAnnealing + gradient clipping
- Saves best.pt / latest.pt / periodic checkpoints with full config

### scripts/evaluate.py
- `run_episode()` — policy rollout with auto-grasp/release
- `check_reach/grasp/pick/place()` — success detectors
- `load_model()` — reconstruct model from checkpoint config
- Per-task reporting with Phase B criterion check

### scripts/generate_demos.py (Phase C additions)
- `generate_place()` — 5-waypoint trajectory for pick-and-place
- `release_after_wp` in `_kinematic_record()` — weld deactivation
- `place_site_id` / `place_pos` in SimWrapper — place target access

### sim/g1_with_camera.xml (Phase C additions)
- `place_marker` — blue visual cylinder on table
- `place_site` — invisible site for position queries

---

## 9. Evaluation Results & Debugging

### 9.1 First Attempt: 0% on All Tasks

Running the trained model (epoch 299, loss 0.000009) with simple open-loop evaluation
produced **0% success on all tasks**. The model predicted near-zero actions that barely
moved the arm.

**Root cause: Compounding error.** In training, the model sees ground-truth observations
at every step (teacher forcing). At inference, it sees its *own* previous outputs fed back
as observations. Small errors accumulate exponentially — after ~30 steps, the state has
drifted so far from the training distribution that predictions become meaningless.

This is the fundamental challenge of behavior cloning (BC). The training loss was 0.000009
— practically perfect on the training data — but this means nothing for closed-loop rollout.

### 9.2 Fix 1: Temporal Ensembling

The ACT paper's solution: **temporal ensembling**. Instead of executing just the first
action from each chunk, we:

1. **Re-plan every K steps** (K=5): get a fresh 20-step action chunk
2. **Overlap chunks**: at any moment, multiple chunks have predictions for the current timestep
3. **Exponentially-weighted average**: newer predictions get higher weight

```python
ensemble_k = 0.01
for step in range(max_steps):
    if step % chunk_exec == 0:
        chunk = model.predict(image, state, task_id)  # (20, 29)
        for j in range(chunk_size):
            future = step + j
            if future < max_steps:
                w = math.exp(-ensemble_k * j)
                action_buffer[future] += chunk[j] * w
                weight_buffer[future] += w
    action = action_buffer[step] / weight_buffer[step]
```

The exponential weighting means: trust the near-future predictions more than the
far-future ones. With re-planning every 5 steps, each action is the consensus of
up to 4 overlapping chunks, smoothing out single-chunk noise.

**Result after temporal ensembling:**
| Task | Before | After |
|------|--------|-------|
| Reach | 0% | **100%** |
| Grasp | 0% | **90%** |
| Pick | 0% | 0% |
| Place | 0% | 0% |

> **Lesson L024:** Temporal ensembling is essential for BC — single-chunk prediction
> compounds errors within 30 steps. Re-plan every 5 steps with exponential weighting.

### 9.3 Fix 2: Hierarchical Task Decomposition

Pick and place still failed at 0%. Debugging revealed the cause: the model predictions
immediately lifted the hand **upward** instead of approaching the cube first.

**Root cause: Task embedding averaging.** With chunk_size=20 in a 90-frame episode,
the "pick" task embedding is trained on samples from *both* the approach phase (move toward
cube) and the lift phase (move upward). At the initial state, the embedding produces
an "average" behavior — net upward movement — which never reaches the cube.

**Solution: Hierarchical decomposition.** For composite tasks (pick, place), we split
execution into two phases:

```
Phase 1: Use "grasp" task embedding → approach the cube → until auto-grasp triggers
Phase 2: Switch to "pick"/"place" embedding → lift/transport → reset ensembling buffers
```

The key insight: the "grasp" task embedding only has approach-phase training data, so it
produces clean approach behavior. Once the cube is grasped, we switch to the task-specific
embedding which now only needs to handle the post-grasp phase.

The ensembling buffers are reset at the phase transition to prevent stale approach-phase
actions from bleeding into the lift phase.

**Result after hierarchical decomposition:**
| Task | Before | After |
|------|--------|-------|
| Pick | 0% | **90%** |
| Place | 0% | 0% |

> **Lesson L025:** For multi-phase tasks, decompose into sub-goals with different
> embeddings. Reset ensembling buffers at phase transitions.

### 9.4 Fix 3: Re-Grasp Prevention

Place task still failed at 0%. Debug output showed the cube reaching the target
(distance = 0.008m from target, hand lowered to z=0.929), and release *did* trigger
in single-episode debug runs.

**Root cause: Auto-grasp re-triggering.** After release, the hand and cube are at the
same position (they were just welded). The auto-grasp check immediately fires:
`if not grasped and dist(hand, cube) < 0.04` → re-grasps the cube.

**Solution:** Added a `released` flag that prevents re-grasping after the cube has been
released:

```python
released = False
# In auto-release:
if release_conditions_met:
    sim.set_weld(False)
    grasped = False
    released = True        # ← never re-grasp
    # Simulate gravity (kinematic mode has no physics)
    sim.data.qpos[cube_z_adr] = 0.825  # table surface
    mujoco.mj_forward(sim.model, sim.data)

# In auto-grasp:
if not grasped and not released:   # ← check released flag
    ...
```

Additionally, in kinematic mode there is no gravity — after releasing the weld,
the cube just floats in mid-air. We simulate gravity by setting the cube's z-position
to the table surface (0.825) and calling `mj_forward`.

> **Lesson L026:** After releasing a welded object, prevent the proximity-based
> auto-grasp from immediately re-triggering. Use a `released` flag.

> **Lesson L027:** Kinematic mode (`mj_forward`) has no physics forces — after
> releasing a weld, manually simulate gravity by setting the object's z to the
> landing surface.

### 9.5 Final Evaluation Results

With all three fixes applied (temporal ensembling + hierarchical decomposition +
re-grasp prevention + gravity simulation):

```
MUJOCO_GL=egl python3 scripts/evaluate.py \
    --checkpoint data/checkpoints/best.pt --episodes 20
```

| Task | Success | Rate | Avg Distance |
|------|---------|------|-------------|
| **Reach** the red cube | 20/20 | **100.0%** | 0.011m |
| **Grasp** the red cube | 18/20 | **90.0%** | 0.005m |
| **Pick up** the red cube | 18/20 | **90.0%** | 0.004m |
| **Place** the red cube on the blue plate | 13/20 | **65.0%** | 0.031m |
| **Overall** | **69/80** | **86.2%** | — |

**Phase B criterion:** ✅ Reach ≥ 50% (achieved 100%)
**Full demo criterion:** ✅ 3+ tasks > 80% (reach 100%, grasp 90%, pick 90%)

### 9.6 Analysis of Failure Cases

**Grasp failures (2/20):** Random cube displacement that places the cube at the edge
of reachable workspace. The approach trajectory drifts slightly, missing the 4cm
auto-grasp threshold.

**Pick failures (2/20):** Same episodes that fail grasp — if grasp doesn't trigger,
pick cannot succeed. When grasp succeeds, lift is 100% reliable.

**Place failures (7/20):** Several failure modes:
- Cube released but drifts slightly during transport (3 cases)
- Hand doesn't lower enough for release threshold, times out at max_steps (2 cases)
- Grasp fails in approach phase, so place never begins (2 cases)

Place is the hardest task by far: it requires successfully completing approach + grasp +
lift + transport + lower + release. Each step introduces error opportunity.

### 9.7 Progressive Improvement Summary

```
Baseline (no ensembling):                      0.0%  overall
+ Temporal ensembling:                         47.5%  overall (reach 100%, grasp 90%)
+ Hierarchical task decomposition:             70.0%  overall (+ pick 90%)
+ Re-grasp prevention + gravity simulation:    86.2%  overall (+ place 65%)
```

Each fix addressed a distinct failure mode. The compounding pattern:
trained loss 0.000009 → 0% closed-loop → 86.2% after inference-time fixes
illustrates why **evaluation engineering is as important as model training** in
behavior cloning.

---

## 10. What's Next: Phase D — RosClaw Integration

The trained ACT model runs at ~30Hz in the MuJoCo simulation. Phase D connects this
to the outside world:

```
Telegram → OpenClaw (Charlie) → RosClaw → VLA Task Manager → ACT Model → MuJoCo
```

The VLA Task Manager becomes a ROS2 action server that:
1. Receives a task goal ("pick up the red cube") via RosClaw
2. Runs the ACT inference loop until success or timeout
3. Reports completion status back to Telegram

The key architectural insight: RosClaw operates at the **task dispatch level** (one message),
while ACT runs in a tight **per-frame control loop** (30Hz). They don't need to be tightly
coupled.
