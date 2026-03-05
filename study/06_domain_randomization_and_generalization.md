# 06 — Domain Randomization & Generalization (Phase F)

> **What this covers:** Why the Phases A-E model was memorizing instead of learning,
> how domain randomization fixes this, the math behind each randomization type,
> and how to train and evaluate models that actually generalize.

---

## 1. The Core Problem: Memorization vs Learning

### 1.1 What Was Happening

Our Phase C2 bimanual model achieved **100% success rate** on evaluation. Sounds great, right?
But look at what the training data actually looked like:

| Factor | Training Range | Result |
|--------|---------------|--------|
| Box position | +/-2cm from nominal | Box is always at the same pixel location |
| Robot posture | Always q=0 (default) | Arms always start identically |
| Table color | Always brown (0.6, 0.4, 0.2) | Same background every frame |
| Lighting | Always same direction/intensity | Same shadows every frame |
| Camera | Fixed ego camera on torso | Identical viewpoint every frame |

The model was essentially doing **trajectory replay** — it memorized a single motion and played
it back regardless of what the camera saw. The ResNet18 image encoder was barely doing anything
useful; you could replace it with a constant vector and get similar results.

### 1.2 How to Prove It's Memorizing

We ran the old model against **out-of-distribution** conditions:

```
Old model (Phase C2) on generalization eval:
  in_dist:     50.0%   (same range, different seeds — already dropping!)
  ood_position: 35.0%  (1.5x wider box positions)
  ood_visual:   50.0%  (new table/light colors)
  ood_posture:  40.0%  (1.5x wider arm starting poses)
  ood_combined: 30.0%  (everything changed at once)
```

A model that truly understood "see box, move hands to box" would handle wider positions gracefully.
The 35% on OOD position proves it was memorizing pixel patterns, not understanding the task.

### 1.3 The Pixel Memorization Trap

Here's the math of why +/-2cm noise is basically zero:

```
Camera resolution:    640 x 480 pixels
Box size:             20 x 15 cm (200 x 150 mm)
Distance from camera: ~30 cm
Box at 30cm covers:   ~200 pixels wide

2cm displacement at 30cm distance:
  Angular shift = atan(0.02 / 0.30) = 3.8 degrees
  Pixel shift = ~13 pixels out of 640

  That's 2% of the image width!
```

A neural network can trivially ignore a 13-pixel shift — the box is always "in the same place"
in the image. The model learns a lookup table: "this pixel pattern → this joint trajectory".

With +/-8cm noise:
```
  Angular shift = atan(0.08 / 0.30) = 14.9 degrees
  Pixel shift = ~53 pixels out of 640 = 8.3% of image width
```

Now the box appears at **meaningfully different positions**. The model must actually
locate the green blob in the image to plan the right trajectory.

---

## 2. Domain Randomization — The Theory

### 2.1 What Is Domain Randomization?

**Domain randomization** (Tobin et al., 2017) is a technique from sim-to-real transfer.
The idea: if you train with enough visual variation in simulation, the real world
becomes "just another variation" that the model has learned to handle.

```
Traditional approach:
  Make simulation look realistic → hope model transfers

Domain randomization:
  Make simulation look RANDOMLY different every episode
  → model learns to ignore irrelevant visual features
  → focuses on task-relevant features (object location, shape)
  → naturally transfers to real world (or any new visual setting)
```

### 2.2 Why It Works — The Information Theory View

Consider what the model needs to extract from an image:

```
Image I = task_relevant_features + nuisance_features

task_relevant: box position, box orientation, hand positions
nuisance:      table color, lighting, shadows, background
```

If nuisance features are **constant** across training, the model can use them as shortcuts:
"dark brown blob at pixel (300, 400) means table is here, box is 50px to the right".

If nuisance features are **randomized**, these shortcuts become unreliable.
The only **consistent** signal across episodes is the task-relevant features.
The model is forced to learn the invariant: "green blob (wherever it is) = target".

This is essentially **data augmentation at the simulation level** — but more powerful
because it affects physics, geometry, and rendering, not just pixel transforms.

### 2.3 The Randomization Spectrum

From cheapest to most impactful:

```
         Cheap                                              Expensive
          |                                                     |
  Image augmentation → Position noise → Posture variation → Visual randomization
  (ColorJitter,etc)    (box +/-10cm)    (random arm start)   (colors, lighting,
   Training-time only   Sim reset only   IK + retry loop      distractors, camera)
   Zero sim cost        Low sim cost     Medium sim cost       Modifies MuJoCo model
```

Phase F implements all four levels.

---

## 3. Phase F Sub-Phases — Implementation Details

### 3.1 F0: Training-Time Image Augmentation

**File:** `scripts/act_model.py` (DemoDataset class), `scripts/train_bimanual.py`

**What it does:** Applies random visual transforms to camera images during training,
BEFORE the ImageNet normalization step. Zero simulation cost — just modifies tensors.

**Transforms applied:**

```python
# 1. Color Jitter — randomly adjust brightness, contrast, saturation, hue
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
# brightness=0.3 means: multiply brightness by random factor in [0.7, 1.3]
# hue=0.05 means: shift hue by random amount in [-0.05, 0.05] (subtle)

# 2. Gaussian Blur — 30% chance, simulates camera defocus
GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

# 3. Random Resized Crop — 50% chance, simulates slight camera movement
RandomResizedCrop(scale=(0.85, 1.0), ratio=(0.95, 1.05))
# scale=0.85 means: crop to 85-100% of original, then resize back to 224x224
# This simulates small position shifts + mild zoom changes
```

**Why this order matters:**

```
Raw image (uint8, 0-255)
  → Convert to float tensor (0-1)
  → ColorJitter (operates on 0-1 range)
  → GaussianBlur (if random < 0.3)
  → RandomResizedCrop (if random < 0.5)
  → ImageNet normalization: (pixel - mean) / std
  → Feed to ResNet18
```

The augmentation MUST happen before normalization, because ColorJitter expects [0, 1] inputs,
not the centered [-2.x, 2.x] range that ImageNet normalization produces.

**CLI flag:** `--no-augment` disables it (for ablation studies). Default: augmentation ON.

### 3.2 F1: Wider Object Position Randomization

**Files:** `scripts/generate_demos.py`, `scripts/generate_bimanual_demos.py`, `scripts/physics_sim.py`

**What changed:**

```
Before (Phase C):     box position = nominal + uniform(-0.02, +0.02)  → +/-2cm
After  (Phase F):     box position = nominal + uniform(-0.08, +0.08)  → +/-8cm (bimanual)
                      cube position = nominal + uniform(-0.10, +0.10) → +/-10cm (single-arm)
```

**The reachability problem:**

With +/-10cm noise, the cube can end up at positions the arm can't reach.
The Unitree G1 right arm has a maximum reach of ~0.51m from the shoulder.

```
Shoulder position (approx): (0.0, -0.10, 1.08)
Nominal cube position:      (0.3, -0.10, 0.825)
Distance: sqrt(0.3^2 + 0^2 + 0.255^2) = 0.394m  ← well within reach

Worst case with +/-10cm:    (0.4, -0.20, 0.825)
Distance: sqrt(0.4^2 + 0.1^2 + 0.255^2) = 0.489m  ← still OK but tight

Beyond +/-10cm:             (0.5, -0.20, 0.825)
Distance: sqrt(0.5^2 + 0.1^2 + 0.255^2) = 0.571m  ← EXCEEDS reach!
```

**Solution:** Reachability pre-check + IK retry loop:

```python
def is_reachable(self, target_xyz):
    """Check if target is within arm workspace."""
    shoulder = np.array([0.0, -0.10, 1.08])
    return np.linalg.norm(target_xyz - shoulder) < 0.40  # conservative limit

# In generate_reach():
for _ in range(10):  # up to 10 random positions
    sim.reset_with_noise(rng, noise_range=0.10)
    cube = sim.cube_pos.copy()
    if not sim.is_reachable(cube):
        continue  # try another random position
    result = _kinematic_record(sim, rng, waypoints, ...)
    if result is not None:
        return result  # IK succeeded
raise RuntimeError("Failed after 10 attempts")
```

**For bimanual (PhysicsSim):** separate x/y noise because the box is symmetric:
- x (forward): +/-8cm — plenty of room
- y (lateral): +/-6cm — tighter because both arms need to reach from opposite sides

### 3.3 F2: Random Starting Posture

**Files:** `scripts/generate_demos.py` (SimWrapper), `scripts/physics_sim.py` (PhysicsSim)

**The problem with always starting at q=0:**

```
q=0 means all joint angles are at their default (usually zero or nominal standing pose).
Every training trajectory starts from the EXACT same arm configuration.
The model never learns to handle: "my arm is slightly to the left, how do I adjust?"
```

**The solution:**

```python
def random_arm_start(self, rng, arm='both', spread=0.25):
    """Sample random arm configuration within safe joint limits.

    spread=0.25 means: sample from the middle 25% of each joint's range.
    This gives visible posture variation without extreme configurations.
    """
    mid = (joint_lo + joint_hi) / 2      # center of joint range
    half_range = (joint_hi - joint_lo) / 2 * spread  # scaled range

    for attempt in range(20):
        q = rng.uniform(mid - half_range, mid + half_range)
        q = np.clip(q, joint_lo, joint_hi)

        # Set joints and check: is the hand above the table?
        self.data.qpos[arm_qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)

        if hand_z > 0.85:  # above table surface (0.80m) + margin
            return q

    # Fallback: use default pose (safe but not random)
    return np.zeros(7)
```

**Why the hand-height check?**

Some random joint configurations put the hand below the table, inside the table,
or at awkward poses where the arm is folded on itself. The height check ensures:
1. Hand is in a physically valid position (not inside geometry)
2. Hand is above the workspace (table at z=0.80m)
3. There's enough room to plan a trajectory to the box

**For bimanual:** Both arms are randomized independently (`arm='both'`),
but with a tighter spread (0.25 vs 0.30) because both hands need to
reach the box from opposite sides.

### 3.4 F3+F4: Visual Domain Randomization

**New file:** `scripts/domain_randomization.py` — the `DomainRandomizer` class

**What gets randomized:**

| Element | How | Range |
|---------|-----|-------|
| Table color | `model.geom_rgba[table_id]` | Brown/grey/white spectrum |
| Floor color | `model.geom_rgba[floor_id]` | RGB each in [0.1, 0.5] |
| Lighting position | `model.light_pos[0]` | Nominal +/- 0.5m each axis |
| Lighting intensity | `model.light_diffuse[0]` | [0.3, 0.9] uniform |
| Red cube color | `model.geom_rgba[cube_id]` | Keep reddish, vary brightness |
| Green box color | `model.geom_rgba[box_id]` | Keep greenish, vary brightness |
| Distractors | Show/hide + reposition | 50% visible, random color/position |
| Camera position | `model.cam_pos[cam_id]` | +/- 2cm each axis |
| Camera orientation | `model.cam_quat[cam_id]` | +/- 3 degrees per axis |

**How MuJoCo runtime modification works:**

MuJoCo separates the **model** (static properties) from **data** (simulation state).
Many model properties can be modified at runtime — they take effect on the next `mj_forward()`:

```python
# This works! Changes are immediate after mj_forward()
model.geom_rgba[table_id] = [0.8, 0.3, 0.1, 1.0]  # orange table
model.light_pos[0] = [0.5, 0.3, 2.0]                # move light
model.cam_pos[cam_id] += [0.02, -0.01, 0.0]         # nudge camera
mujoco.mj_forward(model, data)  # changes take effect
# Next render will show the new colors/lighting/camera
```

This is much faster than reloading the XML, and it's reversible:

```python
# Save nominal values in __init__
self._nominal['table_rgba'] = model.geom_rgba[table_id].copy()

# Restore after episode
model.geom_rgba[table_id] = self._nominal['table_rgba']
```

**Distractor objects:**

Three static objects are defined in the XML (`sim/g1_with_camera.xml`):
- `distractor_0`: small box (2cm)
- `distractor_1`: small cylinder (1.5cm radius)
- `distractor_2`: small sphere (2cm radius)

They have `contype=0 conaffinity=0` (no collision) and `rgba alpha=0` (invisible by default).
The randomizer shows/hides them by setting alpha to 0 or 1, and repositions them on the table.

**Purpose:** Force the model to distinguish the target object (green box) from
other colored objects. Without distractors, "find the colored blob" is trivially easy.

**Camera perturbation math:**

Small-angle quaternion approximation for orientation:
```
Given rotation angles dx, dy, dz (in radians):
  delta_quaternion = normalize([1, dx/2, dy/2, dz/2])
  new_quaternion = nominal_quaternion * delta_quaternion   (Hamilton product)
```

This is valid for small angles (<10 degrees) and avoids Euler angle gimbal lock issues.
We use +/-3 degrees = +/-0.052 radians, well within the small-angle regime.

### 3.5 F5: Generalization Evaluation Framework

**File:** `scripts/eval_generalization.py`

**The evaluation protocol:**

```
For each test distribution:
    For each task:
        Run N episodes (default 20) with that distribution's settings
        Record: success/failure, distance metric or lift height
    Compute: per-task success rate, overall success rate

Test distributions:
  in_dist      → noise_mult=1.0, start_mult=1.0, no visual randomization
  ood_position → noise_mult=1.5, start_mult=1.0  (50% wider positions)
  ood_visual   → noise_mult=1.0, visual randomization ON
  ood_posture  → noise_mult=1.0, start_mult=1.5  (50% wider starting poses)
  ood_combined → noise_mult=1.5, start_mult=1.5, visual randomization ON
```

**Why 1.5x for OOD?**

The multiplier should be large enough to test extrapolation but not so large that
the task becomes physically impossible. 1.5x means:
- If training used +/-8cm box noise, OOD uses +/-12cm
- If training used spread=0.25 for posture, OOD uses spread=0.375

This tests whether the model learned a **general strategy** or just memorized
the training distribution's boundaries.

**Expected behavior after Phase F training:**

| Model | In-dist | OOD-pos | OOD-vis | OOD-posture | OOD-combined |
|-------|---------|---------|---------|-------------|--------------|
| Old (Phase C2) | 50% | 35% | 50% | 40% | 30% |
| Phase F (expected) | 60-70% | 50-60% | 60-70% | 50-60% | 40-50% |

Note: Phase F in-dist might be **lower** than old in-dist because the training task
is harder (more variation = harder optimization). But OOD scores should be much higher.
The total **area under the generalization curve** is what matters, not just in-dist performance.

---

## 4. The Training Pipeline — Step by Step

### 4.1 Generate Demos

```bash
# Bimanual demos with all Phase F randomizations
MUJOCO_GL=egl python3 scripts/generate_bimanual_demos.py \
    --episodes 50 \
    --noise-x 0.08 --noise-y 0.06 \
    --random-start 0.25 \
    --domain-rand \
    --seed 200 \
    --output data/bimanual_demos_phase_f
```

**What each flag does:**

| Flag | Value | Effect |
|------|-------|--------|
| `--episodes 50` | 50 | Generate 50 demonstration episodes |
| `--noise-x 0.08` | 8cm | Box X position randomized +/-8cm |
| `--noise-y 0.06` | 6cm | Box Y position randomized +/-6cm |
| `--random-start 0.25` | 25% | Sample arm start from middle 25% of joint range |
| `--domain-rand` | ON | Enable visual randomization (colors, lighting, distractors, camera) |
| `--seed 200` | 200 | Random seed for reproducibility |

**What happens internally per episode:**

```
1. sim.reset()                              — reset to default state
2. sim.reset_with_noise(rng, 0.08, 0.06)   — randomize box position
3. sim.random_arm_start(rng, 'both', 0.25)  — randomize arm start
4. randomizer.randomize(rng)                — randomize visuals
5. plan_bimanual_trajectory(sim, box_pos)    — IK solve for waypoints
6. Execute trajectory with physics (mj_step)
7. Record: images, joint_pos, joint_vel, actions at 30Hz
8. Save to HDF5 file
9. randomizer.restore()                     — reset visuals for next episode
```

**Expected output:** ~50 HDF5 files, each ~6MB, ~170 frames per episode.
Not all episodes succeed (scripted expert fails on some extreme positions — ~58% success).
That's OK — the model learns from the full trajectory, including partial successes.

### 4.2 Train the Model

```bash
MUJOCO_GL=egl python3 scripts/train_bimanual.py \
    --demos data/bimanual_demos_phase_f \
    --output data/bimanual_checkpoints_phase_f \
    --epochs 300
```

**What happens during training:**

```
For each epoch (300 total):
    For each batch of 32 random samples from all episodes:
        1. Load image from HDF5 (480x640x3 uint8)
        2. Apply image augmentation (F0):
            - ColorJitter(brightness=0.3, contrast=0.3, sat=0.2, hue=0.05)
            - GaussianBlur (30% chance)
            - RandomResizedCrop (50% chance, scale 0.85-1.0)
        3. ImageNet normalize → (224, 224, 3) float tensor
        4. Load state vector (14 pos + 14 vel = 28D)
        5. Load action chunk (next 20 timesteps × 14D = 20×14 target)
        6. Forward pass through ACT:
            image → ResNet18 → 512D visual features
            state → Linear → 256D state features
            [visual; state; task_embedding] → Transformer decoder → 20×14 actions
        7. Loss = MSE(predicted_chunk, target_chunk)
        8. Backprop + AdamW optimizer step

    Every 50 epochs: save checkpoint
    Track best loss: save best.pt
```

**Training time:** ~20-30 minutes on RTX 4050 for 300 epochs with 50 episodes.

**What to watch for in the training log:**

```
Good:   Loss drops from ~0.01 to ~0.0001 over 300 epochs, smooth curve
Bad:    Loss plateaus early (model underfitting) → need more epochs or lower LR
Bad:    Loss oscillates wildly → learning rate too high, reduce from 1e-4 to 5e-5
Bad:    Loss drops then rises → overfitting, need more data or stronger augmentation
```

### 4.3 Evaluate Generalization

```bash
# Test against all 5 distributions
MUJOCO_GL=egl python3 scripts/eval_generalization.py \
    --checkpoint data/bimanual_checkpoints_phase_f/best.pt \
    --mode bimanual \
    --episodes 20 \
    --train-noise-x 0.08 --train-noise-y 0.06 \
    --train-random-start 0.25
```

**Important:** The `--train-noise-x/y` and `--train-random-start` values must match
what you used during demo generation. The eval script uses these as the baseline
and applies multipliers (1.0x for in-dist, 1.5x for OOD) on top.

**Reading the output:**

```
SUMMARY (bimanual)
Task                                          in-dist   position     visual    posture   combined
-------------------------------------------------------------------------------------------------
pick up the green box with both hands           60.0%      50.0%      55.0%      50.0%      40.0%
-------------------------------------------------------------------------------------------------
OVERALL                                         60.0%      50.0%      55.0%      50.0%      40.0%
```

What matters:
- **in_dist vs old in_dist**: Did the harder training data hurt in-dist performance?
- **OOD-position gap**: `in_dist - ood_position` — smaller gap = better generalization
- **OOD-combined**: The hardest test — everything changes at once

---

## 5. Key Concepts Deep Dive

### 5.1 Why Augmentation AND Randomization?

They complement each other:

```
Image augmentation (F0):
  + Free (no sim cost)
  + Regularizes the image encoder (ResNet18)
  + Prevents overfitting to exact pixel values
  - Only modifies pixel-level appearance
  - Doesn't change object positions or physics

Domain randomization (F1-F4):
  + Changes actual scene geometry and physics
  + Box appears at genuinely different positions
  + Robot starts in different configurations
  - Costs simulation time (need to generate new demos)
  - Some random configs are physically impossible (need retries)
```

Using both gives maximum coverage: augmentation handles pixel-level variation,
randomization handles geometric/physical variation.

### 5.2 The Ego Camera Field of View Problem

From the visualization, you saw that the ego camera (mounted on the robot's torso,
looking forward) has a narrow downward view of the table. The box often appears
at the very bottom edge of the image, or partially out of frame.

This is actually **realistic** for a head-mounted camera — humans also have limited
peripheral vision downward. The model must learn to use the limited visual information
available in the bottom portion of the image.

**Implications for the model:**
- The ResNet18 encoder must learn to focus on the bottom ~30% of the image
- Small position changes cause the box to enter/exit the field of view
- Camera perturbation (+/-2cm, +/-3deg) makes this even more challenging
- The model must be robust to partial visibility

### 5.3 Physics-Based vs Kinematic Demo Generation

| Aspect | Kinematic (single-arm) | Physics (bimanual) |
|--------|----------------------|-------------------|
| Sim function | `mj_forward()` | `mj_step()` |
| Grasping | Weld constraint (instant attach) | Friction from both palms |
| Success rate | ~100% (IK always works) | ~58% with wide randomization |
| Realism | Low (no forces, no contact) | High (real contact, gravity) |
| Speed | ~0.6s/episode | ~1.4s/episode |

The bimanual scripted expert fails more often because:
1. Wider box positions → IK for both arms harder to satisfy simultaneously
2. Random arm start → trajectory to box is longer/more complex
3. Physics friction → box can slip if squeeze angle isn't right
4. Box can fall off table if pushed too hard during approach

These failures are actually **useful training data** — the model sees what failure
looks like and learns to avoid those configurations.

### 5.4 The Action Space

**Single-arm:** 7 DOF (right arm joint positions)
```
action = [shoulder_pitch, shoulder_roll, shoulder_yaw,
          elbow_pitch, wrist_roll, wrist_pitch, wrist_yaw]
```

**Bimanual:** 14 DOF (left arm 7 + right arm 7)
```
action = [L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw,
          L_elbow_pitch, L_wrist_roll, L_wrist_pitch, L_wrist_yaw,
          R_shoulder_pitch, R_shoulder_roll, R_shoulder_yaw,
          R_elbow_pitch, R_wrist_roll, R_wrist_pitch, R_wrist_yaw]
```

The ACT model predicts a **chunk** of these: 20 timesteps x 14 dimensions = 280 values
per forward pass. Temporal ensembling smooths overlapping chunks.

---

## 6. File Reference

| File | Purpose | Phase F Changes |
|------|---------|-----------------|
| `scripts/act_model.py` | ACT model + DemoDataset | F0: Added image augmentation to `__getitem__` |
| `scripts/train_bimanual.py` | Bimanual training loop | F0: Added augmentation to BimanualDemoDataset, `--no-augment` flag |
| `scripts/generate_demos.py` | Single-arm demo generator | F1: `--noise-range`, F2: `--random-start`, F4: `--domain-rand` |
| `scripts/generate_bimanual_demos.py` | Bimanual demo generator | F1: `--noise-x/y`, F2: `--random-start`, F4: `--domain-rand` |
| `scripts/physics_sim.py` | Physics sim wrapper | F1: parameterized `reset_with_noise()`, F2: `random_arm_start()` |
| `scripts/domain_randomization.py` | **NEW** — DomainRandomizer class | F3+F4: All visual randomization |
| `scripts/eval_generalization.py` | **NEW** — Generalization eval | F5: 5 test distributions, single-arm + bimanual |
| `scripts/evaluate_bimanual.py` | Bimanual evaluator | F5: Added `noise_x/y`, `random_start` params |
| `sim/g1_with_camera.xml` | MuJoCo scene | F4: Added 3 distractor objects |

---

## 7. Ablation Study Design

To prove each Phase F component contributes, you would train 4 models:

| Model | Augmentation (F0) | Wide Position (F1) | Random Start (F2) | Visual Rand (F3+F4) |
|-------|:-:|:-:|:-:|:-:|
| Baseline | - | - | - | - |
| +AugOnly | Y | - | - | - |
| +WidePos | - | Y | - | - |
| +Full (Phase F) | Y | Y | Y | Y |

**Expected ranking on OOD-combined:**
```
Full > WidePos > AugOnly > Baseline
```

**Expected ranking on in-dist:**
```
Baseline >= AugOnly > Full >= WidePos
```

Baseline wins on in-dist because it's the easiest task (narrow range, fixed everything).
But Full wins on the metric that matters: total area across all distributions.

---

## 8. Common Issues and Solutions

### "Training loss doesn't go below 0.001"
More data variation = harder optimization. Try:
- Increase epochs from 300 to 500
- Lower learning rate from 1e-4 to 5e-5
- Reduce augmentation intensity (smaller ColorJitter ranges)

### "0% success on evaluation"
The model probably isn't loading correctly, or the eval noise params don't match training.
Check:
- `--train-noise-x/y` values match demo generation
- Checkpoint file contains the right model (check `ckpt['config']`)

### "Bimanual demos fail >50% of the time"
With wide randomization, the scripted expert struggles. Options:
- Reduce noise to +/-5cm instead of +/-8cm
- Reduce `--random-start` from 0.25 to 0.15
- Generate more episodes (100 instead of 50) to get enough successes

### "Visual randomization looks broken (artifacts)"
Check that `randomizer.restore()` is called after each episode.
If not, visual changes accumulate and become unrealistic.

---

## 9. What's Next After Phase F

Phase F teaches the model to handle **variation within a single task**.
The remaining challenges:

1. **Multi-task generalization** — can one model handle reach + grasp + pick + place + bimanual?
2. **Language conditioning** — does the model actually use the task label, or ignore it?
3. **Sim-to-real** — does visual randomization in MuJoCo transfer to real camera images?
4. **Longer horizons** — multi-step tasks ("pick up red cube, place it on blue plate, then pick up green box")

Phase F is the foundation — without it, none of these harder problems can be solved
because the model would just memorize instead of learning.
