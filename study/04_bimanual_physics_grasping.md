# 04 — Bimanual Physics-Based Grasping (Phase C2)

> **What this covers:** Extending from single-arm weld-based manipulation to bimanual
> physics-based grasping. Covers mj_step vs mj_forward, PD torque control, contact
> physics, friction cones, compliance grasping, bimanual coordination, and training
> a separate bimanual ACT model.

---

## 1. Motivation — Why Physics-Based Grasping?

Phase C achieved 86.2% success on single-arm tasks using `mj_forward` (kinematic
positioning) with a weld constraint for grasping. While great for learning ACT
fundamentals, this approach has a key unrealism: the cube teleports to the hand
and is rigidly attached. No contact forces, no friction, no possibility of dropping.

**Real robots grasp via friction.** Two hands squeeze an object, applying normal
force to create friction that resists gravity. If squeeze force is too low, the
object slips. If too high, the object (or robot) may be damaged.

| Phase C (single-arm) | Phase C2 (bimanual) |
|-----------------------|---------------------|
| `mj_forward` — kinematic | `mj_step` — full dynamics |
| Weld constraint grasp | Friction-only grasp |
| 1 arm, 7 DOF | 2 arms, 14 DOF |
| Position control (direct qpos) | Torque control (PD + gravity comp) |
| No slip possible | Slip = failure |
| Cube: 4cm, no mass matters | Box: 20×15×15cm, 0.3kg |

---

## 2. Physics Foundations: mj_step vs mj_forward

### 2.1 mj_forward — Kinematic Mode

```python
data.qpos[joint_addresses] = desired_positions
mujoco.mj_forward(model, data)
```

`mj_forward` computes:
1. Positions → forward kinematics (site/body positions)
2. Jacobians, contact geometry
3. Does NOT integrate equations of motion
4. No forces applied, no time progresses

**Use case:** Instantaneous "teleport" — useful for IK solving, visualization,
and tasks where physics fidelity isn't critical.

### 2.2 mj_step — Full Dynamics

```python
data.ctrl[:] = torques  # set actuator commands
mujoco.mj_step(model, data)  # integrate one timestep
```

`mj_step` computes:
1. Forward kinematics (like mj_forward)
2. Contact detection and constraint forces
3. Equations of motion: $M\ddot{q} + c(q, \dot{q}) = \tau + J^T f_{\text{contact}}$
4. Time integration: $q_{t+1} = q_t + \dot{q}_t \cdot dt$

Where:
- $M$ — mass matrix (inertia)
- $c(q, \dot{q})$ — Coriolis + gravitational forces
- $\tau$ — applied torques (from `data.ctrl`)
- $J^T f_{\text{contact}}$ — contact constraint forces

**Use case:** Realistic simulation with forces, contacts, and dynamics.

### 2.3 Implications for Robot Control

With `mj_forward`, we directly set joint positions — the robot is infinitely stiff.
With `mj_step`, we command torques and the physics engine determines resulting motion.
This means:
- Joints can drift if torques don't compensate gravity
- Contacts are resolved by the constraint solver (no fake welds)
- The robot can fall over if not properly supported

---

## 3. PD Torque Control with Gravity Compensation

### 3.1 The PD Controller

A PD (Proportional-Derivative) controller generates torques to track desired positions:

$$\tau = K_p (q_{\text{des}} - q) - K_d \dot{q} + \tau_{\text{bias}}$$

Where:
- $K_p$ — proportional gain (stiffness). Higher = stiffer tracking
- $K_d$ — derivative gain (damping). Higher = less oscillation
- $q_{\text{des}}$ — desired joint position (from policy/IK)
- $q$ — current joint position
- $\dot{q}$ — current joint velocity
- $\tau_{\text{bias}}$ — gravity + Coriolis compensation (`qfrc_bias`)

### 3.2 Our Gain Selection

```python
# Per-actuator gains
# Shoulders (large joints): Kp=40, Kd=4
# Elbows: Kp=40, Kd=4
# Wrists (small joints): Kp=10, Kd=1
_KP = np.array([...])  # 29 values
_KD = np.array([...])  # 29 values
```

**Why different gains?** Shoulder joints carry more mass (entire arm + payload)
and need higher stiffness. Wrist joints move less mass and benefit from lower
gains to avoid oscillation. The 10:1 Kp:Kd ratio provides critical damping.

### 3.3 Gravity Compensation

Without gravity compensation, the arm falls under its own weight. MuJoCo provides
`data.qfrc_bias` which contains the torques needed to exactly cancel gravity and
Coriolis forces at the current configuration:

```python
tau_bias = data.qfrc_bias[actuated_dof_start:actuated_dof_end]
tau = Kp * (q_des - q) - Kd * q_dot + tau_bias
```

This is called "computed torque" or "inverse dynamics feedforward." The PD terms
then only need to handle tracking errors, not fight gravity.

### 3.4 Torque Clipping

All torques are clipped to the actuator control range:
```python
tau = np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])
# Shoulders: ±25 Nm, Wrists: ±5 Nm
```

This prevents unrealistic forces and matches physical actuator limits.

---

## 4. Contact Physics and Friction

### 4.1 How MuJoCo Resolves Contacts

When two geoms overlap, MuJoCo:
1. Detects penetration depth and contact normal
2. Creates contact constraints
3. Solves for constraint forces that prevent interpenetration
4. Applies these forces at each timestep

Key parameters:
- **condim** — contact dimensionality (1=normal only, 3=normal+tangential, 4=+torsion, 6=full)
- **friction** — [slide, spin, roll] friction coefficients
- **solimp** — solver impedance (stiffness of contact)
- **solref** — solver reference (natural frequency and damping)

### 4.2 Friction Cones

The Coulomb friction model says tangential friction force $f_t$ is bounded by:

$$|f_t| \leq \mu \cdot f_n$$

Where $\mu$ is the friction coefficient and $f_n$ is the normal force. This forms
a "friction cone" — the set of force directions that maintain static friction.

For our box grasping:
- Normal force = squeeze force from PD controller pushing palms into box
- Friction force = what holds the box against gravity

**Condition for holding:**
$$\mu \cdot F_{\text{squeeze}} \geq m \cdot g / 2$$

With $\mu = 1.5$, $m = 0.3$ kg, $g = 9.81$ m/s²:
$$1.5 \cdot F_{\text{squeeze}} \geq 1.47 \text{ N}$$
$$F_{\text{squeeze}} \geq 0.98 \text{ N per palm}$$

Our measured forces are 13-14 N per palm — a 14x safety margin.

### 4.3 Why condim=4?

We chose `condim=4` for palm pads and box:
- **condim=1:** Normal force only — no friction, box slides
- **condim=3:** Normal + 2D tangential friction — standard sliding friction
- **condim=4:** + torsional friction — resists rotation
- **condim=6:** + rolling friction — resists rolling

`condim=4` prevents the box from spinning in the palms while being held.
(Though we still observe some rotation — see Lesson L031.)

### 4.4 Our Contact Setup

```xml
<!-- Palm pads: box geoms attached to wrist links -->
<geom name="left_palm_pad" type="box"
      pos="0.08 0.003 0" size="0.04 0.03 0.01"
      friction="1.5 0.005 0.0001" condim="4"
      contype="1" conaffinity="1"/>

<!-- Box: the object to be grasped -->
<geom name="box_geom" type="box" size="0.10 0.075 0.075"
      friction="1.5 0.005 0.0001" condim="4"
      contype="1" conaffinity="1"/>
```

**contype/conaffinity:** MuJoCo's bitwise collision filtering. A contact is
generated only when `(geom1.contype & geom2.conaffinity) != 0` OR
`(geom2.contype & geom1.conaffinity) != 0`. We set both to 1 for collision geoms.

> **Lesson L029:** The G1's rubber hand meshes have `contype=0` — they're visual
> only. We added separate box-shaped palm pad geoms for actual collision.

---

## 5. Compliance Grasping

### 5.1 The Concept

Instead of commanding the palms to the box surface (zero force), we command them
**inside** the box surface. The PD controller generates continuous force trying
to reach the unreachable target. The contact constraint prevents actual penetration,
and the resulting equilibrium produces a steady squeeze force.

```
Palm target (inside box)    Box surface    Palm actual position
        ←  ←  ←  ←  ←  ← | ← ←  ←  ← |→ →  → → → → →
       PD force tries to   Contact force  PD force from
       reach target         pushes back   other palm
```

This is elegant because:
1. No force controller needed — PD position control naturally creates force
2. Self-regulating — deeper target = more force
3. Bilateral — both palms push inward, creating symmetric squeeze

### 5.2 Implementation

```python
# Target positions during squeeze phase:
# Right hand: box center - (half_width - penetration_depth)
# Left hand:  box center + (half_width - penetration_depth)
SQUEEZE_OFFSET_Y = 0.045  # box half-width is 0.075m, so 3cm inside surface
```

The 3cm penetration target is chosen to produce ~13N force with our PD gains.
Too little → insufficient friction → box slips. Too much → wrist torque limits
saturate → control becomes unreliable.

---

## 6. Joint Freezing and Stability

### 6.1 The Problem

Under `mj_step`, ALL joints experience dynamics. The G1 has:
- 6 DOF floating base (pelvis)
- 12 DOF legs (6 per leg)
- 3 DOF waist
- 14 DOF arms (7 per arm)

Without proper control of legs and waist, the robot tilts, legs buckle, and
the torso drifts. The arm trajectories become meaningless because the shoulder
reference frame moves.

### 6.2 The Solution — Freeze Non-Arm Joints

Every physics substep, we:
1. Set torques for arms via PD controller
2. Reset pelvis (qpos[:7]) and velocity (qvel[:6]) to initial values
3. Reset all leg/waist joint positions and velocities to initial values

```python
for _ in range(SUBSTEPS):
    data.ctrl[:] = compute_pd_torques()
    mujoco.mj_step(model, data)
    # Freeze base
    data.qpos[:7] = base_qpos
    data.qvel[:6] = 0.0
    # Freeze legs + waist
    data.qpos[locked_qpos_adr] = locked_qpos_vals
    data.qvel[locked_qvel_idx] = 0.0
```

> **Lesson L028:** Initially we only froze the pelvis. The waist joints drifted
> under contact reaction forces, tilting the torso. Hands moved 8cm from targets.
> Freezing ALL non-arm joints solved this.

### 6.3 Substep Frequency

```python
CONTROL_HZ = 30    # Policy rate (30 Hz)
PHYSICS_HZ = 500   # Physics timestep (500 Hz)
SUBSTEPS = 16      # ≈ 500/30
```

Multiple physics substeps per control frame ensures stable contact resolution.
At 500 Hz, each timestep is 2ms — well within MuJoCo's stability limits for
our contact stiffness.

---

## 7. Bimanual Coordination

### 7.1 Trajectory Design

The bimanual trajectory has 6 waypoints, applied symmetrically:

| Phase | Left Hand | Right Hand | Purpose |
|-------|-----------|------------|---------|
| 1. Home | rest position | rest position | Starting pose |
| 2. Pre-approach | above + left of box | above + right of box | Clear path |
| 3. Approach | 5cm left of box | 5cm right of box | Align at box height |
| 4. Squeeze | inside left surface | inside right surface | Generate contact force |
| 5. Lift | squeeze pos + 15cm up | squeeze pos + 15cm up | Lift with friction |
| 6. Hold | lift position | lift position | Stabilize |

**Timing variation:** ±10% random noise on segment durations for training diversity.

### 7.2 IK for Both Arms

We solve IK independently for each arm:
```python
# Left arm: 7 joints (ctrl indices 15-21)
# Right arm: 7 joints (ctrl indices 22-28)
solve_ik_left(target_xyz)   # position-only, damped least squares
solve_ik_right(target_xyz)  # position-only, damped least squares
```

**Position-only IK** (no orientation) works well for lateral squeeze because
the natural wrist orientation from the kinematic chain aligns the palm pads
with the box sides.

> **Lesson L030:** We expected to need orientation IK constraints, but position-only
> IK naturally produces good palm alignment for the lateral approach.

### 7.3 Coordination Guarantee

Both arms use the **same timing** for each phase. This ensures:
- Simultaneous approach → no knocking the box sideways
- Simultaneous squeeze → balanced forces
- Simultaneous lift → no tilting

The trajectory is generated from a single `plan_bimanual_trajectory()` call
that returns synchronized (left_traj, right_traj) arrays.

---

## 8. Demo Generation

### 8.1 Recording Format

Each episode records:
```python
obs/joint_positions   # (T, 14) — both arms [left_7, right_7]
obs/joint_velocities  # (T, 14) — both arms [left_7, right_7]
obs/camera_frames     # (T, 480, 640, 3) — ego camera
action                # (T, 14) — PD targets [left_7, right_7]
```

Note: we record ONLY arm joints (14), not all 29 actuators. The policy
only controls arms; legs/waist are frozen.

### 8.2 Results

- **30 episodes** generated with ±2cm box position noise
- **100% success rate** (all lifted ≥ 3cm)
- **Lift range:** 4.4 — 9.5 cm (target is 15cm, actual limited by IK workspace)
- **Contact forces:** 13-14 N bilateral
- **~1.5 seconds per episode** generation time

---

## 9. Bimanual ACT Model

### 9.1 Architecture Differences from Single-Arm

| Parameter | Single-Arm (Phase C) | Bimanual (Phase C2) |
|-----------|---------------------|---------------------|
| state_dim | 58 (29 pos + 29 vel) | 28 (14 pos + 14 vel) |
| action_dim | 29 (all actuators) | 14 (both arms only) |
| num_tasks | 4 (reach/grasp/pick/place) | 1 (bimanual pickup) |
| chunk_size | 20 | 20 |
| hidden_dim | 256 | 256 |
| Params | 15.6M total | 15.6M total |

The state is more focused (only arm joints), and we have a single task
(no task conditioning needed yet).

### 9.2 Training Results

```
Epochs: 300
Learning rate: 1e-4 (cosine annealing)
Batch size: 32
Optimizer: AdamW (weight_decay=1e-5)
Training time: 52.5 minutes (RTX 4050)
Final loss: 0.000009
```

The loss converged quickly because:
1. Only 14 action dimensions (vs 29 for single-arm)
2. Single task — no multi-task interference
3. Consistent demonstrations (100% success, similar trajectories)

### 9.3 Evaluation Results

```
Episodes: 20
Success rate: 100% (20/20)
Lift: mean=8.5cm, min=6.5cm, max=10.4cm
Force: L_mean=13.6N, R_mean=13.3N
Time: 0.8s per episode
```

**100% success** — the model perfectly reproduces the bimanual squeeze-and-lift
behavior with full physics. Key factors:
- Consistent demonstrations with little variation
- Temporal ensembling (re-plan every 5 steps, exponential decay k=0.01)
- PD controller provides natural compliance and disturbance rejection

---

## 10. Key Lessons Learned

### L028: Freeze ALL Non-Arm Joints Under mj_step
Freezing only the pelvis was insufficient. Waist joints drifted from contact
reaction forces, tilting the torso and displacing hands by 8cm.
**Fix:** Freeze legs (ctrl[0:12]) + waist (ctrl[12:15]) every substep.

### L029: G1 Hand Meshes Have No Collision
The rubber_hand meshes have `contype=0` — visual only. Physical grasping requires
purpose-built collision geoms (palm pads).

### L030: Position-Only IK Works for Lateral Squeeze
The kinematic chain naturally aligns palm orientation for lateral approach. Full
6-DOF IK was unnecessary for this task geometry.

### L031: Box Rotation During Lift (Known Issue)
The box rotates ~5-10° around vertical during the lift phase. Potential fixes:
- Orientation IK to actively maintain palm alignment
- condim=6 for full rolling friction
- Multiple contact points per palm (second pad per hand)

---

## 11. Comparison: Weld-Based vs Physics-Based Grasping

| Aspect | Weld (Phase C) | Physics (Phase C2) |
|--------|----------------|---------------------|
| **Realism** | Low — teleport + rigid attach | High — contact + friction |
| **Sim fidelity** | Kinematic | Full dynamics |
| **Failure modes** | None (guaranteed grasp) | Slip, drop, misalign |
| **Training diversity** | Must vary trajectories | Physics provides natural variation |
| **Sim-to-real gap** | Large | Much smaller |
| **Control method** | Position (qpos) | Torque (PD + gravity comp) |
| **Complexity** | Simple | Moderate (PD tuning, joint freezing) |
| **Applicable to real robots** | No | Yes — PD controllers are standard |

The physics-based approach is harder to set up but produces policies that are
far more transferable to real hardware. A PD controller with gravity compensation
is exactly how real torque-controlled robots operate.

---

## 12. What's Next

- **Fix box rotation (L031):** Add orientation targets to IK or increase friction
- **More tasks:** bimanual place, handover between hands, coordinated manipulation
- **Domain randomization:** vary box mass, friction, size for robustness
- **RosClaw integration:** send bimanual task commands from Telegram
- **GR00T N1:** test bimanual foundation model (designed for humanoid dual-arm tasks)
