# 02 — Scripted Expert Demo Generation (Phase B4.5)

> **What this covers:** How we built a scripted expert that auto-generates reach/grasp/pick
> demonstrations for training the ACT (Action Chunking with Transformers) model.
> This replaces manual teleoperation — instead of a human operating the arm 60 times,
> a programmatic expert uses inverse kinematics to produce perfect demonstrations.

---

## 1. The Big Picture: Why Scripted Demos?

The VLA pipeline needs training data: pairs of (observation, action) that show the robot
successfully performing tasks. There are two ways to get this:

| Approach | Pros | Cons |
|----------|------|------|
| **Manual teleoperation** | Most realistic | Slow (hours), inconsistent, human error |
| **Scripted expert** | Fast (seconds), perfect, reproducible | Less variation, simplified physics |

We chose the scripted expert because:
1. We need 20+ demos per task × 3 tasks = 60+ episodes minimum
2. Manual teleop of a 7-DOF arm through keyboard is painful and error-prone
3. The scripted expert can be re-run with different seeds for unlimited data
4. Once the VLA model is trained, it will generalize beyond the exact demo trajectories

---

## 2. Architecture: Two-Pass Kinematic Pipeline

The core insight: **separate planning from recording**. Don't try to do IK and record
observations in the same pass — it leads to messy state management.

```
Pass 1: PLANNING (pure math, no rendering)
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────┐
│  Reset +  │───▶│  IK solve │───▶│  IK solve    │───▶│  Joint       │
│  add cube │    │  waypoint1│    │  waypoint2   │    │  configs     │
│  noise    │    │  (8cm up) │    │  (2cm above) │    │  [q0,q1,q2] │
└──────────┘    └───────────┘    └──────────────┘    └──────┬───────┘
                                                            │
                                            ┌───────────────▼──────────┐
                                            │  Interpolate in          │
                                            │  joint space → (T, 7)   │
                                            │  trajectory array        │
                                            └───────────────┬──────────┘
                                                            │
Pass 2: RECORDING (renders camera, produces HDF5)           │
┌──────────┐    ┌───────────────────┐    ┌─────────────────▼──────────┐
│  Reset + │───▶│  For each frame:  │───▶│  Record obs:               │
│  restore │    │   set qpos        │    │    - joint positions (29)  │
│  cube    │    │   mj_forward      │    │    - joint velocities (29) │
│  noise   │    │   render camera   │    │    - camera frame (640×480)│
└──────────┘    │   record obs+act  │    │    - action (29 targets)   │
                └───────────────────┘    └──────────────────────────── ┘
```

### Why two passes?

During Pass 1 (IK solving), the simulation state gets modified — arm joints move to find
the solution. We need a clean starting state for recording, so we reset the sim and replay
the solved trajectory. The cube noise (random ±3cm offset) is saved before Pass 1 and
restored after reset in Pass 2.

---

## 3. Inverse Kinematics: How the Robot Plans Arm Movements

### 3.1 The Problem

Given a target position in 3D space (e.g., "2cm above the cube"), find the 7 joint angles
of the right arm that place the hand there.

This is **inverse kinematics (IK)** — going from Cartesian space (x, y, z) to joint space
(7 angles). It's harder than forward kinematics because:
- There may be multiple solutions (7 joints for 3D position = 4 redundant DOF)
- There may be no solution (target is out of reach)
- The relationship is highly nonlinear

### 3.2 The Jacobian Method

We use **iterative damped least-squares IK**. Here's the math:

**Step 1: Compute position error**
```
error = target_xyz - current_hand_position    # (3,) vector
```

**Step 2: Get the Jacobian matrix**

The Jacobian J is a 3×7 matrix where entry J[i,j] tells you: "if I increase joint j by
a tiny amount, how much does the hand move in direction i?"

```python
jacp = np.zeros((3, model.nv))           # full Jacobian (3 × 35 DOFs)
mujoco.mj_jacSite(model, data, jacp,     # MuJoCo computes it for us
                   None, hand_site_id)
J = jacp[:, RIGHT_ARM_DOF]               # extract 3×7 for arm joints only
```

**Step 3: Solve for joint updates (damped least-squares)**

We want `dq` (joint angle changes) such that `J @ dq ≈ error`. Direct pseudoinverse
`J†` is unstable near singularities, so we add damping:

```
dq = J^T @ (J @ J^T + λ²I)^(-1) @ error
```

Where λ = 0.05 is a small damping factor. This is equivalent to:
- "Move joints to reduce error, but don't go crazy if the Jacobian is ill-conditioned"
- Near singularity, λ² prevents division by zero in the matrix inverse

```python
JJT = J @ J.T + damping**2 * np.eye(3)   # 3×3 regularized matrix
dq = J.T @ np.linalg.solve(JJT, error)   # 7-vector of joint updates
```

**Step 4: Limit step size and apply**

```python
if np.linalg.norm(dq) > step:         # step = 0.02 rad max
    dq *= step / np.linalg.norm(dq)   # normalize to max step
q_new = current_q + dq
q_new = np.clip(q_new, joint_lo, joint_hi)  # respect joint limits
```

**Step 5: Update sim and repeat**

```python
data.qpos[arm_qpos_adr] = q_new
mujoco.mj_forward(model, data)        # recompute all positions
# → back to step 1, with updated hand position
```

### 3.3 Parameters (tuned through trial and error)

| Parameter | Value | Why |
|-----------|-------|-----|
| `step` | 0.02 rad | Small enough for smooth convergence, large enough to not be slow |
| `damping` | 0.05 | Prevents instability near workspace boundary |
| `tol` | 0.01 m | 1cm accuracy — good enough for grasping a 5cm cube |
| `max_iter` | 500 | Converges in ~10-20 iterations typically; 500 is a safety net |

### 3.4 Important: Joint Limits vs. Torque Limits

A critical bug we encountered: the code was clipping joint angles to `ctrlrange` (±25 Nm)
instead of `jnt_range` (±1-3 rad). For G1's motor actuators:

| | `ctrlrange` | `jnt_range` |
|-|-------------|-------------|
| **What it is** | Torque limit (Nm) | Joint position limit (radians) |
| **Shoulder pitch** | [-25, 25] | [-3.089, 2.670] |
| **Wrist pitch** | [-5, 5] | [-1.614, 1.614] |
| **For position clamping** | ❌ WRONG | ✅ CORRECT |

The fix: build `arm_pos_lo/hi` from `model.jnt_range`, not `model.actuator_ctrlrange`.

---

## 4. Kinematic vs. Dynamic Simulation

### 4.1 Two MuJoCo Functions

| Function | What it does | Physics? | Use case |
|----------|-------------|----------|----------|
| `mj_forward(model, data)` | Computes positions from qpos | No | Kinematics only |
| `mj_step(model, data)` | Full physics: forces → accelerations → new qpos | Yes | Real-time sim |

### 4.2 Why We Use Kinematic Mode

For scripted demo generation, we just need:
1. Set arm joint angles → compute hand position (for rendering)
2. No forces, no gravity, no contacts needed

With `mj_step` (full physics):
- ❌ PD controller needed to track desired joint angles
- ❌ Arm has inertia — can't instantly reach target
- ❌ Physics forces push the cube around unpredictably
- ❌ Gain tuning nightmare (KP/KD per joint)

With `mj_forward` (kinematics only):
- ✅ Set qpos directly → instant, perfect positioning
- ✅ No forces = no disturbances
- ✅ Zero gain tuning
- ✅ Deterministic, reproducible

### 4.3 The Weld Constraint Problem

One consequence of kinematic mode: **equality constraints don't work**.

MuJoCo's weld constraint (used to simulate grasping — "glue cube to hand") works by
applying constraint forces during `mj_step`. With `mj_forward`, no forces are computed,
so the cube stays put even when the weld is "active".

**Solution:** Manually enforce the weld in code:

```python
def step_frame(self):
    self.data.qpos[self.arm_qpos_adr] = self.target_pos[RIGHT_ARM_CTRL]
    mujoco.mj_forward(self.model, self.data)

    # Manual weld: if grasping, teleport cube to hand
    if self.data.eq_active[self.weld_id]:
        hand_xyz = self.data.site_xpos[self.hand_site_id]
        self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = hand_xyz
        mujoco.mj_forward(self.model, self.data)  # recompute with new cube pos
```

---

## 5. The Three Tasks

Each task builds on the previous one:

### 5.1 Reach: "reach the red cube"

```
Waypoints:
  home → 8cm above cube → 2cm above cube

Timeline (85 frames = 2.8 sec at 30 FPS):
  [40 frames: approach] [25 frames: descend] [20 frames: hold]
```

The hand ends ~2cm above the cube center. Success metric: hand-cube distance < 6cm.

### 5.2 Grasp: "grasp the red cube"

```
Same waypoints as reach, but:
  After final waypoint → activate weld constraint

The weld "glues" the cube to the hand (simulating a power grasp).
```

### 5.3 Pick: "pick up the red cube"

```
Waypoints:
  home → 8cm above → 2cm above → 17cm above (lift)

Timeline (120 frames = 4.0 sec):
  [40 frames: approach] [25 frames: descend] [35 frames: lift] [20 frames: hold]

Grasp activates after waypoint 1 (the 2cm-above position),
then the lift waypoint carries the cube upward.
```

### 5.4 Randomization

Each episode has the cube at a slightly different position (±3cm in x and y):

```python
def reset_with_noise(self, rng):
    self.reset()
    self.data.qpos[self.cube_qpos_adr + 0] += rng.uniform(-0.03, 0.03)  # x noise
    self.data.qpos[self.cube_qpos_adr + 1] += rng.uniform(-0.03, 0.03)  # y noise
```

This teaches the VLA model that the cube isn't always at the exact same spot — it must
use the camera to find it.

---

## 6. The Unified Recording Function: `_kinematic_record`

All three tasks flow through one function. Here's the complete logic:

```python
def _kinematic_record(sim, rng, waypoints, frames_per, hold=15,
                      grasp_after=False, grasp_after_wp=-1):

    # ─── Save cube noise ───
    cube_dx = sim.data.qpos[cube_adr + 0] - sim.model.qpos0[cube_adr + 0]
    cube_dy = sim.data.qpos[cube_adr + 1] - sim.model.qpos0[cube_adr + 1]

    # ─── Pass 1: IK solve ───
    configs = [sim.arm_q.copy()]        # start = current arm config
    for wp in waypoints:
        sim.solve_ik(wp)                # moves arm to target
        configs.append(sim.arm_q.copy())

    # ─── Interpolate ───
    traj = interpolate_trajectory(configs, frames_per)  # (T, 7) smooth path

    # ─── Pass 2: Reset + replay ───
    sim.reset()
    sim.data.qpos[cube_adr + 0] += cube_dx   # restore cube noise
    sim.data.qpos[cube_adr + 1] += cube_dy
    mujoco.mj_forward(sim.model, sim.data)

    # Record each frame
    for t in range(len(traj)):
        sim.target_pos[RIGHT_ARM_CTRL] = traj[t]
        rec.record(sim)           # saves joints + camera + action
        sim.step_frame()          # kinematic update

        # Activate grasp at the right moment
        if grasp_after_wp >= 0 and t >= wp_frame_idx[grasp_after_wp] - 1:
            sim.set_weld(True)

    # Grasp after all waypoints (for "grasp" task)
    if grasp_after:
        sim.set_weld(True)

    # Hold final pose
    for _ in range(hold):
        rec.record(sim)
        sim.step_frame()

    return rec.pack()   # dict of numpy arrays → saved to HDF5
```

---

## 7. HDF5 Output Format

Each episode is saved as an HDF5 file with this structure:

```
episode_0042.hdf5
├── obs/
│   ├── joint_positions   (T, 29)     float32  — actuator lengths
│   ├── joint_velocities  (T, 29)     float32  — actuator velocities
│   └── camera_frames     (T, 480, 640, 3) uint8 — RGB egocentric view
├── action                (T, 29)     float32  — target joint positions
└── attrs:
    ├── episode_id:       42
    ├── task_description: "pick up the red cube"
    ├── fps:              30
    ├── timestamp:        "2026-03-02T..."
    └── num_frames:       120
```

**Key design choices:**
- `task_description` is a string attribute — this is what enables language-conditioned training
- `action` contains ALL 29 actuator targets, not just the 7 arm joints (ACT predicts full body)
- Camera frames are RGB (3 channels), 640×480 — matches common VLA input sizes
- gzip compression keeps file sizes reasonable (~15 MB per episode for 85 frames)

---

## 8. Arm Workspace and Reachability

### 8.1 Robot Geometry

```
                    Shoulder (0, -0.10, 1.08)
                    │
                    │ upper arm: ~0.25m
                    │
                    ├── Elbow
                    │
                    │ forearm: ~0.16m
                    │
                    ├── Wrist
                    │
                    │ hand offset: ~0.10m (from right_hand_site)
                    │
                    └── Hand tip

Total reach from shoulder: ~0.51m
```

### 8.2 Comfortable Workspace

```
         Y (left)
         │
   ------┼-------
   │     │      │
   │  REACHABLE │
   │     │      │
   ------┼-------───── X (forward)
         │
    x: 0.20 → 0.40
    y: -0.20 → 0.05
    z: 0.80 → 1.00

    Sweet spot: (0.30, -0.10, 0.85)
```

**Critical lesson:** The original cube was at (0.6, 0, 0.825) — that's 0.66m from the
shoulder, but the arm can only reach 0.51m. The IK correctly reported failure (singular
Jacobian). We moved everything to (0.3, -0.1) which is in the comfortable middle of the
workspace.

---

## 9. Scene Layout (g1_with_camera.xml)

```
Top-down view:

              +X (forward)
              │
              │   ┌─────────┐
              │   │  TABLE  │  (0.3, -0.1)  40×40cm
              │   │  ■cube  │  cube at table center
              │   └─────────┘
              │
     ─────── ROBOT ────────── +Y (left)
              (0, 0)
              │
              │
    scene_camera at (0.3, -0.8, 1.6)
    looking up toward robot hand+table
```

The scene camera provides a debug/monitoring view. The ego_camera is in the robot's
torso (head height), providing the egocentric view that the VLA model will use.

---

## 10. Bugs We Encountered and Fixed

### Bug 1: IK windup (early version)

**Symptom:** Arm flails wildly, joints hit limits, never converges.
**Cause:** IK deltas were accumulated on a target variable instead of computing from
current position each step. The error grew instead of shrinking.
**Fix:** Always compute `error = target - CURRENT_hand_pos`, not `target += dq`.

### Bug 2: Clipping to torque limits

**Symptom:** Joints appeared unclamped (values reaching ±25 rad).
**Cause:** Code used `model.actuator_ctrlrange` (torque ±25Nm) for position clamping.
**Fix:** Use `model.jnt_range` (actual joint angle limits ±1-3 rad).

### Bug 3: Cube out of reach

**Symptom:** IK converges to a stretched-out pose, d=0.22m from target.
**Cause:** Cube at (0.6, 0, 0.825) was 0.66m from shoulder; max reach is 0.51m.
**Fix:** Moved table+cube to (0.3, -0.1).

### Bug 4: PD tracking failure

**Symptom:** Physics-based playback — arm barely moves despite large joint targets.
**Cause:** KP=12 too low for the arm's inertia. mj_step dynamics can't track fast IK.
**Fix:** Abandoned PD-based mj_step entirely. Switched to kinematic mj_forward.

### Bug 5: Cube displacement during physics

**Symptom:** Cube slides away even though arm trajectory is correct.
**Cause:** mj_step applies contact forces — arm approach pushes cube via collisions.
**Fix:** Pure kinematic mode (mj_forward) has no forces → cube stays put.

### Bug 6: `generate_reach` corrupting cube position

**Symptom:** Reach episodes report d≈0.224 but grasp works fine (d≈0.03).
**Cause:** Leftover code from an earlier refactoring in `generate_reach` — it called
`reset_with_noise(rng)` twice, consumed extra rng values, then manually computed cube
position with a wrong formula (`cube[0] - 0.3 + 0.1`).
**Fix:** Made `generate_reach` match the clean pattern of `generate_grasp`/`generate_pick`.

### Bug 7: Weld doesn't work in kinematic mode

**Symptom:** Pick task lifts hand but cube stays on table, d≈0.16.
**Cause:** `mj_forward` computes positions from qpos but doesn't solve constraint forces.
Weld constraint is a force-based mechanism that only works with `mj_step`.
**Fix:** Manual weld: if `eq_active[weld_id]`, set `cube_qpos = hand_xyz`.

---

## 11. Running the Demo Generator

```bash
cd ~/projects/humanoid_vla

# Generate all 60 demos (20 per task)
MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20

# Generate just reach demos
MUJOCO_GL=egl python3 scripts/generate_demos.py --task reach --episodes 20

# Custom seed for different randomization
MUJOCO_GL=egl python3 scripts/generate_demos.py --all-tasks --episodes 20 --seed 123

# Output goes to data/demos/ by default
ls data/demos/
# episode_0000.hdf5 ... episode_0059.hdf5
```

**Output from the final successful run:**
```
Generating 20 episodes: 'reach the red cube'      → 20/20 OK  (d=0.025-0.030)
Generating 20 episodes: 'grasp the red cube'      → 20/20 OK  (d=0.000)
Generating 20 episodes: 'pick up the red cube'    → 20/20 OK  (d=0.000)
Convergence: 60/60 (100%)
```

---

## 12. What This Data Enables (Next Steps)

The 60 HDF5 episodes are the raw training data. The pipeline going forward:

```
data/demos/*.hdf5
       │
       ▼
scripts/convert_to_lerobot.py    ← Convert HDF5 → LeRobot dataset format
       │                            + add language labels per episode
       ▼
LeRobot Dataset
       │
       ▼
scripts/train_act.py             ← Train ACT model on dataset
       │                            Input: camera + joints + language
       │                            Output: 29 joint action targets
       ▼
Trained ACT Model (.pt)
       │
       ▼
vla_node.py (ROS2)              ← Real-time inference: camera → model → joints
       │                            30 Hz loop, language instruction from RosClaw
       ▼
MuJoCo Sim                      ← Robot executes predicted actions
```

The key thing the language label enables: **one model handles all three tasks**.
Instead of training three separate models, the ACT model receives "reach the red cube"
or "pick up the red cube" as input, and produces different actions accordingly.

---

## 13. Files Changed in This Iteration

| File | What Changed |
|------|-------------|
| `scripts/generate_demos.py` | **NEW** — entire scripted expert + IK solver |
| `sim/g1_with_camera.xml` | Table (0.3,-0.1), cube (0.3,-0.1,0.825), camera adjusted |
| `sim/models/g1_29dof.xml` | `right_hand_site` added to `right_wrist_yaw_link` body |
| `ros2_ws/.../mujoco_sim.py` | `get_site_xpos`, `get_body_xpos`, `get_site_jacp`, `set_grasp` |
| `ros2_ws/.../bridge_node.py` | `/grasp` service, `/hand_pos` + `/cube_pos` publishers |
| `tasks/lessons.md` | L013-L016: ctrlrange, arm reach, kinematic IK, weld |
| `tasks/todo.md` | B4.5 milestone documented as complete |

**Commit:** `de9e9da` — "Phase B4.5: scripted expert demo generation — 100% convergence"
