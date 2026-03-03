# Humanoid VLA — Deep Dive Study Guide

**Author:** Generated for Özkan Ceylan  
**Scope:** Everything built so far (Phase A + Phase B infrastructure)  
**Goal:** Understand the structure, math, and engineering decisions — not just what the code does, but *why*

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [MuJoCo Fundamentals](#2-mujoco-fundamentals)
3. [The Robot: Unitree G1](#3-the-robot-unitree-g1)
4. [MJCF XML — How the Scene is Defined](#4-mjcf-xml--how-the-scene-is-defined)
5. [Torque Control & the PD Controller](#5-torque-control--the-pd-controller)
6. [Gravity Compensation — The Math](#6-gravity-compensation--the-math)
7. [Fixed Base — Why and How](#7-fixed-base--why-and-how)
8. [ROS2 Architecture](#8-ros2-architecture)
9. [The Bridge Pattern](#9-the-bridge-pattern)
10. [Threading Model](#10-threading-model)
11. [Camera & Image Pipeline](#11-camera--image-pipeline)
12. [Teleoperation Design](#12-teleoperation-design)
13. [Demo Recording & HDF5 Format](#13-demo-recording--hdf5-format)
14. [Imitation Learning — ACT Model](#14-imitation-learning--act-model)
15. [LeRobot Framework](#15-lerobot-framework)
16. [The Full Data Pipeline](#16-the-full-data-pipeline)
17. [File-by-File Reference](#17-file-by-file-reference)

---

## 1. The Big Picture

### What are we building?

A simulated humanoid robot (Unitree G1) that can be controlled by a **Vision-Language-Action (VLA)** model. The robot:

1. **Sees** the world through an egocentric camera (mounted on its torso)
2. **Understands** natural language task commands ("pick up the red cup")
3. **Acts** by generating joint-level motor commands through a trained neural network

### The control loop we're building toward

```
                    ┌─────────────────────┐
                    │   Camera (480×640)   │
                    │   from MuJoCo sim    │
                    └──────────┬──────────┘
                               │ 30 Hz
                               ▼
                    ┌─────────────────────┐
                    │   VLA Model (ACT)   │
                    │  "reach red cube"   │──── language instruction (once)
                    │  image + joint_pos  │──── observation (every frame)
                    └──────────┬──────────┘
                               │ outputs 29 target joint angles
                               ▼
                    ┌─────────────────────┐
                    │   PD Controller     │
                    │  τ = Kp(q*-q) - Kd·q̇│
                    └──────────┬──────────┘
                               │ 29 torques (Nm)
                               ▼
                    ┌─────────────────────┐
                    │   MuJoCo Physics    │
                    │    500 Hz step      │
                    └─────────────────────┘
```

### Why simulation first?

- A real Unitree G1 costs ~$50,000+
- Breaking stuff in simulation is free
- You need thousands of training steps — can't do that on real hardware without risking damage
- Sim-to-real transfer is a well-studied problem for later

---

## 2. MuJoCo Fundamentals

### What is MuJoCo?

**MuJoCo** (Multi-Joint dynamics with Contact) is a physics engine designed for robotics research. Created by Emanuel Todorov, now owned by Google DeepMind (and open-sourced in 2022).

**What makes it special:**
- Very fast and accurate contact physics (crucial for manipulation)
- Supports smooth, continuous simulation (no jittering like game engines)
- Built-in support for complex robot models (articulated rigid bodies)
- Computes analytical derivatives (useful for optimization-based control)

### The equations of motion

MuJoCo solves this equation at every physics step:

$$M(q)\ddot{q} + c(q, \dot{q}) = \tau + J^T f$$

Where:
- $q$ = **generalized coordinates** (all joint angles + floating base position/orientation)
- $\dot{q}$ = **generalized velocities** (time derivative of $q$)
- $\ddot{q}$ = **generalized accelerations** (what the simulator solves for)
- $M(q)$ = **mass matrix** (inertia of the whole system at configuration $q$)
- $c(q, \dot{q})$ = **bias forces** (gravity + Coriolis + centrifugal forces)
- $\tau$ = **applied torques** (from actuators — what we control)
- $J^T f$ = **contact forces** (from the ground, table, objects)

**In MuJoCo's API, these are stored in `data.qfrc_bias`** — this is the $c(q, \dot{q})$ term. This is the key to gravity compensation (Section 6).

### Key MuJoCo data structures

| API | Python access | What it holds |
|-----|---------------|---------------|
| `model` | `mujoco.MjModel` | The **static** description: geometry, masses, joint limits, everything from XML |
| `data` | `mujoco.MjData` | The **dynamic** state: current positions, velocities, forces, contacts |
| `data.qpos` | array of size `nq` | All generalized positions (root + joints) |
| `data.qvel` | array of size `nv` | All generalized velocities |
| `data.ctrl` | array of size `nu` | Actuator control inputs (what we set) |
| `data.qfrc_bias` | array of size `nv` | Bias forces (gravity + Coriolis) |
| `data.actuator_length` | array of size `nu` | Current joint angles (for actuated joints) |
| `data.actuator_velocity` | array of size `nu` | Current joint velocities (for actuated joints) |

### Simulation step

```python
mujoco.mj_step(model, data)
```

This does:
1. Compute $M(q)$, $c(q, \dot{q})$ from current state
2. Read `data.ctrl` → compute actuator forces
3. Detect contacts → compute contact forces
4. Solve for $\ddot{q}$
5. Integrate: $\dot{q} \leftarrow \dot{q} + \ddot{q} \cdot dt$, $q \leftarrow q + \dot{q} \cdot dt$

One call = one timestep. Our sim runs at 500 Hz → `dt = 0.002s`.

---

## 3. The Robot: Unitree G1

### Physical specs (simulated)

| Property | Value |
|----------|-------|
| Total mass | ~35 kg |
| Standing height | 0.793 m (pelvis centre) |
| Total DOFs (nv) | 35 |
| Actuated joints (nu) | 29 |
| Root DOFs (freejoint) | 6 |
| Total qpos entries (nq) | 36 |

### Why nv=35 and nq=36?

The root body (pelvis) has a **freejoint** — it can translate (x,y,z) and rotate freely in 3D.

- **Position** ($q$): 3 translations + 4 quaternion components = 7 entries for the root
  - `qpos[0:3]` = (x, y, z) position
  - `qpos[3:7]` = (w, x, y, z) quaternion orientation
- **Velocity** ($\dot{q}$): 3 linear + 3 angular = 6 entries for the root
  - `qvel[0:3]` = (vx, vy, vz) linear velocity
  - `qvel[3:6]` = (wx, wy, wz) angular velocity

**Why qpos has 7 but qvel has 6?** Because orientations live in SO(3) — a quaternion has 4 components but only 3 degrees of freedom (the 4th is constrained: $w^2+x^2+y^2+z^2 = 1$). MuJoCo uses 4 numbers for position (quaternion) but only 3 for velocity (angular velocity vector).

So: nq = 7 (root) + 29 (joints) = **36**, but nv = 6 (root) + 29 (joints) = **35**.

### Joint ordering (29 actuators)

The actuator indices in `data.ctrl[0:29]` map to:

```
Index  Joint Name                  Body Part
──────────────────────────────────────────────
 0     left_hip_pitch_joint        Left leg
 1     left_hip_roll_joint
 2     left_hip_yaw_joint
 3     left_knee_joint
 4     left_ankle_pitch_joint
 5     left_ankle_roll_joint
 6     right_hip_pitch_joint       Right leg
 7     right_hip_roll_joint
 8     right_hip_yaw_joint
 9     right_knee_joint
10     right_ankle_pitch_joint
11     right_ankle_roll_joint
12     waist_yaw_joint             Waist
13     waist_roll_joint
14     waist_pitch_joint
15     left_shoulder_pitch_joint   Left arm
16     left_shoulder_roll_joint
17     left_shoulder_yaw_joint
18     left_elbow_joint
19     left_wrist_roll_joint
20     left_wrist_pitch_joint
21     left_wrist_yaw_joint
22     right_shoulder_pitch_joint  Right arm
23     right_shoulder_roll_joint
24     right_shoulder_yaw_joint
25     right_elbow_joint
26     right_wrist_roll_joint
27     right_wrist_pitch_joint
28     right_wrist_yaw_joint
```

**Critical insight for DOF indexing:**

- `data.ctrl` is indexed 0-28 (29 actuators, no root)
- `data.qpos` is indexed 0-35 (7 root + 29 joints). Joint $i$ in ctrl → `qpos[7+i]`
- `data.qvel` is indexed 0-34 (6 root + 29 joints). Joint $i$ in ctrl → `qvel[6+i]`
- `data.qfrc_bias` is indexed 0-34 (same as qvel). Joint $i$ → `qfrc_bias[6+i]`

This is why the code uses `_ACTUATED_DOF_START = 6` and `_ACTUATED_DOF_END = 35`.

### Actuator type: torque motors

The G1 XML uses `<motor>` actuators — these are **torque actuators**:

```xml
<actuator>
  <motor name="left_hip_pitch" joint="left_hip_pitch_joint"
         ctrlrange="-88 88" />
</actuator>
```

When you set `data.ctrl[0] = 50.0`, you're saying: *"Apply 50 Newton-metres of torque to the left hip pitch joint."*

This is in contrast to **position actuators** (which would internally compute "what torque do I need to reach angle $\theta$?"). With torque actuators, **we have to do that computation ourselves** — that's the PD controller.

---

## 4. MJCF XML — How the Scene is Defined

### File structure

```
sim/g1_with_camera.xml    ← scene file (table, cube, lights, floor)
  └── includes sim/models/g1_29dof.xml  ← robot model (links, joints, actuators, camera)
        └── references meshes in repos/unitree_mujoco/unitree_robots/g1/meshes/
```

### The scene file (`g1_with_camera.xml`)

This defines **everything the robot interacts with**:

```xml
<mujoco model="g1_with_camera">
  <include file="models/g1_29dof.xml"/>  <!-- the robot -->

  <worldbody>
    <light .../>
    <geom name="floor" type="plane" .../>   <!-- ground plane -->
    <camera name="scene_camera" .../>        <!-- overhead debug camera -->

    <body name="table" pos="0.6 0 0">       <!-- table 60cm in front -->
      <geom type="box" size="0.3 0.3 0.4" pos="0 0 0.4" .../>
    </body>

    <body name="red_cube" pos="0.6 0.0 0.825">  <!-- on the table -->
      <freejoint name="cube_joint"/>              <!-- can be pushed -->
      <geom type="box" size="0.025 0.025 0.025"  <!-- 5cm cube -->
            mass="0.05" .../>
    </body>
  </worldbody>
</mujoco>
```

**Key geometry details:**

| Object | Position | Size | Notes |
|--------|----------|------|-------|
| Robot pelvis | (0, 0, 0.793) | — | Standing height |
| Table | (0.6, 0, 0) | 60×60×80 cm | Centre is at x=0.6, top surface at z=0.8 |
| Red cube | (0.6, 0, 0.825) | 5×5×5 cm | Centre at z=0.825 = table_top(0.8) + half_cube(0.025) |

**Why `freejoint` on the cube?** Without it, the cube would be rigidly attached to the world and couldn't be pushed. The freejoint lets it translate and rotate freely — so when the robot's hand touches it, it slides and tumbles realistically.

### The robot model (`g1_29dof.xml`)

The 531-line file defines the robot's **kinematic tree** as nested `<body>` elements:

```
pelvis (root, freejoint)
├── left_hip_pitch_link
│   └── left_hip_roll_link
│       └── left_hip_yaw_link
│           └── left_knee_link
│               └── left_ankle_pitch_link
│                   └── left_ankle_roll_link
├── right_hip_pitch_link (mirror of left)
│   └── ...
└── waist_yaw_link
    └── waist_roll_link
        └── torso_link  ← ego_camera is HERE
            ├── left_shoulder_pitch_link
            │   └── left_shoulder_roll_link
            │       └── left_shoulder_yaw_link
            │           └── left_elbow_link
            │               └── left_wrist_roll_link
            │                   └── left_wrist_pitch_link
            │                       └── left_wrist_yaw_link
            └── right_shoulder_pitch_link (mirror of left)
                └── ...
```

Each body has:
- `<inertial>` — mass, centre of mass, moment of inertia (crucial for physics accuracy)
- `<joint>` — how it connects to its parent (axis, range, damping)
- `<geom>` — collision shapes and visual meshes (STL files)

**MuJoCo defaults system:** The `<default>` blocks define shared properties. For example, all leg joints share `damping="0.05"` and `armature="0.01"`. This avoids repeating the same attributes 12 times.

```xml
<default class="leg_motor">
  <joint damping="0.05" armature="0.01" frictionloss="0.2"/>
</default>
```

- **damping**: viscous friction (resists velocity) — `τ_damp = -d·q̇`
- **armature**: reflected motor inertia (stabilises the simulation)
- **frictionloss**: Coulomb friction (constant resistance to motion)

---

## 5. Torque Control & the PD Controller

### The problem

We want to command: *"move right shoulder pitch to 0.5 radians"*  
But `data.ctrl[22]` accepts **torque in Newton-metres**, not angle.

We need a controller that converts position error into torque.

### PD Controller

The **Proportional-Derivative** controller:

$$\tau = K_p (q^* - q) - K_d \dot{q}$$

Where:
- $q^*$ = desired joint angle (what arm_teleop or VLA model outputs)
- $q$ = current joint angle (from `data.actuator_length`)
- $\dot{q}$ = current joint velocity (from `data.actuator_velocity`)
- $K_p$ = proportional gain (spring stiffness — "how hard to pull toward target")
- $K_d$ = derivative gain (damping — "how much to resist fast motion")

**Intuition:**
- **P term** ($K_p(q^* - q)$): Acts like a spring. The further from the target, the more torque.
- **D term** ($-K_d \dot{q}$): Acts like a shock absorber. Prevents overshooting and oscillation.

### Why not just a P controller?

With only $\tau = K_p(q^* - q)$, the joint would oscillate around the target forever — like an undamped spring. The D term damps oscillations.

**Analogy:** A car's suspension has both a spring (P) and a shock absorber (D). Without the shock absorber, you'd bounce forever.

### Gain tuning in the code

```python
_KP = np.array([
    44, 44, 44, 70, 25, 25,   # left leg
    44, 44, 44, 70, 25, 25,   # right leg
    44, 25, 25,               # waist
    12, 12, 12, 12,  3,  3,  3,  # left arm
    12, 12, 12, 12,  3,  3,  3,  # right arm
])
```

**Why different gains per joint?**
- **Legs** (44-70): Heavy limbs that need to support body weight. The knee (70) carries the most load.
- **Arms** (12): Lighter limbs, don't need as much force.
- **Wrists** (3): Very light end-effectors. High gains would cause jittering because of low inertia.

**Rule of thumb used:** $K_p \approx 0.5 \times \text{ctrlrange\_max}$, $K_d \approx 0.1 \times K_p$.

### Torque clipping

```python
tau = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
```

Real motors have maximum torque. The XML defines this:
```xml
<motor name="left_hip_pitch" joint="left_hip_pitch_joint" ctrlrange="-88 88"/>
```

The PD controller might compute τ = 200 Nm, but the motor can only deliver 88 Nm. Clipping ensures physical realism.

---

## 6. Gravity Compensation — The Math

### The problem without gravity comp

Without any control, gravity pulls the robot down. Every joint experiences a gravitational torque — the weight of everything "below" it in the kinematic chain.

Even with the PD controller commanding $q^* = 0$ (stand still), the gravity torque pushes joints away from zero. The PD fights back but there's always some **steady-state error**:

At equilibrium: $K_p(q^* - q_{ss}) = \tau_{gravity}$

$$q_{ss} = q^* - \frac{\tau_{gravity}}{K_p}$$

With finite $K_p$, the robot sags. With infinite $K_p$, the simulation becomes unstable.

### The solution: feedforward gravity compensation

We add the gravity torque directly to our control:

$$\tau = K_p(q^* - q) - K_d \dot{q} + g(q)$$

Where $g(q)$ is the gravity torque at configuration $q$. Now at equilibrium ($q = q^*$, $\dot{q} = 0$):

$$\tau = 0 - 0 + g(q^*) = g(q^*)$$

The PD error is zero, and the gravity term exactly cancels gravity. **Perfect pose holding.**

### Where does $g(q)$ come from?

MuJoCo computes it for us: `data.qfrc_bias`

More precisely, `qfrc_bias` contains both gravity AND Coriolis/centrifugal forces:

$$\text{qfrc\_bias} = c(q, \dot{q}) = g(q) + C(q, \dot{q})\dot{q}$$

For slow motions (our case — teleoperation is slow), the Coriolis terms are negligible and `qfrc_bias ≈ g(q)`.

### The code

```python
def _compute_pd_torques(self) -> np.ndarray:
    q = self.data.actuator_length.copy()      # current joint angles
    qd = self.data.actuator_velocity.copy()   # current joint velocities
    tau = _KP * (self._target_pos - q) - _KD * qd

    if self.gravity_comp:
        tau += self.data.qfrc_bias[6:35]      # gravity compensation

    tau = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
    return tau
```

**Why `[6:35]`?**  
`qfrc_bias` is indexed by DOF (nv=35). DOFs 0-5 are the floating base; DOFs 6-34 are the 29 actuated joints. We need the gravity forces for the actuated joints only, because those are what our 29 torque actuators can counteract.

### Mathematical summary

The full control law implemented:

$$\tau_i = K_{p,i}(q_i^* - q_i) - K_{d,i}\dot{q}_i + c_i(q, \dot{q})$$

clipped to $[\tau_{min,i}, \tau_{max,i}]$.

Where $i \in \{0, 1, ..., 28\}$ for the 29 actuated joints.

---

## 7. Fixed Base — Why and How

### The balance problem

A humanoid robot is an **inverted pendulum** — it's inherently unstable. Standing still requires active balance control:
- Measuring centre of mass vs. centre of pressure
- Computing corrective ankle/hip torques in real time
- This is a whole separate research field (ZMP control, whole-body control)

When we just command arm joints to move, the centre of mass shifts, and without balance control, the robot falls over.

### The fixed-base trick

For **manipulation tasks** (reaching, picking up objects), we don't need locomotion. So we freeze the pelvis in space:

```python
if self.fixed_base:
    self.data.qpos[:7] = self._base_qpos   # reset position + orientation
    self.data.qvel[:6] = 0.0                # zero linear + angular velocity
    mujoco.mj_forward(self.model, self.data)  # recompute kinematics
```

**What this does:**
1. After each `mj_step()`, overwrite the root position/velocity with the initial values
2. Call `mj_forward()` to propagate the frozen state through the kinematic tree
3. The robot can't translate or rotate at the pelvis — it's effectively bolted to the world

**Why `mj_forward()` after freezing?**  
`mj_step()` computes the full dynamics, including root body motion. By overwriting `qpos[:7]` and `qvel[:6]`, we've changed the state — but derived quantities (joint frames, contact forces, etc.) are now inconsistent. `mj_forward()` recomputes everything from the current `qpos/qvel` without stepping time forward.

**Why this is the standard approach:**
- ACT paper: robot arms are physically bolted to a table
- Most LeRobot demos: fixed-base robots
- Isolated arm control is the correct Phase B scope; locomotion + manipulation is Phase C+

---

## 8. ROS2 Architecture

### What is ROS2?

**ROS2** (Robot Operating System 2) is a middleware framework that lets different parts of a robot system communicate. It's NOT an operating system — it runs on top of Linux.

### Core concepts

| Concept | What it is | Our usage |
|---------|-----------|-----------|
| **Node** | A single process that does one thing | `bridge_node`, `arm_teleop_node`, `demo_recorder` |
| **Topic** | A named channel for publishing/subscribing data | `/joint_states`, `/camera/image_raw`, `/joint_commands` |
| **Publisher** | Sends messages to a topic | Bridge publishes joint states |
| **Subscriber** | Receives messages from a topic | Teleop subscribes to nothing; demo_recorder subscribes to all three |
| **Message** | A typed data structure | `JointState`, `Image` |
| **Executor** | Drives callbacks (timers, subscribers) | `MultiThreadedExecutor` in bridge |

### Publisher/Subscriber pattern

```
arm_teleop_node                    bridge_node                    demo_recorder
 (keyboard)                        (MuJoCo sim)
      │                                 │                              │
      │  publishes                      │  publishes                   │
      ├──► /joint_commands ────────────►├──► /joint_states  ──────────►│
      │   (JointState)                  │   (JointState, 100Hz)        │  subscribes to all
      │                                 ├──► /camera/image_raw ───────►│
      │                                 │   (Image, 30Hz)              │
      │                                 │                              │
      │                                 │ subscribes                   │
      │                                 ◄── /joint_commands ◄──────────┤
```

**Decoupled design:** Each node handles one responsibility. You can swap any node without affecting the others. Want a different controller? Replace `arm_teleop_node`. Want a different sim? Replace `bridge_node` with a real robot driver. The topics stay the same.

### Message types used

**`sensor_msgs/JointState`:**
```python
msg.header.stamp = # timestamp
msg.name = ["left_hip_pitch_joint", ...]  # 29 joint names
msg.position = [0.0, 0.1, ...]           # 29 joint angles (radians)
msg.velocity = [0.0, 0.0, ...]           # 29 joint velocities (rad/s)
```

**`sensor_msgs/Image`:**
```python
msg.height = 480
msg.width = 640
msg.encoding = "rgb8"           # 3 bytes per pixel
msg.step = 640 * 3              # bytes per row
msg.data = frame.tobytes()      # raw pixel bytes (480 * 640 * 3)
msg.header.frame_id = "ego_camera"
```

### QoS (Quality of Service)

The numbers in `create_publisher(JointState, "/joint_states", 10)`:
- `10` = queue depth. If the subscriber can't keep up, up to 10 messages are buffered.
- For the camera: queue depth 1 — we always want the latest frame, not a backlog.

---

## 9. The Bridge Pattern

### Why a bridge?

MuJoCo doesn't know about ROS2. ROS2 doesn't know about MuJoCo. The **bridge node** is the adapter between them:

```
┌──────────────┐         ┌──────────────────┐         ┌─────────────┐
│  MujocoSim   │ ◄─────► │ MujocoBridgeNode │ ◄─────► │   ROS2      │
│  (physics)   │  Python  │ (adapter)        │  topics │   network   │
└──────────────┘  calls   └──────────────────┘         └─────────────┘
```

### What bridge_node does

1. **Reads** from MujocoSim → **publishes** to ROS2:
   - Every 10ms (100 Hz): reads joint state, publishes `/joint_states`
   - Every 33ms (30 Hz): reads camera frame, publishes `/camera/image_raw`

2. **Subscribes** from ROS2 → **writes** to MujocoSim:
   - On any `/joint_commands` message: calls `sim.set_joint_command()`

### Configuration via ROS2 parameters

```bash
ros2 run vla_mujoco_bridge bridge_node \
  --ros-args -p fixed_base:=true -p gravity_comp:=true
```

Parameters are read at startup via a temporary node (because `MujocoSim` needs them in its constructor, before the bridge node exists):

```python
param_node = rclpy.create_node("_bridge_params")
param_node.declare_parameter("gravity_comp", True)
# ... read all params ...
param_node.destroy_node()

sim = MujocoSim(gravity_comp=gravity_comp, ...)  # now we can construct
node = MujocoBridgeNode(sim)
```

---

## 10. Threading Model

### Why threads?

Two things need to happen simultaneously:
1. **Physics simulation** — must run at 500 Hz (every 2ms), blocking loop
2. **ROS2 callbacks** — handle timers (publish at 100Hz/30Hz) and subscribers

You can't do both in one thread — the physics loop would block ROS2 from firing callbacks.

### The threading design

```
Main Thread:                    Background Thread(s):
┌─────────────────────┐        ┌─────────────────────┐
│ sim.run_physics_loop│        │ executor.spin()     │
│   while running:    │        │   fires timers:     │
│     compute PD      │        │     _pub_joints()   │
│     mj_step()       │        │     _pub_camera()   │
│     freeze base     │        │   handles subs:     │
│     render camera   │        │     _on_joint_cmd() │
│     sync viewer     │        │                     │
│     sleep for dt    │        │                     │
└─────────────────────┘        └─────────────────────┘
         │                              │
         └────────── shared state ──────┘
              protected by self.lock
```

### Thread safety

Any data read by both threads must be protected:

```python
# Physics loop writes:
with self.lock:
    np.copyto(self.latest_joint_pos, self.data.actuator_length)

# ROS2 callback reads:
def get_joint_state(self):
    with self.lock:
        return self.latest_joint_pos.copy()  # .copy() = snapshot, safe
```

The `.copy()` is important — without it, the caller gets a **reference** to `self.latest_joint_pos`, which the physics thread could overwrite mid-use.

### Why `MultiThreadedExecutor`?

ROS2's default executor is single-threaded — callbacks run one at a time. With `MultiThreadedExecutor(num_threads=2)`, two callbacks can run simultaneously. Combined with `ReentrantCallbackGroup`, this means the 100Hz joint publisher and the subscriber callback don't block each other.

---

## 11. Camera & Image Pipeline

### How MuJoCo renders

MuJoCo can render to an **offscreen buffer** using OpenGL/EGL (no display needed):

```python
os.environ["MUJOCO_GL"] = "egl"  # headless rendering

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="ego_camera")  # update positions
frame = renderer.render()  # returns RGB numpy array (480, 640, 3)
```

**EGL** is an API for creating OpenGL contexts without a window. This lets us render on a GPU without a monitor connected — crucial for servers and headless testing.

### Camera definition in XML

```xml
<!-- In torso_link, the camera moves with the robot's chest -->
<camera name="ego_camera" pos="0.23 0.0 0.06" 
        xyaxes="0 -1 0 0.45 0 1" fovy="80"/>
```

- `pos="0.23 0.0 0.06"` — 23cm forward, 6cm up from torso_link origin
- `xyaxes` — defines camera orientation (which way is right, which way is up)
- `fovy="80"` — vertical field of view in degrees (wide, like a head-mounted camera)

### Why we dropped cv_bridge

ROS2 Jazzy's `cv_bridge` was compiled against NumPy 1.x. Our system has NumPy 2.2.6 (installed by lerobot). The binary incompatibility causes a crash at import.

**Solution:** Manual conversion (it's actually simpler):

```python
# Publish: numpy → Image message
msg = Image()
msg.height, msg.width = frame.shape[:2]
msg.encoding = "rgb8"
msg.step = frame.shape[1] * 3  # bytes per row = width × 3 channels
msg.data = frame.tobytes()     # flatten to bytes

# Subscribe: Image message → numpy
frame = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
    msg.height, msg.width, 3
)
```

No external dependencies. No ABI issues. Works everywhere.

---

## 12. Teleoperation Design

### arm_teleop_node.py

**Purpose:** Let a human control the robot's arms with the keyboard to demonstrate tasks.

**Architecture:**

```
pynput.Listener → on_press callback → apply_key() → update array → publish /joint_commands
```

**Why only arm joints?**

The 29-joint array is published with:
- Indices 0-14 (legs + waist) = `0.0` → gravity comp holds them still
- Indices 15-28 (arms) = keyboard-controlled

**Joint limits:**

```python
_ARM_LIMITS = {
    22: (-2.8, 2.8),   # R shoulder_pitch — wide range
    25: (-0.1, 2.8),   # R elbow — can't hyperextend (min -0.1, not -2.8)
    ...
}
```

These are **conservative** — inside the XML ctrlrange. Going to the full mechanical limit risks unrealistic contact with the robot's own body.

**Why DELTA = 0.04 radians?**

0.04 rad ≈ 2.3°. Each keypress moves a joint by ~2°. This gives:
- Fine enough control for precise reaching
- Fast enough that you don't need 100 keypresses to reach the cube
- Shoulder full range (~5.6 rad) takes ~140 keypresses

### pynput vs. stdin

`pynput` captures **individual key presses** without needing Enter. This is essential — you want immediate response when pressing W, not "type W then press Enter".

---

## 13. Demo Recording & HDF5 Format

### What we're recording

At 30 Hz, each timestep captures:

| Dataset key | Shape | Type | Source |
|-------------|-------|------|--------|
| `obs/joint_positions` | (T, 29) | float32 | `/joint_states` msg.position |
| `obs/joint_velocities` | (T, 29) | float32 | `/joint_states` msg.velocity |
| `obs/camera_frames` | (T, 480, 640, 3) | uint8 | `/camera/image_raw` |
| `action` | (T, 29) | float32 | `/joint_commands` msg.position |

**Why separate observation and action?**

This is the standard **imitation learning** data format:
- **Observation** = what the robot **sees** at time $t$ (proprioceptive state + camera image)
- **Action** = what the robot **did** at time $t$ (the command the operator gave)

The model will learn: *given this observation, what action should I take?*

### HDF5 format

**HDF5** (Hierarchical Data Format 5) is a binary container format. Think of it as a filesystem inside a file:

```
episode_0000.hdf5
├── obs/
│   ├── joint_positions      (150, 29) float32
│   ├── joint_velocities     (150, 29) float32
│   └── camera_frames        (150, 480, 640, 3) uint8
├── action                   (150, 29) float32
└── attrs:
    ├── episode_id = 0
    ├── task_description = "reach red cube"
    ├── fps = 30
    ├── timestamp = "2026-03-02T14:30:00"
    └── num_frames = 150
```

**Why HDF5?**
- Supports gzip compression (camera frames compress well — ~10x smaller)
- Random access (you can read frame 50 without loading frames 0-49)
- Self-describing (dataset shapes/types are stored in the file)
- Standard in robotics (RoboTurk, RoboCasa, DROID datasets all use HDF5)

### Recording architecture

```
/joint_states    ──► _on_joints()    ──► _latest_joints (updated async)
/joint_commands  ──► _on_cmd()       ──► _latest_action (updated async)
/camera/image_raw──► _on_image()     ──► _latest_frame  (updated async)
                                              │
                 _record_tick() (30Hz timer) ──┘
                 reads latest values, appends to buffers
```

The subscribers update the latest values at arbitrary rates. The timer samples them at exactly 30 Hz. This ensures uniform timestep in the recorded data even if messages arrive irregularly.

---

## 14. Imitation Learning — ACT Model

### What is imitation learning?

Instead of defining a reward function and letting the robot learn by trial and error (reinforcement learning), we **show** the robot what to do and let it learn from our demonstrations.

$$\pi_\theta(a | o) \approx \pi_{\text{expert}}(a | o)$$

Learn a policy $\pi_\theta$ (parameterised by neural network weights $\theta$) that maps observations $o$ to actions $a$, by minimising the difference from expert demonstrations.

The simplest approach: **behavioural cloning** — supervised learning on (observation, action) pairs:

$$\mathcal{L}(\theta) = \mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \| \pi_\theta(o) - a \|^2 \right]$$

Minimise the L2 distance between predicted actions and expert actions across the dataset $\mathcal{D}$.

### Why not plain behavioural cloning?

Two problems:

1. **Compounding error:** Small prediction errors accumulate over time, pushing the robot into states never seen in demos — then it doesn't know what to do.

2. **Multi-modal actions:** For some observations, there might be multiple valid actions (reach left OR right around an obstacle). L2 loss averages them, producing an action that's in between (reaching straight into the obstacle).

### ACT: Action Chunking with Transformers

**Paper:** "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)

ACT addresses both problems:

**1. Action Chunking** — instead of predicting one action at a time, predict $H$ future actions at once:

$$\pi_\theta(o_t) \rightarrow [a_t, a_{t+1}, ..., a_{t+H-1}]$$

In our config: `chunk_size = 100` → predict 100 future timesteps (3.3 seconds at 30 Hz).

**Why this helps:** If a single prediction is slightly off, the next 99 are still guided by the original chunk. You don't re-predict from the corrupted state — you follow the plan. This dramatically reduces compounding error.

**2. Conditional VAE (CVAE)** — handles multi-modality:

```
            Encoder (training only)
            ┌───────────────┐
Expert      │ transforms    │ → z (latent style)
action ────►│ to latent     │         │
sequence    │ space         │         │
            └───────────────┘         │
                                      ▼
Observation ──────────────► Decoder ──────► Predicted action chunk
(visual +                   (transformer)
proprioceptive)
```

- During **training**: the encoder sees the expert's action sequence and produces a latent $z$ that captures the "style" of the demonstration. The decoder reconstructs the actions from observation + $z$.
- During **inference**: $z$ is sampled from the prior $\mathcal{N}(0, I)$ — the decoder generates a plausible action sequence from just the observation.

The VAE loss:

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|o,a) \| p(z))$$

Where:
- $\mathcal{L}_{recon}$ = reconstruction loss (L2 between predicted and expert actions)
- $D_{KL}$ = KL divergence (regularises the latent space to be close to a standard normal)
- $\beta$ = weighting factor (typically small, ~0.01)

### ACT Architecture Details

```
Camera Image (480×640×3)
       │
       ▼
ResNet-18 backbone (pretrained on ImageNet)
       │
       ▼
Visual tokens (spatial features → flattened → projected)
       │
       ├── + Joint state tokens (29 positions + 29 velocities = 58)
       │
       ├── + [CLS] token (start of sequence)
       │
       ├── + Latent z token (from CVAE encoder during training)
       │
       ▼
Transformer Decoder (causal attention)
       │ outputs H action embeddings
       ▼
Linear projection → (H, 29) action chunk
```

**ResNet-18:** A convolutional neural network that extracts visual features from raw pixels. Pretrained on ImageNet = it already knows about edges, textures, objects. We fine-tune it on our specific camera setup.

### Hyperparameters for RTX 4050 (6GB VRAM)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `chunk_size` | 100 | 3.3s horizon at 30Hz — covers full reach motion |
| `batch_size` | 8 | Maximum that fits in 6GB VRAM |
| `backbone` | ResNet-18 | Smallest standard backbone (~11M params) |
| `lr` | 1e-5 | Conservative; ACT is sensitive to learning rate |
| `training_steps` | 100k | ~2-4 hours; typically converges for simple tasks |

---

## 15. LeRobot Framework

### What is LeRobot?

**LeRobot** (by HuggingFace) is an open-source framework for robot learning. It provides:
- Dataset format (v2: parquet + mp4)
- Training scripts for ACT, Diffusion Policy, TDMPC, etc.
- Evaluation utilities
- Hub integration (share datasets/models on HuggingFace)

### LeRobot v2 Dataset Format

```
data/lerobot_dataset/
├── meta/
│   ├── info.json          ← dataset metadata (shapes, dtypes, robot_type)
│   └── episodes.jsonl     ← per-episode metadata
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  ← tabular data (states, actions)
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.ego_camera/
            ├── episode_000000.mp4  ← video frames
            ├── episode_000001.mp4
            └── ...
```

**Why parquet?** Apache Parquet is a columnar format optimised for large tabular data. A single file can contain thousands of rows (one per timestep) with columns for each state dimension. Much faster to load than CSV.

**Why mp4 instead of raw images?** A 5-second episode at 30Hz × 480×640×3 = **138 MB** raw. As mp4: **~1 MB**. The training dataloader decodes frames on-the-fly.

### info.json structure

```json
{
  "codebase_version": "v2.0",
  "robot_type": "unitree_g1",
  "fps": 30,
  "total_episodes": 25,
  "total_frames": 3750,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [58],         // 29 positions + 29 velocities
      "names": ["j0", "j1", ...]
    },
    "action": {
      "dtype": "float32",
      "shape": [29],
      "names": ["j0", "j1", ...]
    },
    "observation.images.ego_camera": {
      "dtype": "video",
      "shape": [3, 480, 640],
      "info": {"video.fps": 30, "video.codec": "mp4v"}
    }
  }
}
```

---

## 16. The Full Data Pipeline

### End-to-end flow

```
Stage 1: TELEOPERATION
────────────────────────
Human operates keyboard
       │
       ▼
arm_teleop_node publishes /joint_commands
       │
       ▼
bridge_node: MuJoCo steps, publishes /joint_states + /camera/image_raw
       │
       ▼
demo_recorder: subscribes all three topics, samples at 30Hz
       │
       ▼
Saves to data/demos/episode_NNNN.hdf5


Stage 2: CONVERSION
────────────────────────
python3 scripts/convert_to_lerobot.py
       │
       ├── Reads each HDF5 episode
       │     pos(T,29), vel(T,29), act(T,29), imgs(T,480,640,3)
       │
       ├── Writes per-episode parquet
       │     columns: observation.state.0, ..., action.0, ..., timestamp, etc.
       │
       ├── Encodes camera_frames → mp4 video
       │     cv2.VideoWriter(mp4v codec)
       │
       └── Writes meta/info.json + meta/episodes.jsonl


Stage 3: TRAINING
────────────────────────
python3 scripts/train_act.py
       │
       ├── Invokes: python -m lerobot.scripts.train
       │     with ACT policy config + hyperparameters
       │
       ├── DataLoader:
       │     reads parquet (state, action)
       │     decodes mp4 frames on the fly
       │     creates (obs, action_chunk) training pairs
       │
       ├── Training loop:
       │     forward pass: obs → ACT model → predicted actions
       │     loss = L2(predicted, expert) + β·KL_divergence
       │     backward pass → optimizer step
       │
       └── Saves checkpoints to data/act_training/


Stage 4: INFERENCE (Phase B6 — not yet built)
────────────────────────
vla_node.py (to be created)
       │
       ├── Subscribes: /camera/image_raw, /joint_states
       │
       ├── Runs ACT model at 30Hz:
       │     observation = (joint_pos, joint_vel, camera_frame)
       │     action_chunk = model(observation)   # (100, 29)
       │     execute first action, shift window
       │
       └── Publishes: /joint_commands
```

### Temporal chunking during inference

At time $t$, the model predicts actions $[a_t, a_{t+1}, ..., a_{t+99}]$.

**Naive approach:** Execute all 100 actions, then re-predict.
- Problem: no feedback for 3.3 seconds.

**Temporal ensemble (what ACT does):**
- At $t$: predict $[a_t^{(0)}, a_{t+1}^{(0)}, ..., a_{t+99}^{(0)}]$
- At $t+1$: predict $[a_{t+1}^{(1)}, a_{t+2}^{(1)}, ..., a_{t+100}^{(1)}]$
- The executed action at $t+1$ is a weighted average: $\bar{a}_{t+1} = w_0 \cdot a_{t+1}^{(0)} + w_1 \cdot a_{t+1}^{(1)}$
- Typically exponential weighting: more recent predictions get more weight

This smooths transitions and incorporates the latest visual feedback while maintaining temporal coherence.

---

## 17. File-by-File Reference

### `mujoco_sim.py` — The Physics Engine Wrapper

| Section | Lines | What it does |
|---------|-------|-------------|
| Constants | 1-60 | Model path, camera name, render size, DOF indices, PD gains |
| Constructor | 62-130 | Loads model, stores initial state, validates camera, creates locks |
| Thread-safe API | 132-155 | `get_joint_state()`, `get_latest_frame()`, `set_joint_command()`, `stop()` |
| PD Controller | 157-168 | Computes τ = Kp(q*-q) - Kd·q̇ + g(q), clips to ctrlrange |
| Physics loop | 170-246 | Main loop: PD → step → freeze base → render → viewer sync → sleep |

**Key design decisions:**
- All MuJoCo access in the physics thread; ROS2 reads snapshots via locks
- Renderer created in physics thread (EGL context must bind to one thread)
- Camera renders at 30Hz even though physics runs at 500Hz (separate clocks)

### `bridge_node.py` — The ROS2 Adapter

| Section | What it does |
|---------|-------------|
| `MujocoBridgeNode.__init__` | Creates pubs, subs, timers |
| `_pub_joints()` | Reads sim state → publishes JointState at 100Hz |
| `_pub_camera()` | Reads sim frame → publishes Image at 30Hz |
| `_on_joint_cmd()` | Receives JointState → writes to sim target |
| `main()` | Reads params → creates sim → creates node → spins executor + physics loop |

### `arm_teleop_node.py` — Keyboard → Joint Commands

| Section | What it does |
|---------|-------------|
| `_KEY_MAP` | Maps keyboard chars to (joint_index, sign) pairs |
| `_ARM_LIMITS` | Per-joint angle limits (conservative) |
| `apply_key()` | Increments joint angle by DELTA, clips to limits, publishes |
| `main()` | pynput listener → ROS2 spin loop |

### `demo_recorder.py` — HDF5 Episode Writer

| Section | What it does |
|---------|-------------|
| Subscribers | Updates `_latest_*` arrays async from ROS2 topics |
| `_record_tick()` | 30Hz timer: samples latest data into buffers |
| `start_episode()` | Clears buffers, sets recording flag |
| `stop_episode()` | Saves or discards, auto-increments episode ID |
| `_save_episode()` | Stacks buffers → writes HDF5 with gzip |

### `convert_to_lerobot.py` — HDF5 → LeRobot Format

| Section | What it does |
|---------|-------------|
| `convert_episode()` | Reads HDF5 → writes parquet (tabular) + mp4 (video) |
| `build_info()` | Generates info.json metadata |
| `main()` | Iterates episodes, builds full dataset directory |

### `train_act.py` — ACT Training Launcher

| Section | What it does |
|---------|-------------|
| `ACT_OVERRIDES` | Hydra config overrides tuned for RTX 4050 |
| `main()` | Validates dataset → builds command → runs `lerobot.scripts.train` |

---

## Key Equations Summary

### PD Controller with Gravity Compensation

$$\tau_i = K_{p,i}(q_i^* - q_i) - K_{d,i}\dot{q}_i + c_i(q, \dot{q})$$

### MuJoCo Equations of Motion

$$M(q)\ddot{q} + c(q, \dot{q}) = \tau + J^T f$$

### Behavioural Cloning Loss

$$\mathcal{L}(\theta) = \mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \| \pi_\theta(o) - a \|^2 \right]$$

### CVAE Loss (ACT)

$$\mathcal{L} = \underbrace{\sum_{h=0}^{H-1} \| \hat{a}_{t+h} - a_{t+h} \|^2}_{\text{reconstruction}} + \beta \cdot \underbrace{D_{KL}(q(z|o,a) \| \mathcal{N}(0,I))}_{\text{regularization}}$$

### Quaternion Constraint (freejoint orientation)

$$w^2 + x^2 + y^2 + z^2 = 1$$

This is why `nq = 36` but `nv = 35` — quaternion has 4 components, 3 DOF.

---

## Practice Questions

Test your understanding:

1. **Why can't we just set `data.ctrl[i] = desired_angle`?** What would happen?

2. **If gravity_comp is ON but fixed_base is OFF, what happens to the robot?**

3. **What would happen if we used `qfrc_bias[0:29]` instead of `qfrc_bias[6:35]` for gravity comp?**

4. **Why does the demo recorder subscribe to `/joint_commands` (action) separately from `/joint_states` (observation)?**

5. **If the physics runs at 500Hz and the camera at 30Hz, how many physics steps happen between camera frames?**

6. **Why do we call `mj_forward()` after freezing the base, but not in a normal step?**

7. **What's the difference between `data.qpos[7:36]` and `data.actuator_length`?** (Hint: they should be the same for our robot. Why might they differ on other robots?)

8. **In the ACT model, what's the purpose of the latent variable z? What problem does it solve that plain L2 regression can't?**

---

## Answers

<details>
<summary>Click to reveal answers</summary>

1. `data.ctrl` expects **torques** (Nm), not angles (rad). Setting it to 0.5 would apply 0.5 Nm of torque — the joint would spin until hitting its limit or equilibrium with friction. It would NOT go to 0.5 radians.

2. The arms and legs would hold their pose (gravity comp prevents sagging), but the robot would still **fall over** because there's no balance controller. Gravity comp cancels joint-level gravity, but the whole-body center of mass is still an unstable inverted pendulum.

3. **Wrong DOFs!** `qfrc_bias[0:5]` are the gravity/Coriolis forces on the floating base (pelvis translation + rotation). These have no corresponding actuators and would be misaligned — joint 0 (left_hip_pitch) would get the pelvis x-force instead of its own gravity torque. The robot would behave erratically.

4. Because they represent different things:
   - `/joint_states` = what the robot **is doing** (measured state)
   - `/joint_commands` = what the operator **wants** it to do (desired target)
   
   Due to PD control dynamics, the actual state lags behind the command. The model needs both: "given current state, predict what command was given" during training, then "given current state, predict what command to give" during inference.

5. 500 / 30 ≈ **16.7 physics steps per camera frame**. The code tracks this with `render_interval` and only renders when enough time has passed.

6. `mj_step()` does `forward + integrate` as a pair — the state is consistent after it. But when we manually overwrite `qpos[:7]` and `qvel[:6]`, we've changed the state without updating dependent quantities (Jacobians, contact points, etc.). `mj_forward()` recomputes these derived quantities from the current state without advancing time.

7. For **hinge joints** (1 DOF per joint, which is all our robot has), they're identical. But for **ball joints** (3 DOF, parameterized by quaternion in qpos but 3-vector in qvel), `actuator_length` gives the actuator-space quantity while `qpos` gives the configuration-space quantity. Our robot only has 1-DOF joints, so they match.

8. The latent $z$ captures the **style** or **mode** of a demonstration. When multiple valid trajectories exist for the same observation (e.g., reach around left vs. right), L2 regression averages them, producing an invalid trajectory in between. The CVAE lets different $z$ samples produce different valid trajectories, solving the **multi-modality** problem.

</details>

---

*Last updated: 2026-03-03*
