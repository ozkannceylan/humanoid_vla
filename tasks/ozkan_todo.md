# Ozkan's Research & Learning TODO

Personal study list — topics to understand before/during Phase B.

---

## 🔴 Priority: Humanoid Balance & Locomotion Control

The G1 falls immediately during teleoperation because torque-controlled humanoids
require active balance. This must be understood before collecting VLA demonstration data.

### 1. Why Humanoids Fall — Core Concepts to Learn

- [ ] **Center of Mass (CoM) vs Center of Pressure (CoP)**
  — A humanoid stays upright only if CoP (where ground pushes back) stays inside the support polygon (foot contact area). When you move an arm joint, the CoM shifts and CoP moves — if it exits the polygon, the robot tips.

- [ ] **Zero Moment Point (ZMP)**
  — The point where the net moment of inertia and gravity forces is zero. Classic criterion: robot is stable iff ZMP stays inside support polygon. Read: Vukobratovic & Borovac (2004) "Zero-Moment Point — Thirty Five Years of its Life."

- [ ] **Inverted Pendulum Model**
  — Simplest model of a humanoid: a point mass on a massless stick. Walking = controlled falling. Linear Inverted Pendulum Model (LIPM) is the basis of most ZMP-based walkers.

### 2. Control Architectures to Research

- [ ] **PD + Gravity Compensation**
  — The simplest "stay standing" controller. Compute the torques needed to counteract gravity at each joint (from the Jacobian), then add PD on top for tracking. MuJoCo can compute this via `mj_gravcomp()` or you implement it manually.
  — **Start here** — makes the G1 stand without falling, even before implementing walk.

- [ ] **Whole Body Control (WBC)**
  — Hierarchical QP-based controller: stack tasks (balance > swing foot > CoM position > joint limits) and solve for joint torques. Used in almost all real humanoid demos.
  — Libraries: `pinocchio` (Python) for kinematics/dynamics, `qpsolvers` for QP.

- [ ] **Model Predictive Control (MPC) for Locomotion**
  — Predict future CoM trajectory over a horizon, optimize foot placements. More reactive than ZMP pre-planning.
  — See: MIT Cheetah, Unitree's own locomotion stack.

- [ ] **RL-based Locomotion (most relevant to this project)**
  — Train a neural network policy in simulation that learns to walk from reward signal. No hand-crafted ZMP math. This is what Humanoid-Gym, Isaac Lab locomotion tasks do.
  — **Most directly applicable** — your MSc background in RL maps here.
  — Key: train with domain randomization → policy transfers to standing/walking robustly.

### 3. Specific Resources to Study

- [ ] **Humanoid-Gym** — RL locomotion for Unitree H1, zero-shot sim2real
  `https://github.com/roboterax/humanoid-gym`
  — Study their reward function design (upright bonus, foot contact reward, velocity tracking)

- [ ] **legged_gym / Isaac Lab locomotion examples**
  — `https://github.com/leggedrobotics/legged_gym`
  — Even if not using Isaac Lab, the reward design and PPO training loop is instructive

- [ ] **MuJoCo Humanoid locomotion baseline (dm_control)**
  — `dm_control` `humanoid` task: stand, walk, run — built-in rewards and observations
  — Good starting point: `dm_control.suite.humanoid`

- [ ] **pinocchio library** — rigid body dynamics in Python
  — `https://github.com/stack-of-tasks/pinocchio`
  — Enables gravity compensation, Jacobian computation, forward/inverse kinematics
  — Can load URDF directly from unitree_ros

### 4. Practical Steps (in order)

- [ ] **Step 1:** Implement gravity compensation for G1 in `mujoco_sim.py`
  — Use `data.qfrc_bias` (MuJoCo precomputes the bias force including gravity and Coriolis)
  — Add it to the PD torques: `tau = Kp*(q_des - q) - Kd*q_dot + data.qfrc_bias[actuated_dofs]`
  — This alone should make the robot hold its pose without falling

- [ ] **Step 2:** Train a simple standing policy with RL (PPO in MuJoCo)
  — Reward: upright (pelvis Z height), joint limit penalty, energy penalty
  — Use `gymnasium` + `stable-baselines3` or `rsl_rl`
  — 1-2 hours of training on RTX 2080 Ti should produce a stable standing policy

- [ ] **Step 3:** Train a walking policy
  — Add velocity tracking reward, foot contact reward
  — Reference: Humanoid-Gym reward structure (but simplified for G1 in MuJoCo)

- [ ] **Step 4:** Use trained standing/walking policy as the "base controller"
  — VLA outputs desired arm/torso targets on top of the walking base
  — This is the standard approach: hierarchical control (locomotion + manipulation)

---

## 🟡 Background Reading (before Phase B)

- [ ] **ACT paper** — "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
  Zhao et al., 2023. Understand action chunking, CVAE architecture, training procedure.

- [ ] **Diffusion Policy** — "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  Chi et al., 2023. Alternative to ACT, also supported by LeRobot.

- [ ] **LeRobot documentation**
  `https://github.com/huggingface/lerobot` — understand dataset format (HDF5 episodes),
  training config, evaluation loop.

- [ ] **unitree_IL_lerobot README**
  `https://github.com/unitreerobotics/unitree_IL_lerobot` — G1-specific IL pipeline,
  how to record teleoperation episodes, how to train ACT.

---

## 🟢 Phase B Prerequisites Checklist

Before starting Phase B (VLA Integration):
- [ ] G1 can stand stably (gravity compensation or trained policy)
- [ ] Understand ACT architecture at a conceptual level
- [ ] LeRobot dataset format understood (HDF5 episodes: obs/action pairs)
- [ ] Can record a teleoperation episode and replay it in simulation
