# Phase A — Lessons Learned

---

## L001: G1 uses torque actuators, not position actuators
**Discovery:** `g1_29dof.xml` uses `<motor>` actuators only. `data.ctrl[i]` = torque (Nm).
There are no `<position>` or `<velocity>` servo actuators.
**Rule:** Always implement a PD controller inside the sim loop to convert position targets
to torques: `τ = Kp*(q_des - q) - Kd*q_dot`. Clip to ctrlrange limits.
**Impact:** Affects `mujoco_sim.py` (PD controller needed), `teleop_node.py` (sends positions,
not torques), and eventually the VLA output layer.

## L002: Cannot augment an existing body via a second worldbody block
**Discovery:** MuJoCo's `<include>` inlines the referenced XML. A second `<worldbody>`
block with `<body name="torso_link">` causes "repeated name" error — MuJoCo doesn't merge
bodies, it tries to create a new one.
**Fix:** Create a local copy of the model file (`sim/models/g1_29dof.xml`) and add the
camera directly inside `torso_link`. Update `meshdir` to an absolute path.
**Rule:** Never try to augment bodies via include + second worldbody. Edit a local copy instead.

## L003: meshdir in included XML needs absolute path
**Discovery:** When `<include>` inlines a child XML, the `meshdir` compiler attribute
is resolved relative to the main file's directory, not the included file's directory.
Using `../../repos/...` in the child file resolves incorrectly.
**Fix:** Use an absolute `meshdir` path in the local model copy:
  `<compiler angle="radian" meshdir="/media/orka/storage/robotics/repos/.../meshes" />`

## L004: head_link is geometry, not a body
**Discovery:** `head_link` in G1 MJCF is a `<geom>` inside `torso_link` body, not a
separate `<body>` element. Camera is added directly inside the `torso_link` body element.

## L005: ros2 run uses system Python — must install packages system-wide
**Discovery:** `ros2 run` uses the Python that's in PATH (`/usr/bin/python3`), not the venv Python.
`import mujoco` fails if only installed in venv.
**Fix:** Install packages system-wide with `pip3 install mujoco opencv-python pynput`.
These land in `~/.local/lib/python3.10/site-packages/` and are always in sys.path.

## L004: MuJoCo GL context — EGL vs GLFW
**Rule:** Passive viewer (GLFW) and offscreen Renderer (EGL) use separate GL contexts on Linux.
They do NOT conflict. Both can be used from the same physics thread.
Create `mujoco.Renderer` inside `run_physics_loop()` (not `__init__`) so the EGL context
is bound to the correct thread. Segfault occurs if Renderer is created on a different thread
than it is used.

## L007: Ubuntu 24.04 uses ROS2 Jazzy, not Humble
**Discovery:** `install_ros2.sh` originally targeted Ubuntu 22.04 (Jammy) with `ros-humble-*`.
On Ubuntu 24.04 (Noble), only `ros-jazzy-*` packages exist.
**Fix:** Script now auto-detects Ubuntu version via `/etc/os-release` and sets `ROS_DISTRO`
accordingly. Path becomes `/opt/ros/jazzy/` instead of `/opt/ros/humble/`.
**Rule:** Always source `/opt/ros/${ROS_DISTRO}/setup.bash`. Check `lsb_release -a` first.

## L008: Ubuntu 24.04 pip requires --break-system-packages
**Discovery:** Ubuntu 24.04 enforces PEP 668 — `pip3 install` without flags raises
"externally managed environment" error.
**Fix:** `pip3 install --break-system-packages mujoco opencv-python numpy pynput`
Installs to `~/.local/lib/python3.12/site-packages/` which is always in sys.path.
This is safe — it does NOT affect system packages, it installs to user-local space.
**Rule:** Use `--break-system-packages` on Ubuntu 24.04 for user pip installs.

## L005: Source setup.zsh in zsh, not setup.bash
**Root cause:** `setup.bash` uses `${BASH_SOURCE[0]}` which is empty in zsh. This makes
`AMENT_CURRENT_PREFIX` resolve to the current working directory instead of `/opt/ros/humble`,
causing "no such file or directory: /path/to/cwd/setup.sh".
**Fix:** In zsh, always source `setup.zsh`:
  `source /opt/ros/humble/setup.zsh`
Added to `~/.zshrc` and `~/.zshrc` also sources the ros2_ws overlay automatically.
bash users: `setup.bash` in `.bashrc` is correct and unaffected.

## L006: venv + ROS2 import order
**Rule:** Always `source /opt/ros/humble/setup.bash` BEFORE activating the venv.
Reversing this order makes `rclpy` unavailable.
Also: set `MUJOCO_GL=egl` before any mujoco import, ideally at the very top of `main()`.
## L009: Gravity compensation uses qfrc_bias[6:35] for G1
**Discovery:** G1's freejoint (floating_base_joint) occupies DOFs 0-5. The 29 actuated
hinge joints occupy DOFs 6-34 in velocity space.
**Fix:** Add `data.qfrc_bias[6:35]` to the PD torques in `_compute_pd_torques`.
This alone makes the robot hold any pose against gravity without active balance control.
**Rule:** For fixed-base manipulation demos, gravity_comp=True + fixed_base=True is the
right starting point. No locomotion controller needed.

## L010: ament_python build fails on Ubuntu 24.04 due to setuptools >= 70
**Discovery:** setuptools >= 70 removed support for `python setup.py install` and
`python setup.py develop --uninstall` which ament_python's colcon build type invokes.
pip installs setuptools 80.x by default on Ubuntu 24.04, breaking `colcon build`.
**Fix:** Downgrade pip-installed setuptools: `pip3 install --break-system-packages setuptools==68.2.2`
The system apt package `python3-setuptools=68.1.2` would also work but pip overrides it.
**Rule:** After any LeRobot or major pip install, check `python3 -c "import setuptools; print(setuptools.__version__)"`.
If >= 70, downgrade to 68.2.2 before running `colcon build`.

## L011: Fixed-base manipulation is the correct Phase B starting point
**Discovery:** G1 falls immediately during teleoperation because torque-controlled humanoids
require active balance (ZMP / whole-body control) which is a separate research problem.
**Fix:** For VLA demo collection, freeze the pelvis kinematically (fixed_base=True) so
only arm joints are controlled. This matches the standard ACT/LeRobot pipeline where the
robot is bolted to a table or mounted on a fixed base.
**Rule:** Only combine locomotion + manipulation in Phase C+. Phase B = arm manipulation only.
## L012: cv_bridge compiled for NumPy 1.x crashes on NumPy 2.x (Ubuntu 24.04)
**Discovery:** ROS2 Jazzy's `cv_bridge` is compiled against NumPy 1.x. With NumPy 2.2.6
(pulled by lerobot), `from cv_bridge import CvBridge` raises ImportError at runtime.
**Fix:** Replace all cv_bridge calls with direct numpy↔Image conversion:
  - Publish (rgb8): `msg=Image(); msg.height,msg.width=frame.shape[:2]; msg.encoding="rgb8"; msg.step=frame.shape[1]*3; msg.data=frame.tobytes()`
  - Subscribe: `np.frombuffer(bytes(msg.data),dtype=np.uint8).reshape(msg.height,msg.width,3)`
**Rule:** Never use cv_bridge in this project. Direct numpy serialization has no ABI dependencies.

## L013: ctrlrange for motor actuators is torque limits, not position limits
**Discovery:** G1's `<motor>` actuators have `ctrlrange` = ±25Nm (shoulder) and ±5Nm (wrist).
These are TORQUE limits. The demo generator was clipping target joint POSITIONS to
`0.95 * ctrlrange` (±23.75 rad) — an absurd range that provides zero actual clamping.
**Fix:** Use `model.jnt_range[jnt_id]` for actual joint position limits (±1-3 rad).
Built `arm_pos_lo/hi` from `jnt_range` instead of `ctrlrange`.
**Rule:** For position control, always use `jnt_range`, never `ctrlrange` (which is torque limits for motor actuators).

## L014: G1 right arm reach is 0.51m from shoulder — place targets within 0.40m
**Discovery:** Cube originally at (0.6, 0.0, 0.825) was 0.66m from shoulder at (0, -0.10, 1.08).
The arm's total reach is only 0.51m. IK correctly hit kinematic singularity (σ₃ ≈ 0.003).
**Fix:** Moved table + cube to (0.3, -0.1) — well within comfortable reach.
Validated with reachability sweep: x=0.20→0.40 all converge in <20 IK iterations.
**Rule:** Keep manipulation targets within 0.40m of shoulder. Test with reachability sweep before committing.

## L015: For demo generation, kinematic IK (qpos + mj_forward) beats PD tracking
**Discovery:** PD-based tracking through `mj_step` failed: (1) KP too weak → arm barely moved,
(2) physics forces displaced the cube, (3) required extensive gain tuning per-joint.
**Fix:** Pure kinematic playback: set `data.qpos` directly + `mj_forward`. No dynamics.
Arm follows trajectory perfectly, cube stays static (or follows hand via manual weld enforcement).
**Rule:** For scripted expert demo generation, skip dynamics entirely. Use mj_forward (kinematics only).
Save mj_step for real-time control or RL.

## L016: mj_forward does NOT enforce equality constraints — enforce weld manually
**Discovery:** With pure kinematic playback, `mj_forward` computes positions from qpos but
does NOT solve constraint forces. Weld constraint between cube and hand is ignored.
During "pick" demos, hand lifted but cube stayed on the table.
**Fix:** After `mj_forward`, check `data.eq_active[weld_id]` and if active, set
`data.qpos[cube_qpos_adr:+3] = data.site_xpos[hand_site_id]` then call `mj_forward` again.
**Rule:** When using kinematic mode, manually enforce equality constraints by updating qpos.

## L017: Standalone ACT training is more reliable than LeRobot CLI wrapping
**Discovery:** LeRobot v0.4.4 uses Hydra configs and expects HuggingFace-format datasets
(parquet + mp4 video + specific metadata files). Wrapping the training CLI with overrides
is fragile and hard to debug.
**Fix:** Wrote standalone ACT training (`act_model.py` + `train_act.py`) that reads HDF5
demos directly, trains with PyTorch, and saves checkpoints with model config.
**Rule:** For small-scale research, standalone training loops beat framework wrappers.
More control, better debugging, and no format-conversion surprises.

## L018: Action chunking handles deterministic demos well without CVAE
**Discovery:** Original ACT uses CVAE (Conditional VAE) to model multimodal behavior.
Scripted expert demos have a single correct trajectory per state — no multimodality.
**Fix:** Used deterministic Transformer decoder without CVAE. chunk_size=20 (0.67s lookahead)
provides temporal coherence. Padded near episode end with last-action repeat.
**Rule:** Skip CVAE for deterministic demos. Add it when training on human (multimodal) data.

## L019: Freeze early ResNet layers for small datasets
**Discovery:** 15.6M param model with only 9000 samples risks overfitting on visual features.
ResNet18 low-level features (edges, textures) transfer well from ImageNet.
**Fix:** Freeze layers 0-6 of ResNet18, only fine-tune layer4 and the img_proj head.
Reduces trainable params from 15.6M to 12.8M. Layer4 adapts to MuJoCo rendering style.
**Rule:** For <50K training samples, freeze all but the last ResNet block.

## L020: Preload + resize images at init for training speed
**Discovery:** Lazy HDF5 image loading (per __getitem__) is too slow for iterative training.
60 episodes × 90 frames × 480×640×3 = 4.7 GB raw. Not feasible to keep in full resolution.
**Fix:** At Dataset init: load all episodes, resize images to 224×224, store as uint8 (~515 MB).
Normalize to float32 only at __getitem__ time. Training throughput: ~19s/epoch on RTX 4050.
**Rule:** For datasets <1GB after resize, preload everything. Normalize lazily.

## L021: Proximity-based auto-grasp/release simplifies VLA evaluation
**Discovery:** ACT model predicts only joint positions (29-dim), not grasp commands.
Need a mechanism to trigger grasp/release during evaluation rollouts.
**Fix:** Auto-grasp when hand < 4cm from cube. Auto-release (place task only) when
grasped > 30 steps AND cube within 6cm of target AND hand z is low.
**Rule:** Standard simplification in VLA research. Model learns trajectory; grasping is mechanical.

## L022: Place marker must have contype=0 conaffinity=0 to avoid physics effects
**Discovery:** Visual markers added to the scene can interfere with physics if they have
default collision properties. A blue plate marker could block cube placement.
**Fix:** Set `contype="0" conaffinity="0"` on purely visual geoms like place_marker.
**Rule:** Always disable collision on visual-only scene elements.

## L023: release_after_wp extends kinematic recording for multi-step tasks
**Discovery:** Pick-and-place requires both grasping and releasing within one episode.
The original `_kinematic_record` only supported `grasp_after_wp`.
**Fix:** Added `release_after_wp` parameter — deactivates weld after specified waypoint.
For place: grasp_after_wp=1 (descend to cube), release_after_wp=4 (lower to target).
**Rule:** Keep kinematic recording extensible with per-waypoint event hooks.

---

# Evaluation & Inference — Lessons Learned

---

## L024: Temporal ensembling is essential for closed-loop behavior cloning
**Discovery:** ACT model with loss=0.000009 produced 0% success on ALL tasks. Single-chunk
predictions compound errors within 30 steps — the state drifts from training distribution.
**Fix:** Temporal ensembling (from ACT paper): re-plan every 5 steps (chunk_exec=5),
exponentially-weighted average of overlapping chunks (ensemble_k=0.01). Newer predictions
get higher weight: `w = exp(-k * j)` where j is offset into the chunk.
**Rule:** Never evaluate BClosed-loop behavior cloning without temporal ensembling. Low training
loss means nothing for closed-loop rollout. Budget equal time for inference engineering as
for training.

## L025: Hierarchical task decomposition for multi-phase tasks
**Discovery:** Pick and place tasks got 0% even with temporal ensembling. The model
immediately moved the hand UP instead of approaching the cube. Root cause: task embedding
averages approach-phase and lift-phase training samples, producing net upward movement.
**Fix:** Split composite tasks into phases: Phase 1 uses "grasp" embedding (approach only),
Phase 2 switches to "pick"/"place" embedding after auto-grasp triggers. Reset ensembling
buffers at phase transition to prevent stale approach actions from bleeding in.
**Rule:** If a task has distinct phases (approach → manipulate), use sub-goal embeddings.
The simplest embedding is the one with the most homogeneous training data.

## L026: Prevent auto-grasp re-triggering after weld release
**Discovery:** Place task: after releasing the cube near the target, auto-grasp immediately
re-triggered because hand and cube were at the same position (just un-welded). The cube
was permanently "stuck" to the hand despite release logic firing correctly.
**Fix:** Added `released = False` flag. After release: set `released = True`. In auto-grasp
check: `if not grasped and not released:`.
**Rule:** Any proximity-based grasping heuristic needs a one-way latch after intentional release.

## L027: Kinematic mode has no gravity — simulate drop after weld release
**Discovery:** In kinematic mode (`mj_forward` only), after releasing a weld constraint,
the cube stays floating in mid-air at whatever z-position the hand was at. Physics forces
are not computed.
**Fix:** After release, manually set `data.qpos[cube_z_adr] = 0.825` (table surface height)
and call `mj_forward()`. This simulates gravity dropping the cube onto the table.
**Rule:** When using `mj_forward` (no `mj_step`), manually handle any physics effects
that would occur naturally in dynamics mode (gravity, collisions, sliding).

# Phase C2 — Bimanual Physics Lessons

---

## L028: Leg/waist joints drift under mj_step even with frozen pelvis
**Discovery:** Freezing only the floating base (qpos[:7], qvel[:6]) is insufficient.
Waist joints connect pelvis to shoulders — any drift tilts the torso, moving shoulders
and hands far from targets. After 20 settle frames, hands drifted 8cm up.
**Fix:** Freeze ALL non-arm actuated joints (legs ctrl[0:12] + waist ctrl[12:15]) every
substep alongside the pelvis. Only arm joints (ctrl[15:28]) run under physics.
**Rule:** For fixed-base manipulation with mj_step, lock every joint except the ones
you intend to control. qvel index = qpos_address - 1 (due to 7-qpos/6-qvel floating base).

## L029: G1 hand geoms have contype=0 — no collision by default
**Discovery:** Both `left_rubber_hand` and `right_rubber_hand` mesh geoms have
`contype="0" conaffinity="0"` in the official MJCF. They are visual only and pass
through all objects. No friction-based grasping is possible with them.
**Fix:** Added box-shaped "palm pad" collision geoms (8×6×2cm) to both wrist_yaw_link
bodies with `contype="1" conaffinity="1"` friction="1.5" condim="4".
**Rule:** Always verify contype/conaffinity before relying on contact physics. Visual
meshes often have collisions disabled for performance.

## L030: Position-only IK gives good palm orientation for lateral squeeze
**Discovery:** Feared that without orientation constraints, palms might not face the
box. Testing showed palm local Y axis is 98% aligned with world Y at squeeze config.
The kinematic chain naturally orients the hand forward when reaching laterally.
**Rule:** For symmetric bimanual lateral squeeze, position-only IK is sufficient.
Add orientation IK only if the task requires non-natural hand orientations.

## L031: Box rotates slightly during lift (known issue)
**Discovery:** The green box rotates ~5-10° around vertical axis during lift. Cause:
slight asymmetry in left/right contact forces and palm pad positioning. Not critical
for demo/training purposes but visually imperfect.
**TODO:** Fix by adding orientation constraint to IK, or by using condim=6 (full
friction cone with torsional component), or by adding a second palm pad per hand.

## L032: Bimanual single-task ACT achieves 100% with 30 demos
**Discovery:** 30 bimanual demos with minimal variation (±2cm box noise) → 100% eval
success (20/20). Low-dimensional action space (14 vs 29), single task, and consistent
demonstrations make the learning problem trivial for ACT.
**Rule:** For focused tasks with consistent demos, 30 episodes is sufficient for ACT.
Multi-task or higher variation may need 50-100+. Start small, scale if needed.

## L033: Compliance grasping works via PD targets inside the object surface
**Discovery:** Setting PD position targets inside the box surface creates steady squeeze
force without a force controller. The PD controller pushes toward the unreachable target,
the contact constraint pushes back, and equilibrium produces 13-14N bilateral force.
**Rule:** For friction grasping, command hand targets 2-3cm inside the object surface.
Force magnitude is proportional to penetration depth × Kp gain.
