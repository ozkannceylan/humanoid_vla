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
