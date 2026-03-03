# 05 — System Integration: ROS2 Task Manager & rosbridge (Phase D)

> **What this covers:** Integrating the trained VLA (ACT) models into a ROS2 node
> that accepts natural language commands, runs closed-loop inference, and reports
> results. Covers NL parsing, ROS2 node design, rosbridge WebSocket bridge,
> thread-safe execution, temporal action ensembling in real-time, and the full
> data flow from Telegram → VLA → MuJoCo → result.

---

## 1. Architecture Overview

The system has three layers, each operating at a different timescale:

```
User on Telegram
    │  "Pick up the red cube"          (~seconds, one-shot command)
    ↓
OpenClaw (Charlie) — AI Agent
    │  Parses intent, dispatches        (~100ms, NLU processing)
    ↓
RosClaw — ROS2 Bridge
    │  WebSocket → rosbridge (port 9090) → /vla/task_goal
    ↓
VLA Task Manager (ROS2 Node)           (~33ms per step, 30Hz control loop)
    │  camera frame + task label → ACT → joint actions
    ↓
MuJoCo Simulation
    │  Steps physics, renders camera    (~2ms per substep, 500Hz physics)
    ↓
Task Complete → JSON on /vla/status → rosbridge → RosClaw → Telegram
```

**Critical architectural insight:** The VLA model runs in a tight 30Hz control
loop (camera → action), while RosClaw/OpenClaw operates at the **task dispatch
level** — it sends the natural language command once and monitors completion
status. It does NOT participate in the per-frame control loop.

### 1.1 Why rosbridge?

rosbridge_server exposes all ROS2 topics/services over a standard WebSocket
protocol (JSON messages). This is the bridge between the ROS2 world and
external systems like RosClaw/OpenClaw.

```
External client (RosClaw)               ROS2 ecosystem
    │                                        │
    │    WebSocket (ws://localhost:9090)      │
    │  ──────────────────────────────────>    │
    │    {"op": "publish",                   │
    │     "topic": "/vla/task_goal",         │
    │     "msg": {"data": "pick up cube"}}   │
    │                                        │
    │  <──────────────────────────────────   │
    │    {"op": "publish",                   │
    │     "topic": "/vla/status",            │
    │     "msg": {"data": "{...json...}"}}   │
```

The alternative would be DDS (ROS2 native) — but that requires the external
client to run a ROS2 stack. WebSocket is universal: JavaScript, Python,
anything can connect.

---

## 2. NL Command Parsing

The task manager needs to map free-form natural language into a canonical
(mode, task_label) pair. Since we have a fixed set of 5 tasks, a simple
keyword-based parser is sufficient — no LLM needed.

### 2.1 Parsing Strategy

```python
BIMANUAL_KEYWORDS = ["box", "both hands", "bimanual", "two hand", "green"]
SINGLE_ARM_COMMANDS = {
    "reach": "reach the red cube",
    "grasp": "grasp the red cube",
    "pick":  "pick up the red cube",
    "place": "place the red cube on the blue plate",
}
```

**Decision logic:**
1. Check if any bimanual keyword appears → bimanual mode
2. Check exact match against TASK_LABELS → single-arm mode
3. Check short alias match ("reach", "grasp", "pick", "place")
4. Default to "pick up the red cube" (most common task)

### 2.2 Example Mappings

| Input Command | Mode | Task Label |
|---|---|---|
| "pick up the red cube" | single_arm | pick up the red cube |
| "grasp it" | single_arm | grasp the red cube |
| "lift the box" | bimanual | pick up the green box with both hands |
| "bimanual grasp" | bimanual | pick up the green box with both hands |
| "hello world" | single_arm | pick up the red cube (default) |
| "reach the red cube" | single_arm | reach the red cube (exact match) |

### 2.3 Why Not Use an LLM?

For 5 fixed tasks, keyword matching is:
- **Deterministic** — same input always gives same output
- **Zero latency** — no API call or inference needed
- **No dependencies** — no API keys, no model loading
- **Testable** — easy to write exhaustive unit tests

In a production system with hundreds of tasks, you'd want an LLM or at least
a sentence embedding model. But for our demo, simplicity wins.

---

## 3. ROS2 Node Design

### 3.1 TaskManagerNode Architecture

```python
class TaskManagerNode(Node):
    def __init__(self):
        # Load both ACT models (single-arm + bimanual) at startup
        self._sa_model = load("data/checkpoints/best.pt")
        self._bm_model = load("data/bimanual_checkpoints/best.pt")

        # ROS2 interfaces
        self.sub  = subscribe("/vla/task_goal", String)
        self.pub1 = publish("/vla/status", String)     # JSON
        self.pub2 = publish("/camera/image_raw", Image) # 6Hz during execution

        # Thread safety
        self._exec_lock = threading.Lock()
        self._executing = False
```

### 3.2 ROS2 Interface

| Direction | Topic | Type | Rate | Purpose |
|---|---|---|---|---|
| Subscribe | `/vla/task_goal` | `std_msgs/String` | on demand | NL task command |
| Publish | `/vla/status` | `std_msgs/String` | every 10 steps | JSON status |
| Publish | `/camera/image_raw` | `sensor_msgs/Image` | every 5 steps | Ego camera |

### 3.3 Why String-Based JSON (Not Custom Action)?

ROS2 action servers require custom `.action` definition files, which need a
separate C++ interfaces package (cmake build). For a Python-only project,
this adds significant build complexity.

The String+JSON approach:
- Works with standard `std_msgs` — no custom interfaces needed
- Compatible with rosbridge out of the box
- Easy to parse in JavaScript/Python on the RosClaw side
- Carries all needed information (step, progress, status, result)

**Trade-off:** No built-in goal cancellation or feedback streaming. For our
use case (tasks complete in ~8 seconds), this is acceptable.

### 3.4 Status Message Format

```json
{
  "step": 120,
  "progress_pct": 48.0,
  "status": "grasped",
  "result": null
}
```

Final message when task completes:

```json
{
  "step": 250,
  "progress_pct": 100.0,
  "status": "success",
  "result": {
    "success": true,
    "task": "pick up the red cube",
    "mode": "single_arm",
    "steps": 250,
    "grasped": true
  }
}
```

Bimanual result includes physics metrics:

```json
{
  "result": {
    "success": true,
    "task": "pick up the green box with both hands",
    "mode": "bimanual",
    "lift_cm": 8.5,
    "left_force": 12.3,
    "right_force": 11.8
  }
}
```

---

## 4. Thread-Safe Execution Model

### 4.1 The Problem

ROS2 callbacks run on the executor thread. VLA inference takes ~8 seconds
(250 steps × 33ms). If we run inference in the callback, we block all other
ROS2 activity (publishing, other subscriptions).

### 4.2 The Solution

```python
def _on_task_goal(self, msg):
    with self._exec_lock:
        if self._executing:
            publish("busy")
            return
        self._executing = True

    # Run in separate thread
    thread = Thread(target=self._execute_task, args=(msg.data,), daemon=True)
    thread.start()
```

Key design choices:
- **One task at a time** — mutex prevents concurrent execution
- **Daemon thread** — won't prevent node shutdown
- **Busy rejection** — if a task is running, subsequent commands get a "busy" response
- **ROS2 executor continues** — can still publish status/camera during execution

### 4.3 Why Not Use ROS2 Executors?

ROS2 has `MultiThreadedExecutor` and `ReentrantCallbackGroup`, but:
- The inference loop is a tight 250-step C-extension-heavy loop (MuJoCo + PyTorch)
- It needs to publish status mid-loop (not just at the start/end of a callback)
- A simple thread with a lock is more transparent and easier to debug

---

## 5. VLA Inference Loop

### 5.1 Control Flow

```
for step in range(250):
    if step % 5 == 0:                      # Re-plan every 5 steps
        image   = sim.render_camera()       # 480×640×3 RGB
        state   = sim.get_joint_state()     # pos + vel concatenated
        chunk   = model.predict(image, state, task_id)  # → (20,) actions
        accumulate(chunk, step)             # temporal ensembling

    action = get_ensembled_action(step)     # weighted average
    sim.apply_action(action)                # set joint targets
    sim.step()                              # advance physics

    if step % 10 == 0:
        publish_status(step, progress)
    if step % 5 == 0:
        publish_camera(image)
```

### 5.2 Temporal Action Ensembling

Each `predict()` call returns a **chunk** of 20 future actions. Instead of
executing the chunk sequentially and discarding, we **overlap** chunks and
average them with exponential decay:

```
Step 0:  chunk_0 = [a0, a1, a2, ..., a19]       weights: [1.0, 0.99, 0.98, ...]
Step 5:  chunk_1 = [a5, a6, a7, ..., a24]       weights: [1.0, 0.99, 0.98, ...]
Step 10: chunk_2 = [a10, a11, ...]

Action at step 7 = weighted_avg(chunk_0[7], chunk_1[2])
```

$$a_t = \frac{\sum_k w_{t-k} \cdot \hat{a}_t^{(k)}}{\sum_k w_{t-k}}, \quad w_i = e^{-0.01 \cdot i}$$

This produces **smoother, more consistent motions** than executing one chunk
at a time. If the model's prediction shifts slightly between re-plans, the
ensembling averages out the jitter.

### 5.3 Two Simulation Backends

The task manager instantiates different sims for different task types:

| Task Type | Sim Class | Physics | Grasping | State Dim |
|---|---|---|---|---|
| Single-arm | `SimWrapper` | Kinematic (`mj_forward`) | Weld constraint | 58 (29+29) |
| Bimanual | `PhysicsSim` | Full dynamics (`mj_step`) | Friction only | 28 (14+14) |

This mirrors how the models were trained. The single-arm ACT model learned
on kinematic demos; feeding it physics-sim obs would cause distribution shift.

### 5.4 Composite Task Handling (Pick & Place)

"Pick up the red cube" is actually two phases:
1. **Approach** (guided by "grasp" task embedding) → reach toward cube
2. **Lift** (guided by "pick" task embedding) → lift after grasping

The task ID switches at the moment of grasp:

```python
active_task_id = task_to_id("grasp the red cube")  # Phase 1

if hand_near_cube(dist < 4cm):
    set_weld(True)
    active_task_id = task_to_id("pick up the red cube")  # Phase 2
    reset_ensembling_buffers()  # Clear old predictions
```

Similarly, "place" uses grasp→place task switching. This hierarchical
decomposition was a key insight from Phase C — the model generalizes better
when each phase has its own task embedding.

---

## 6. Launch System

### 6.1 vla_system.launch.py

The launch file brings up the complete system with one command:

```bash
ros2 launch vla_mujoco_bridge vla_system.launch.py
```

This starts:
1. **rosbridge_server** — WebSocket on port 9090
2. **task_manager_node** — VLA inference with MUJOCO_GL=egl

### 6.2 Configurable Parameters

```bash
ros2 launch vla_mujoco_bridge vla_system.launch.py \
    device:=cpu \
    rosbridge_port:=8080 \
    single_arm_checkpoint:=/path/to/custom/model.pt
```

### 6.3 Environment Setup

The launch file injects `MUJOCO_GL=egl` via `additional_env`. This is
critical — without it, MuJoCo tries to open a display (GLFW) which fails
on headless servers.

---

## 7. Full Data Flow: End-to-End

### 7.1 Telegram → MuJoCo → Telegram

```
1. User types "pick up the red cube" in Telegram
2. OpenClaw (Charlie) receives message, identifies as robot command
3. Charlie → RosClaw plugin: dispatch("pick up the red cube")
4. RosClaw opens WebSocket to ws://robot:9090
5. RosClaw publishes: /vla/task_goal = "pick up the red cube"
6. rosbridge forwards to ROS2 topic
7. TaskManagerNode receives on /vla/task_goal
8. parse_task_command("pick up the red cube") → ("single_arm", "pick up the red cube")
9. Thread spawned for execution
10. SimWrapper created, model loaded (already in memory)
11. 250-step inference loop:
    - Every 5 steps: render ego camera → ACT → action chunk
    - Every step: apply action, step sim
    - Every 10 steps: publish JSON status
    - Every 5 steps: publish camera image
12. SUCCESS_FN evaluates: cube above initial + grasped → True
13. Final status published: {"status": "success", "result": {...}}
14. rosbridge forwards to WebSocket
15. RosClaw receives result
16. Charlie sends "Task complete! Successfully picked up the red cube." to Telegram
```

### 7.2 Timing Breakdown

| Component | Latency | Notes |
|---|---|---|
| Telegram → OpenClaw | ~200ms | Network + API |
| OpenClaw → RosClaw | ~50ms | Local IPC |
| RosClaw → rosbridge | ~10ms | WebSocket |
| NL parsing | <1ms | Keyword matching |
| Model loading | 0ms | Pre-loaded at startup |
| Sim initialization | ~100ms | MuJoCo model + renderer |
| Inference per step | ~5ms | ACT (15.6M params, CUDA) |
| Physics per step | ~2ms | mj_step × 16 substeps |
| Rendering per step | ~3ms | 480×640 EGL offscreen |
| Full episode (250 steps) | ~8.3s | At 30Hz control rate |
| Result → Telegram | ~300ms | Reverse path |
| **End-to-end** | **~9s** | Command to response |

---

## 8. Two Operational Modes

The `vla_mujoco_bridge` package provides two distinct entry points:

### 8.1 Standalone Sim Bridge (`bridge_node`)

```bash
ros2 run vla_mujoco_bridge bridge_node
```

- Opens a MuJoCo viewer window
- Publishes raw joint states and camera at fixed rates
- Accepts joint-level commands from external controllers
- Provides `/grasp` service for weld constraint
- **Use case:** Manual teleoperation, debugging, ROS2 integration testing

### 8.2 VLA Task Manager (`task_manager_node`)

```bash
ros2 run vla_mujoco_bridge task_manager_node
```

- Headless (EGL rendering, no viewer)
- Self-contained: embeds its own MuJoCo sim
- Accepts NL commands, runs ACT inference
- Publishes status and camera during execution
- **Use case:** Autonomous task execution via NL commands

These two modes are **complementary, not competing**. The bridge is for
development/debugging; the task manager is for production.

---

## 9. Camera Snapshot for Telegram

During task execution, the task manager publishes ego camera frames on
`/camera/image_raw` at ~6Hz. RosClaw can:

1. **Subscribe** to this topic for a live feed
2. **Capture a single frame** at any point via rosbridge

Example rosbridge subscription:

```json
{
  "op": "subscribe",
  "topic": "/camera/image_raw",
  "type": "sensor_msgs/Image",
  "throttle_rate": 5000
}
```

With `throttle_rate: 5000` (5 seconds), RosClaw receives one frame every
5 seconds — perfect for sending periodic Telegram updates ("Here's what the
robot sees:").

---

## 10. Key Lessons & Design Decisions

### 10.1 Self-Contained Inference

The task manager creates a new sim instance per task execution rather than
reusing a persistent sim. This ensures:
- Clean state for each task (no residual physics)
- No sensor drift or accumulated errors
- Simple error recovery (if MuJoCo crashes, just create a new sim)

**Trade-off:** ~100ms overhead per task. Negligible for 8-second tasks.

### 10.2 No Custom Action Interfaces

We chose `std_msgs/String` over custom ROS2 action definitions to avoid:
- A separate `vla_interfaces` CMake package
- Build complexity (ament_cmake + rosidl_generate_interfaces)
- Dependency on specific message formats that evolve

JSON-in-String is flexible and rosbridge-friendly.

### 10.3 Model Pre-Loading

Both ACT models are loaded at node startup, not per-task. Loading takes
~2 seconds; we don't want that latency on every command.

### 10.4 Lazy Sim Imports

PhysicsSim and SimWrapper are imported lazily because:
- They import MuJoCo and create model instances at import time
- We only need one per task type
- Lazy loading keeps startup fast and memory-efficient

---

## 11. Testing the System

### 11.1 Standalone Test (No rosbridge)

```bash
# Terminal 1: Start task manager
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
MUJOCO_GL=egl ros2 run vla_mujoco_bridge task_manager_node

# Terminal 2: Send a command
ros2 topic pub --once /vla/task_goal std_msgs/String "data: 'pick up the red cube'"

# Terminal 3: Monitor status
ros2 topic echo /vla/status
```

### 11.2 Full Stack Test (with rosbridge)

```bash
# Start everything
ros2 launch vla_mujoco_bridge vla_system.launch.py

# Test via WebSocket (Python example)
import websocket, json
ws = websocket.create_connection("ws://localhost:9090")
ws.send(json.dumps({
    "op": "publish",
    "topic": "/vla/task_goal",
    "msg": {"data": "pick up the green box with both hands"}
}))
```

### 11.3 NL Parser Test

```python
from vla_mujoco_bridge.task_manager_node import parse_task_command

assert parse_task_command("pick up the red cube") == ("single_arm", "pick up the red cube")
assert parse_task_command("lift the box") == ("bimanual", "pick up the green box with both hands")
assert parse_task_command("bimanual grasp") == ("bimanual", "pick up the green box with both hands")
assert parse_task_command("reach") == ("single_arm", "reach the red cube")
```

---

## 12. Summary

Phase D transforms the VLA models from standalone scripts into a **production-
ready ROS2 service** accessible via natural language over WebSocket. The key
components:

| Component | Purpose | File |
|---|---|---|
| NL Parser | Free text → (mode, task_label) | `task_manager_node.py` |
| Task Manager | ROS2 node, loads models, runs inference | `task_manager_node.py` |
| rosbridge | WebSocket ↔ ROS2 translation | `rosbridge_server` (system pkg) |
| Launch File | One-command system startup | `vla_system.launch.py` |

The architecture cleanly separates **task dispatch** (slow, human-scale) from
**motor control** (fast, 30Hz). This pattern — NL command → closed-loop VLA →
status feedback — is the standard approach for language-conditioned robot
policies and matches the GR00T N1 dual-system architecture (System 1: fast
diffusion actions, System 2: slow VLM reasoning).
