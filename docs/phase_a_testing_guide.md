# Phase A Comprehensive Testing Guide

**Project:** Humanoid VLA
**Phase:** A - MuJoCo + ROS2 Setup
**Date:** 2026-02-26
**Hardware:** RTX 4050 (6GB VRAM), Windows 11 + WSL Ubuntu 22.04

---

## Overview

This guide provides systematic verification of all Phase A components before moving to Phase B (VLA Integration). Tests are organized in 5 levels, from standalone simulation to long-duration robustness.

**Test Pyramid:**
```
    L4: Robustness (headless mode, 1hr stability)
   L3: Performance (VRAM profiling, FPS benchmarks)
  L2: Integration (end-to-end teleoperation)
 L1: Components (ROS2 topics, PD controller)
L0: Standalone (MuJoCo without ROS2)
```

**Execution order:** L0 → L1 → L2 → L3 → L4 (bottom to top)

---

## Prerequisites

### Required installations:
- MuJoCo 3.5.0
- ROS2 Humble
- Python packages: `mujoco`, `opencv-python`, `numpy`, `pynput`, `cv_bridge`
- `nvidia-smi` available for VRAM monitoring

### Environment setup:
```bash
# Source ROS2 (adjust for bash/zsh)
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash

# Set working directory
cd ~/projects/humanoid_vla
```

---

## L0: Standalone Simulation Verification

**Goal:** Confirm MuJoCo simulation runs correctly without ROS2.

### Automated Test

```bash
# Run automated test (60 seconds with viewer)
python3 tests/test_l0_standalone.py

# Run longer test (5 minutes, headless for accurate VRAM)
python3 tests/test_l0_standalone.py --duration 300 --headless
```

**Expected output:**
```
✅ TEST PASSED

Performance Metrics:
  Model load time:        < 500 ms
  Avg camera FPS:         ≥ 28 Hz (target: 30 Hz)
  Physics realtime factor: 0.95-1.05 (target: 1.0)
  VRAM peak:              < 1500 MB (with viewer)
                          < 800 MB (headless)
  VRAM headroom:          > 4.5 GB
```

### Manual Verification (Optional)

```bash
# Run original test script
python3 sim/test_g1.py
```

**Verify:**
- ✓ Viewer window opens with G1 model visible
- ✓ Camera preview window shows egocentric view
- ✓ Console shows "actual_fps ≥ 28" every 3 seconds
- ✓ Robot falls gradually (expected physics - no balance controller)
- ✓ Press 'q' in camera window or close viewer to exit cleanly

---

## L1: ROS2 Component Verification

**Goal:** Verify each ROS2 node publishes/subscribes correctly.

### Test 1.1: Bridge Node Topic Publishing

**Terminal 1:** Run bridge
```bash
ros2 run vla_mujoco_bridge bridge_node
```

**Terminal 2:** Check joint states rate (run for 30 seconds)
```bash
ros2 topic hz /joint_states
```
**Expected:** `average rate: 95-105 Hz` (target 100Hz ±5%)

**Terminal 3:** Check camera rate (run for 30 seconds)
```bash
ros2 topic hz /camera/image_raw
```
**Expected:** `average rate: 28-32 Hz` (target 30Hz ±7%)

**Terminal 4:** Validate joint state schema
```bash
ros2 topic echo /joint_states --once
```
**Expected output:**
```yaml
header:
  stamp: ...
  frame_id: ''
name:
  - left_hip_pitch
  - left_hip_roll
  # ... (29 total)
position: [0.0, 0.0, ...]  # 29 floats, no NaN
velocity: [0.0, 0.0, ...]  # 29 floats, no NaN
effort: []
```

**Terminal 5:** Visual camera check
```bash
ros2 run rqt_image_view rqt_image_view
```
**Verify:**
- ✓ Select `/camera/image_raw` from dropdown
- ✓ Image shows G1's egocentric view (640×480)
- ✓ Image updates smoothly, not frozen
- ✓ No black frames or corruption

**Stop bridge with Ctrl+C**, verify clean shutdown.

---

### Test 1.2: Teleop Node Command Generation

**Terminal 1:** Run teleop only (no bridge yet)
```bash
ros2 run vla_mujoco_bridge teleop_node
```

**Terminal 2:** Monitor commands
```bash
ros2 topic echo /joint_commands
```

**Actions:**
1. Press 'w' (left hip pitch +)
   - **Expected:** `position[0]` increases by 0.05
2. Press 's' (left hip pitch -)
   - **Expected:** `position[0]` decreases by 0.05
3. Press 'i' (right shoulder pitch +)
   - **Expected:** `position[22]` increases by 0.05
4. Press 'r' (reset)
   - **Expected:** All 29 positions → 0.0
5. Press ESC
   - **Expected:** Node exits cleanly

**Verify:**
- ✓ Each keypress changes correct joint
- ✓ Commands publish at ~20 Hz
- ✓ Joint positions respect limits (e.g., hip pitch: -1.5 to +1.5)
- ✓ Clean exit with ESC

---

## L2: End-to-End Integration Test

**Goal:** Verify full teleoperation loop with acceptable latency.

### Setup (3 terminals)

**Terminal 1:** Bridge with viewer
```bash
ros2 run vla_mujoco_bridge bridge_node
```

**Terminal 2:** Teleop
```bash
ros2 run vla_mujoco_bridge teleop_node
```

**Terminal 3:** Camera feed
```bash
ros2 run rqt_image_view rqt_image_view
# Select /camera/image_raw
```

### Test Procedure

1. **Focus Terminal 2** (teleop window)
2. Press 'i' (right shoulder pitch+) repeatedly
3. **Watch MuJoCo viewer:** Right arm should move forward
4. **Watch rqt_image_view:** Arm motion should appear in camera feed
5. Measure perceived latency: keypress → visible motion in viewer

### Success Criteria

- ✓ Keypress → viewer motion within **100ms** (feels responsive)
- ✓ Keypress → camera feed motion within **2-3 frames** (67-100ms)
- ✓ No spinning/unstable behavior (thread safety OK)
- ✓ Camera feed never freezes (no deadlocks)
- ✓ Robot falls gradually (expected - no balance controller)
- ✓ All nodes exit cleanly with Ctrl+C

### Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| No motion in viewer | Teleop not publishing | Check `ros2 topic echo /joint_commands` |
| Camera frozen | Renderer thread blocked | Restart bridge, check VRAM |
| Latency > 200ms | CPU overload | Close other apps, check `htop` |
| Bridge crashes | Model path wrong | Check `mujoco_sim.py:25` MODEL_PATH |

---

## L3: RTX 4050 Performance Profiling

**Goal:** Establish VRAM baseline and confirm headroom for Phase B VLA models.

### Test 3.1: VRAM Profiling

**Step 1:** Monitor GPU in separate terminal
```bash
# Start continuous logging (run in background)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv --loop=1 > gpu_log.csv
```

**Step 2:** Run bridge with viewer for 5 minutes
```bash
ros2 run vla_mujoco_bridge bridge_node
# Let it run for 5 minutes, move some joints with teleop
```

**Step 3:** Analyze VRAM usage
```bash
# Peak VRAM during test
cat gpu_log.csv | grep -v timestamp | awk -F',' '{print $2}' | sort -n | tail -1

# Average VRAM
cat gpu_log.csv | grep -v timestamp | awk -F',' '{print $2}' | \
  awk '{sum+=$1; n++} END {print sum/n}'
```

**Step 4:** Repeat in headless mode

Modify `ros2_ws/src/vla_mujoco_bridge/vla_mujoco_bridge/bridge_node.py`:
```python
# Line 88: change
sim.run_physics_loop(launch_viewer=True)
# to
sim.run_physics_loop(launch_viewer=False)
```

Rebuild:
```bash
cd ros2_ws
colcon build --packages-select vla_mujoco_bridge
source install/setup.bash
```

Run bridge in headless mode for 5 minutes, repeat VRAM analysis.

**Expected Results:**

| Mode | Peak VRAM | Avg VRAM | Headroom (6GB total) |
|------|-----------|----------|---------------------|
| With viewer | < 1500 MB | < 1200 MB | ~4.5 GB |
| Headless | < 800 MB | < 600 MB | ~5.2 GB |

**Phase B Model Fit Assessment:**
- **ACT model (~500MB):** Should fit comfortably in headless mode
- **GR00T N1 INT8 (~3-4GB):** Marginal, requires headless mode + careful management

---

### Test 3.2: Physics FPS Stability

**Goal:** Confirm physics maintains 500Hz over extended run.

```bash
# Run bridge with viewer for 10 minutes
# Observe console output - no warnings about slow physics
ros2 run vla_mujoco_bridge bridge_node
```

**Monitor:**
- Physics loop should not print slowdown warnings
- Real-time factor should stay ~1.0
- GPU utilization < 30% (not GPU-bottlenecked)

**Success Criteria:**
- ✓ No "physics running slow" messages
- ✓ Simulation time ≈ wall clock time (±1%)
- ✓ GPU utilization steady (not spiking to 100%)

---

## L4: Robustness Verification

**Goal:** Confirm system stability and error handling.

### Test 4.1: Headless Mode (Critical for Phase B)

**Why:** VLA training will run headless to save VRAM. Must verify camera rendering works without viewer.

**Step 1:** Modify bridge_node.py for headless
```python
# Line 88 in ros2_ws/src/vla_mujoco_bridge/vla_mujoco_bridge/bridge_node.py
sim.run_physics_loop(launch_viewer=False)
```

**Step 2:** Rebuild
```bash
cd ros2_ws && colcon build --packages-select vla_mujoco_bridge
source install/setup.bash
```

**Step 3:** Run bridge (no viewer window should appear)
```bash
ros2 run vla_mujoco_bridge bridge_node
```

**Step 4:** Verify topics still work
```bash
# In separate terminals
ros2 topic hz /joint_states    # Should still be ~100 Hz
ros2 topic hz /camera/image_raw  # Should still be ~30 Hz
ros2 run rqt_image_view rqt_image_view  # Camera feed should still work
```

**Success Criteria:**
- ✓ Bridge starts without GLFW window
- ✓ Topics publish normally
- ✓ Camera renders correctly (EGL offscreen)
- ✓ VRAM ~300-500MB lower than viewer mode

---

### Test 4.2: Long-Duration Stability

**Goal:** Confirm no memory leaks or degradation over 1 hour.

**Setup:**
```bash
# Terminal 1: Start VRAM monitoring
nvidia-smi --query-gpu=timestamp,memory.used --format=csv --loop=10 > vram_1hr.csv

# Terminal 2: Run bridge
ros2 run vla_mujoco_bridge bridge_node

# Terminal 3: Run teleop (optional - occasionally press keys)
ros2 run vla_mujoco_bridge teleop_node
```

**Let it run for 1 hour.** Go do something else.

**After 1 hour, analyze:**
```bash
# Check VRAM growth
python3 << EOF
import pandas as pd
df = pd.read_csv('vram_1hr.csv', names=['time', 'vram_mb'])
df['vram_mb'] = pd.to_numeric(df['vram_mb'], errors='coerce')
initial = df['vram_mb'].iloc[:10].mean()
final = df['vram_mb'].iloc[-10:].mean()
growth = final - initial
print(f"Initial VRAM: {initial:.1f} MB")
print(f"Final VRAM:   {final:.1f} MB")
print(f"Growth:       {growth:.1f} MB ({growth/initial*100:.1f}%)")
print("PASS" if growth < 50 else "FAIL - memory leak suspected")
EOF
```

**Verify in logs:**
- No Python exceptions or ROS2 errors
- Topic rates remain stable (run `ros2 topic hz` at start and end)
- Clean shutdown with Ctrl+C

**Success Criteria:**
- ✓ VRAM growth < 50 MB per hour (<5%)
- ✓ No crashes or exceptions
- ✓ Topic rates stable (no degradation)
- ✓ Clean exit with Ctrl+C

---

## Test Results Summary

After completing all tests, fill in this table:

| Test | Status | Key Metrics | Notes |
|------|--------|-------------|-------|
| **L0: Standalone** | ⬜ Pass / ⬜ Fail | Camera FPS: ___ Hz<br>Physics RT: ___ | |
| **L1.1: Bridge topics** | ⬜ Pass / ⬜ Fail | /joint_states: ___ Hz<br>/camera/image_raw: ___ Hz | |
| **L1.2: Teleop** | ⬜ Pass / ⬜ Fail | Commands publish: ⬜ Yes | |
| **L2: Integration** | ⬜ Pass / ⬜ Fail | Latency: ___ ms | |
| **L3.1: VRAM (viewer)** | ⬜ Pass / ⬜ Fail | Peak: ___ MB<br>Headroom: ___ GB | |
| **L3.1: VRAM (headless)** | ⬜ Pass / ⬜ Fail | Peak: ___ MB<br>Headroom: ___ GB | |
| **L3.2: Physics stability** | ⬜ Pass / ⬜ Fail | RT factor: ___ | |
| **L4.1: Headless mode** | ⬜ Pass / ⬜ Fail | Topics work: ⬜ Yes | |
| **L4.2: 1hr stability** | ⬜ Pass / ⬜ Fail | VRAM growth: ___ MB | |

---

## Known Limitations (Not Bugs)

These are **expected behaviors**, not failures:

1. **Robot falls immediately during teleop**
   - **Why:** Torque-controlled humanoid without balance controller
   - **Resolution:** Phase B.0 prerequisite (gravity compensation or RL standing policy)

2. **High leg torques at rest**
   - **Why:** PD controller fights gravity with no feedforward term
   - **Resolution:** Add `data.qfrc_bias` to PD output (see `tasks/ozkan_todo.md`)

3. **Camera wobbles with body motion**
   - **Why:** Egocentric camera is attached to torso_link
   - **This is correct:** VLA should see what robot sees

4. **Joint limits not enforced in sim**
   - **Why:** MuJoCo doesn't enforce `<joint range=...>` for `<motor>` actuators
   - **Resolution:** Add software limits in PD controller if needed

---

## Phase B Readiness Decision

After completing all tests:

✅ **Proceed to Phase B** if:
- All L0-L4 tests pass
- VRAM headroom > 4GB (headless mode)
- No stability issues in 1-hour test

⚠️ **Caution - modifications needed** if:
- VRAM headroom 3-4GB: Use headless + reduce camera resolution (640×480 → 320×240)
- Topic rates < 90% of target: Debug performance bottlenecks

❌ **Block Phase B** if:
- Camera not rendering in headless mode (critical for VLA training)
- VRAM headroom < 3GB (cannot fit VLA model)
- Memory leaks or crashes in stability test

---

## Next Steps

Once all tests pass:

1. **Document results** in `docs/phase_a_baseline.md`
2. **Update** `tasks/todo.md` with "Phase A Verified ✅"
3. **Commit verification results** to git
4. **Choose VLA model** based on VRAM headroom:
   - **ACT** if headroom > 4GB (recommended)
   - **GR00T N1 INT8** if headroom > 3.5GB (marginal, use headless)
5. **Begin Phase B.0:** Implement gravity compensation (prerequisite for data collection)

---

## Troubleshooting

### Common Issues

**Issue:** `nvidia-smi: command not found`
- **Fix:** VRAM tests will be skipped. Install nvidia-utils or run tests on host (not WSL).

**Issue:** `Camera 'ego_camera' not found in model`
- **Fix:** Check `sim/g1_with_camera.xml` and `sim/models/g1_29dof.xml` - camera must be in `<body name="torso_link">`

**Issue:** `pynput._util.xorg.XError: Failed to connect to X`
- **Fix:** Teleop requires graphical terminal. In WSL, ensure X server (VcXsrv/WSLg) is running.

**Issue:** ROS2 nodes don't see topics
- **Fix:** Source workspace: `source ros2_ws/install/setup.bash`

**Issue:** `colcon build` fails
- **Fix:** Ensure ROS2 sourced: `source /opt/ros/humble/setup.bash`

---

## Appendix: Quick Command Reference

```bash
# Environment setup
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash
cd ~/projects/humanoid_vla

# L0 test
python3 tests/test_l0_standalone.py --duration 60

# L1-L2 manual tests
ros2 run vla_mujoco_bridge bridge_node        # Terminal 1
ros2 run vla_mujoco_bridge teleop_node        # Terminal 2
ros2 topic hz /joint_states                    # Check rates
ros2 topic hz /camera/image_raw
ros2 run rqt_image_view rqt_image_view        # View camera

# VRAM monitoring
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1

# Rebuild after changes
cd ros2_ws && colcon build --packages-select vla_mujoco_bridge
source install/setup.bash
```
