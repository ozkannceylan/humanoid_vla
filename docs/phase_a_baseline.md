# Phase A Performance Baseline

**Project:** Humanoid VLA - Vision-Language-Action Controlled Humanoid Robot
**Phase:** A - MuJoCo + ROS2 Setup Verification
**Date:** 2026-02-26
**Hardware:** NVIDIA RTX 4050 (6GB VRAM), Windows 11 + WSL Ubuntu 22.04
**MuJoCo:** 3.5.0
**ROS2:** Humble

---

## Executive Summary

Phase A establishes the foundation for VLA-controlled humanoid manipulation:
- ✅ Unitree G1 (29 DOF, torque-controlled) in MuJoCo simulation
- ✅ Egocentric camera pipeline (640×480 @ 30Hz) on torso_link
- ✅ ROS2 bridge with real-time joint state and camera publishing
- ✅ Keyboard teleoperation for demonstration data collection

This document records baseline performance metrics to:
1. Verify system meets real-time requirements for Phase B (VLA control loop)
2. Establish VRAM headroom for VLA model selection (ACT vs GR00T N1)
3. Identify performance bottlenecks before scaling to autonomous control

---

## Test Environment

### Hardware
- **GPU:** NVIDIA GeForce RTX 4050 Laptop (6GB GDDR6)
- **CPU:** ___ (fill in: check with `lscpu`)
- **RAM:** ___ GB (fill in: check with `free -h`)
- **OS:** Windows 11 + WSL2 Ubuntu 22.04

### Software
- **MuJoCo:** 3.5.0 (EGL offscreen rendering)
- **ROS2:** Humble (built 2024-05-23)
- **Python:** 3.10.x (check with `python3 --version`)
- **Key libraries:** `mujoco`, `opencv-python`, `numpy`, `pynput`, `rclpy`, `cv_bridge`

### Model Configuration
- **Robot:** Unitree G1 (29 actuated DOF, torque control)
- **Model file:** `sim/g1_with_camera.xml`
- **Camera:** `ego_camera` (640×480, RGB) attached to `torso_link`
- **Physics:** 500 Hz timestep, implicit integrator
- **PD controller:** Tuned gains (Kp: 3-70 Nm/rad, Kd: 0.1*Kp)

---

## L0: Standalone Simulation Performance

**Test:** `python3 tests/test_l0_standalone.py --duration 60`

### Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Model load time | < 500 ms | ___ ms | ⬜ Pass / ⬜ Fail |
| Camera FPS (avg) | 30 Hz | ___ Hz | ⬜ Pass / ⬜ Fail |
| Physics real-time factor | 1.0 ± 0.05 | ___ | ⬜ Pass / ⬜ Fail |
| VRAM peak (with viewer) | < 1500 MB | ___ MB | ⬜ Pass / ⬜ Fail |
| VRAM peak (headless) | < 800 MB | ___ MB | ⬜ Pass / ⬜ Fail |

### Raw Data (60-second test, with viewer)

```
[Paste output from test_l0_standalone.py here]

Example:
  Model load time:        234.5 ms
  Test duration:          60.1 s
  Physics steps:          30050
  Camera frames:          1803
  Avg camera FPS:         30.0 Hz
  Physics realtime factor: 0.999
  VRAM peak:              1234 MB
  VRAM average:           1189 MB
```

### Analysis

**Camera rendering:**
- Target: 30 Hz (33.3ms per frame)
- Actual: ___ Hz
- ✓ / ✗ Meets requirement for VLA observation frequency

**Physics simulation:**
- Target: 500 Hz (2ms per step)
- Real-time factor: ___
- ✓ / ✗ Can maintain real-time control loop

**Memory usage:**
- Viewer mode: ___ MB
- Headless mode: ___ MB
- Savings from headless: ___ MB (~___%)

---

## L1: ROS2 Component Performance

### L1.1: Bridge Node Topic Rates

**Test:** `ros2 topic hz <topic>` (30-second samples)

| Topic | Target Rate | Measured Rate | Jitter | Status |
|-------|------------|---------------|--------|--------|
| `/joint_states` | 100 Hz | ___ Hz | ___ ms | ⬜ Pass |
| `/camera/image_raw` | 30 Hz | ___ Hz | ___ ms | ⬜ Pass |

**Raw measurements:**
```
[Paste output from ros2 topic hz commands]

Example:
/joint_states:
  average rate: 99.8 Hz
  min: 0.009s max: 0.011s std dev: 0.0003s
```

### L1.2: Teleop Node Responsiveness

| Test | Expected | Observed | Status |
|------|----------|----------|--------|
| Keypress → command published | < 50 ms | ___ ms | ⬜ Pass |
| Command rate | 20 Hz | ___ Hz | ⬜ Pass |
| Joint limits respected | Yes | ⬜ Yes / ⬜ No | ⬜ Pass |
| Reset ('r') zeroes all joints | Yes | ⬜ Yes / ⬜ No | ⬜ Pass |

---

## L2: End-to-End Integration

**Test:** Full teleoperation loop (bridge + teleop + camera viewer)

### Latency Measurements

| Path | Target | Measured | Method |
|------|--------|----------|--------|
| Keypress → viewer motion | < 100 ms | ___ ms | Perceived (stopwatch) |
| Keypress → camera frame | < 100 ms | ___ ms | rqt_image_view timestamp |
| Joint command → physics update | < 10 ms | ___ ms | Code instrumentation |

### Stability Observations

**10-minute teleoperation test:**
- Viewer: ⬜ Stable / ⬜ Freezes / ⬜ Crashes
- Camera feed: ⬜ Stable / ⬜ Freezes / ⬜ Drops frames
- ROS2 nodes: ⬜ No errors / ⬜ Warnings / ⬜ Crashes
- Robot behavior: ⬜ Responsive / ⬜ Laggy / ⬜ Unstable

**Notes:**
___ (e.g., "Robot falls as expected, no balance controller", "Joint motion smooth and responsive")

---

## L3: RTX 4050 VRAM Profiling

### 5-Minute VRAM Stress Test

**Test setup:** Bridge running with viewer, teleop actively moving joints

| Mode | Peak VRAM | Avg VRAM | Min Free | GPU Util |
|------|-----------|----------|----------|----------|
| With viewer | ___ MB | ___ MB | ___ MB | ___% |
| Headless | ___ MB | ___ MB | ___ MB | ___% |

**VRAM timeline (sample every 10s):**
```
[Paste relevant rows from gpu_log.csv]

Time    Used(MB)  Free(MB)  Util(%)
0:00    1234      4910      15
0:10    1245      4899      18
...
```

### Phase B Model Fit Assessment

**Total VRAM:** 6144 MB (RTX 4050)
**Baseline usage (headless):** ___ MB
**Available headroom:** ___ MB (___ GB)

| VLA Model | Size | Fits? | Recommendation |
|-----------|------|-------|----------------|
| **ACT** | ~500 MB | ⬜ Yes / ⬜ Tight / ⬜ No | ___ |
| **GR00T N1 INT8** | ~3500-4000 MB | ⬜ Yes / ⬜ Marginal / ⬜ No | ___ |
| **OpenVLA** | ~7000 MB | ⬜ Yes / ⬜ No | Not recommended for RTX 4050 |

**Recommendations:**
- ✓ / ✗ Can train ACT model on RTX 4050
- ✓ / ✗ Can run GR00T N1 inference (INT8, headless)
- ✓ / ✗ Need cloud GPU for GR00T N1 fine-tuning

---

## L4: Robustness & Stability

### L4.1: Headless Mode Verification

**Critical for Phase B:** VLA training must run headless to save VRAM

| Test | Status | Notes |
|------|--------|-------|
| Bridge starts without viewer | ⬜ Pass / ⬜ Fail | ___ |
| `/joint_states` publishes | ⬜ Pass / ⬜ Fail | Rate: ___ Hz |
| `/camera/image_raw` publishes | ⬜ Pass / ⬜ Fail | Rate: ___ Hz |
| Camera image valid (not black) | ⬜ Pass / ⬜ Fail | rqt_image_view shows ego camera |
| VRAM savings vs viewer mode | ___ MB | ~___% reduction |

**EGL offscreen rendering:** ⬜ Working / ⬜ Failed
**Notes:** ___ (e.g., "EGL works, no GLFW window, camera renders correctly")

---

### L4.2: Long-Duration Stability (1 hour)

**Test:** Bridge + occasional teleop, VRAM monitored every 10 seconds

| Metric | Initial | After 30m | After 1h | Status |
|--------|---------|-----------|----------|--------|
| VRAM used | ___ MB | ___ MB | ___ MB | ⬜ Stable |
| `/joint_states` rate | ___ Hz | ___ Hz | ___ Hz | ⬜ Stable |
| `/camera/image_raw` rate | ___ Hz | ___ Hz | ___ Hz | ⬜ Stable |

**VRAM growth:** ___ MB per hour (___ MB / ___ MB initial = ___%)
- ✓ / ✗ Acceptable (< 50 MB/hr growth = <5%)

**Errors/warnings in logs:**
- ⬜ None (clean run)
- ⬜ Python exceptions: ___ (describe)
- ⬜ ROS2 errors: ___ (describe)

**Exit behavior:**
- ⬜ Clean shutdown with Ctrl+C
- ⬜ Hung (required kill -9)
- ⬜ Crashed

**Notes:** ___ (e.g., "No memory leaks detected, stable over 1 hour")

---

## Known Issues & Limitations

### Expected Behaviors (Not Bugs)

1. **Robot falls during teleoperation**
   - **Status:** Expected
   - **Reason:** Torque-controlled humanoid without balance controller
   - **Impact:** Cannot collect standing manipulation demos yet
   - **Resolution:** Phase B.0 prerequisite - implement gravity compensation or train standing policy
   - **Documented in:** `tasks/ozkan_todo.md`

2. **High leg joint torques at rest**
   - **Status:** Expected
   - **Reason:** PD controller compensates gravity inefficiently (no feedforward term)
   - **Impact:** High power consumption, potential instability
   - **Resolution:** Add `data.qfrc_bias` to PD output (see MuJoCo docs)

3. **Camera wobbles with body motion**
   - **Status:** Expected (correct behavior)
   - **Reason:** Egocentric camera attached to torso_link
   - **Impact:** VLA observes robot's perspective (intended design)
   - **Resolution:** N/A - this is the desired behavior

4. **Joint position limits not enforced in simulation**
   - **Status:** MuJoCo limitation with `<motor>` actuators
   - **Reason:** MuJoCo doesn't enforce `<joint range=...>` for torque actuators
   - **Impact:** Can command positions outside mechanical limits
   - **Resolution:** Enforced in software (teleop_node.py has conservative limits)

### Actual Issues (If Any)

___ (Document any unexpected bugs found during testing)

---

## Performance Bottlenecks

### CPU Usage
- **Physics loop:** ___% (single-threaded)
- **ROS2 executor:** ___% (multi-threaded)
- **Renderer:** ___% (EGL offscreen)
- **Total system:** ___% of ___ cores

### GPU Usage
- **Rendering:** ___% utilization
- **VRAM bandwidth:** ___ (if available from nvidia-smi)

### Identified Bottlenecks
1. ___ (e.g., "Physics loop not bottlenecked, real-time factor > 0.99")
2. ___ (e.g., "Camera rendering not bottlenecked, consistent 30 FPS")
3. ___ (e.g., "ROS2 publishing has minimal overhead")

---

## Conclusions

### Summary

| Component | Status | Performance | Notes |
|-----------|--------|-------------|-------|
| MuJoCo simulation | ⬜ Pass | ___ RT factor | ___ |
| Camera pipeline | ⬜ Pass | ___ Hz | ___ |
| ROS2 bridge | ⬜ Pass | ___/100 Hz joints, ___/30 Hz camera | ___ |
| Teleoperation | ⬜ Pass | ___ ms latency | ___ |
| VRAM headroom | ⬜ Sufficient / ⬜ Tight | ___ GB free | ___ |
| Long-term stability | ⬜ Pass | No leaks, ___% growth | ___ |

### Phase B Readiness

**✅ Ready for Phase B** if all of the following:
- ⬜ All L0-L4 tests pass
- ⬜ VRAM headroom > 4 GB (for ACT model)
- ⬜ No memory leaks or stability issues
- ⬜ Headless mode works correctly

**⚠️ Proceed with caution** if:
- ⬜ VRAM headroom 3-4 GB (use headless, reduce camera resolution)
- ⬜ Minor performance issues (can be optimized in Phase B)

**❌ Block Phase B** if:
- ⬜ Camera doesn't work in headless mode
- ⬜ VRAM headroom < 3 GB
- ⬜ Memory leaks or crashes

### Recommended VLA Model

Based on VRAM profiling:
- **Primary choice:** ⬜ ACT (~500MB) / ⬜ GR00T N1 INT8 (~4GB)
- **Reasoning:** ___
- **Configuration:** ⬜ Headless required / ⬜ Viewer OK

### Next Steps

1. ⬜ **Phase B.0:** Implement gravity compensation (`data.qfrc_bias` in PD controller)
   - **Why:** Robot must stand stably before collecting demos
   - **Timeline:** 2-3 days
   - **Resource:** MuJoCo docs, ozkan_todo.md

2. ⬜ **Phase B.1:** Set up LeRobot / unitree_IL_lerobot
   - **Goal:** Demonstration data collection pipeline
   - **Timeline:** 1 week

3. ⬜ **Phase B.2:** Train ACT model on simple pick task
   - **Goal:** Validate VLA control loop works end-to-end
   - **Timeline:** 2 weeks

---

## Appendix: Raw Test Outputs

### L0 Test (Full Output)
```
[Paste complete output from test_l0_standalone.py]
```

### ROS2 Topic Hz (Sample Output)
```
[Paste output from ros2 topic hz tests]
```

### VRAM Log (1-hour test, first/last 20 lines)
```
[Paste head and tail of gpu_log.csv]
```

---

**Document prepared by:** Claude Sonnet 4.5 (AI Assistant)
**Reviewed by:** ___ (User to sign off after completing tests)
**Date finalized:** ___ (After all metrics filled in)
**Git commit:** ___ (Hash of commit with test results)
