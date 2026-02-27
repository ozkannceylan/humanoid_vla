# Phase A Verification Quick Start

**30-Second Summary:** Run these commands to verify Phase A is working correctly before Phase B.

---

## Automated Test (5 minutes)

```bash
cd ~/projects/humanoid_vla

# L0: Standalone simulation test
python3 tests/test_l0_standalone.py --duration 60

# L0: Headless test for VRAM profiling
python3 tests/test_l0_standalone.py --duration 60 --headless
```

**Expected:** Both tests pass with camera @ 30Hz, physics real-time factor ~1.0, VRAM < 800MB (headless)

---

## Manual Tests (30 minutes)

### L1: ROS2 Topics

```bash
# Terminal 1
ros2 run vla_mujoco_bridge bridge_node

# Terminal 2 (check rates for 30 seconds each)
ros2 topic hz /joint_states        # Target: 100 Hz
ros2 topic hz /camera/image_raw    # Target: 30 Hz
```

---

### L2: Integration

```bash
# Terminal 1
ros2 run vla_mujoco_bridge bridge_node

# Terminal 2
ros2 run vla_mujoco_bridge teleop_node

# Terminal 3
ros2 run rqt_image_view rqt_image_view  # Select /camera/image_raw
```

**Test:** Press keys in teleop window, verify motion appears in viewer + camera feed

---

### L3: VRAM Profiling (10 minutes)

```bash
# Terminal 1 (background)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1 > gpu_log.csv

# Terminal 2 (run for 5 minutes)
ros2 run vla_mujoco_bridge bridge_node

# Analyze
cat gpu_log.csv | sort -n | tail -1
# Expected: < 1500 MB with viewer
```

**Repeat in headless mode** (modify bridge_node.py line 88, rebuild):
```bash
cd ros2_ws && colcon build --packages-select vla_mujoco_bridge
source install/setup.bash
ros2 run vla_mujoco_bridge bridge_node  # No viewer window
```
Expected: < 800 MB headless → ~5GB headroom for VLA models

---

## Results

After tests, fill in: `docs/phase_a_baseline.md`

**Phase B Readiness:**
- ✅ VRAM headroom > 4GB → Proceed with ACT model
- ⚠️ VRAM headroom 3-4GB → Use headless + reduce camera resolution
- ❌ VRAM headroom < 3GB → Need cloud GPU for VLA

---

## Full Documentation

- **Comprehensive guide:** [phase_a_testing_guide.md](phase_a_testing_guide.md) (all L0-L4 tests)
- **Performance baseline:** [phase_a_baseline.md](phase_a_baseline.md) (fill in your metrics)
- **Project README:** [../README.md](../README.md)

---

## Troubleshooting

**Python not found:**
```bash
which python3  # Use this path instead
```

**ROS2 topics not found:**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

**Camera doesn't work in headless:**
- Check `MUJOCO_GL=egl` is set (automatic in code)
- Verify EGL available: `glxinfo | grep EGL` (install `mesa-utils` if needed)

**VRAM monitoring fails:**
- Install: `sudo apt install nvidia-utils-535` (adjust version)
- Or skip VRAM tests (not critical)

---

**Questions?** See [phase_a_testing_guide.md](phase_a_testing_guide.md) for detailed procedures.
