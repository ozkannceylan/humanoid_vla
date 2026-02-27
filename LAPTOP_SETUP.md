# Laptop Setup Quick Guide

**Goal:** Set up this laptop for Phase A verification testing
**Current status:** Windows 11, no WSL/ROS2 installed yet
**Time:** 2-3 hours total

---

## 🚀 Quick Start (Choose Your Path)

### Path A: Automated Installation (Recommended)

**1. Install WSL2 Ubuntu 22.04**

Open **PowerShell as Administrator**:
```powershell
# Install WSL with Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Restart computer when prompted
```

**2. First Launch Setup**
- Open "Ubuntu 22.04" from Start menu
- Create username (e.g., `ozkan`)
- Set password
- Wait for setup to complete

**3. Copy Project to WSL**

In Ubuntu terminal:
```bash
# Copy from Windows to WSL (faster performance)
cp -r /mnt/c/Users/ozkan/projects/humanoid_vla ~/projects/
cd ~/projects/humanoid_vla

# Or work directly from Windows (slower but convenient)
cd /mnt/c/Users/ozkan/projects/humanoid_vla
```

**4. Run Automated Install**

```bash
# Make script executable (if not already)
chmod +x install.sh

# Run installation
bash install.sh
```

This will:
- ✅ Update system packages
- ✅ Install ROS2 Humble (~15 min)
- ✅ Install Python packages
- ✅ Clone Unitree models
- ✅ Update model paths
- ✅ Build ROS2 workspace
- ✅ Run verification test

**Expected time:** 30-45 minutes

---

### Path B: Manual Installation

Follow step-by-step guide: `docs/INSTALLATION.md`

---

## After Installation

### 1. Verify Everything Works

**Close and reopen Ubuntu terminal**, then:

```bash
cd ~/projects/humanoid_vla

# Test 1: Standalone simulation (10 seconds)
python3 tests/test_l0_standalone.py --duration 10 --headless

# Expected: ✅ TEST PASSED
```

---

### 2. Test ROS2 Bridge

```bash
# Source environment (already in .bashrc, but run once manually)
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash

# Run bridge
ros2 run vla_mujoco_bridge bridge_node
```

**Expected:**
- "MuJoCo bridge node ready" message
- No errors
- Ctrl+C to stop

---

### 3. Run Full Verification

```bash
cd ~/projects/humanoid_vla

# Run comprehensive L0 test (60 seconds)
python3 tests/test_l0_standalone.py --duration 60 --headless
```

**Expected output:**
```
✅ TEST PASSED

Performance Metrics:
  Model load time:        < 500 ms
  Avg camera FPS:         ≥ 28 Hz
  Physics realtime factor: 0.95-1.05
  VRAM peak:              < 800 MB
  VRAM headroom:          > 4.5 GB

  ACT model (~500MB):     ✓ Fits
```

---

### 4. Follow Full Test Guide

```bash
# Read comprehensive testing guide
cat docs/phase_a_testing_guide.md

# Or view quick reference
cat docs/VERIFICATION_QUICKSTART.md
```

---

## Troubleshooting Common Issues

### Issue: WSL not installed

**Fix:**
```powershell
# In PowerShell as Administrator
wsl --install
# Restart computer
```

---

### Issue: "install.sh: Permission denied"

**Fix:**
```bash
chmod +x install.sh
bash install.sh
```

---

### Issue: ROS2 commands not found after installation

**Fix:**
```bash
# Close and reopen terminal (to reload .bashrc)
# Or manually source:
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash
```

---

### Issue: Viewer window doesn't open

**Expected in WSL** - GUI support can be tricky. Use headless mode:
```bash
# Run tests in headless mode
python3 tests/test_l0_standalone.py --duration 10 --headless

# For ROS2 bridge, edit bridge_node.py line 88:
# Change: launch_viewer=True
# To:     launch_viewer=False
```

---

### Issue: "Camera 'ego_camera' not found"

**Fix:** Model paths need updating
```bash
cd ~/projects/humanoid_vla

# Check mesh directory exists
ls repos/unitree_mujoco/unitree_robots/g1/meshes/

# If not found, run install script again or update manually
# Edit sim/models/g1_29dof.xml line 2
vim sim/models/g1_29dof.xml
```

---

## Quick Command Reference

```bash
# After installation, these should work:

# Test standalone sim
python3 sim/test_g1.py

# Automated verification
python3 tests/test_l0_standalone.py --duration 60

# Run ROS2 bridge
ros2 run vla_mujoco_bridge bridge_node

# Run teleop
ros2 run vla_mujoco_bridge teleop_node

# Check topics
ros2 topic list
ros2 topic hz /joint_states
ros2 topic hz /camera/image_raw
```

---

## Hardware Notes

**Your laptop specs (to check):**
```bash
# Check CPU
lscpu | grep "Model name"

# Check RAM
free -h

# Check GPU (if NVIDIA)
lspci | grep -i nvidia
# Or: nvidia-smi (if drivers installed)
```

**For VRAM profiling:** Only works if you have NVIDIA GPU with nvidia-smi. Otherwise, skip L3 VRAM tests.

---

## What's Different from Other PC?

**Other PC (Phase A complete):**
- ✅ Installed and tested
- ✅ RTX 4050 (6GB VRAM)
- ✅ Everything verified working

**This laptop (new setup):**
- 🔄 Installing fresh
- ❓ GPU specs unknown (check with `lspci | grep -i vga`)
- ❓ May or may not have NVIDIA GPU

**If no NVIDIA GPU:** Can still run everything except VRAM profiling (L3 tests).

---

## Next Steps After Setup

1. ✅ Installation complete
2. 📋 Run `python3 tests/test_l0_standalone.py --duration 60`
3. 📖 Follow `docs/phase_a_testing_guide.md`
4. 📊 Document results in `docs/phase_a_baseline.md`
5. 🚀 Ready for Phase B!

---

## Time Breakdown

| Step | Time |
|------|------|
| WSL2 install + restart | 10 min |
| Run install.sh | 30-45 min |
| Verification tests | 10-60 min |
| **Total** | **1-2 hours** |

*Manual installation adds 1-2 hours more*

---

## Support

- **Installation guide:** `docs/INSTALLATION.md`
- **Testing guide:** `docs/phase_a_testing_guide.md`
- **Quick reference:** `docs/VERIFICATION_QUICKSTART.md`

---

**Ready to start?** Open PowerShell as Administrator and run:
```powershell
wsl --install -d Ubuntu-22.04
```
