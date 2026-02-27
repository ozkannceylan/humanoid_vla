# Complete Installation Guide - Humanoid VLA Project

**Target:** Fresh Windows 11 laptop setup for Phase A verification
**Time:** 2-3 hours
**Requirements:** Windows 11, ~20GB free space, internet connection

---

## Part 1: WSL2 Ubuntu 22.04 Setup (30 minutes)

### Step 1: Enable WSL2 (if not already enabled)

Open **PowerShell as Administrator** and run:

```powershell
# Enable WSL
wsl --install

# If already installed, update to WSL2
wsl --set-default-version 2

# Restart computer if prompted
```

---

### Step 2: Install Ubuntu 22.04

```powershell
# Install Ubuntu 22.04 from Microsoft Store
wsl --install -d Ubuntu-22.04

# Or if already installed, set as default
wsl --set-default Ubuntu-22.04
```

**First launch:**
- Create UNIX username (e.g., `ozkan`)
- Set password
- Remember these credentials!

---

### Step 3: Update Ubuntu

Open **Ubuntu 22.04** from Start menu:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential git curl wget vim \
  software-properties-common apt-transport-https ca-certificates
```

---

## Part 2: ROS2 Humble Installation (20 minutes)

### Step 1: Add ROS2 Repository

```bash
# Add ROS2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update

# Add ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
```

---

### Step 2: Install ROS2 Humble Desktop

```bash
# Install ROS2 Humble (full desktop install)
sudo apt install -y ros-humble-desktop

# Install development tools
sudo apt install -y python3-colcon-common-extensions \
  python3-rosdep python3-argcomplete

# Initialize rosdep
sudo rosdep init
rosdep update
```

---

### Step 3: Configure ROS2 Environment

```bash
# Add to ~/.bashrc (or ~/.zshrc if using zsh)
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify installation
ros2 --version
# Expected: ros2 cli version: 0.x.x
```

---

## Part 3: Python & MuJoCo Setup (15 minutes)

### Step 1: Install Python Dependencies

```bash
# Update pip
python3 -m pip install --upgrade pip

# Install core Python packages (system-wide for ROS2 compatibility)
pip3 install mujoco opencv-python numpy pynput

# Install ROS2 Python bridge tools
pip3 install cv-bridge

# Verify installations
python3 -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')"
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

**Expected output:**
```
MuJoCo 3.5.0 (or newer)
OpenCV 4.x.x
NumPy 1.x.x
```

---

## Part 4: GPU Support (NVIDIA Only) (10 minutes)

**Only if you have NVIDIA GPU and want GPU acceleration**

### Step 1: Check GPU

```bash
# Check if NVIDIA GPU is available
lspci | grep -i nvidia
```

---

### Step 2: Install NVIDIA CUDA Toolkit (Optional)

```bash
# Install CUDA toolkit (for GPU-accelerated rendering)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Install nvidia-utils for nvidia-smi
sudo apt install -y nvidia-utils-535
```

**Note:** MuJoCo uses EGL offscreen rendering by default, which works without NVIDIA drivers in WSL2. GPU support is optional for better performance.

---

## Part 5: Clone Project & Setup (10 minutes)

### Step 1: Clone Repository

```bash
# Create projects directory
mkdir -p ~/projects
cd ~/projects

# Clone the repository (if not already cloned)
# Option A: If you have it on this laptop already, just move it
# Option B: Clone fresh
git clone <your-repo-url> humanoid_vla
cd humanoid_vla
```

**Or if project already exists in Windows:**

```bash
# Access Windows files from WSL
cd /mnt/c/Users/ozkan/projects/humanoid_vla
# Or copy to WSL home for better performance
cp -r /mnt/c/Users/ozkan/projects/humanoid_vla ~/projects/
cd ~/projects/humanoid_vla
```

---

### Step 2: Download Unitree G1 Models

```bash
# Create repos directory
mkdir -p repos
cd repos

# Clone Unitree MuJoCo repository
git clone https://github.com/unitreerobotics/unitree_mujoco.git

# Clone MuJoCo Menagerie (optional, for reference)
git clone https://github.com/google-deepmind/mujoco_menagerie.git

cd ..
```

---

### Step 3: Update Model Paths

The model file needs absolute paths. Update `sim/models/g1_29dof.xml`:

```bash
# Get absolute path to meshes
MESH_PATH="$(pwd)/repos/unitree_mujoco/unitree_robots/g1/meshes"
echo "Mesh path: $MESH_PATH"

# Update meshdir in model file (line 2)
# Open in editor:
vim sim/models/g1_29dof.xml
# Or use sed:
sed -i "s|meshdir=\".*\"|meshdir=\"$MESH_PATH\"|" sim/models/g1_29dof.xml
```

**Verify the change:**
```bash
head -n 3 sim/models/g1_29dof.xml
```

Should show:
```xml
<mujoco model="g1_29dof">
  <compiler angle="radian" meshdir="/home/ozkan/projects/humanoid_vla/repos/unitree_mujoco/unitree_robots/g1/meshes" />
```

---

## Part 6: Build ROS2 Workspace (15 minutes)

### Step 1: Build the Package

```bash
cd ~/projects/humanoid_vla

# Source ROS2
source /opt/ros/humble/setup.bash

# Build the workspace
cd ros2_ws
colcon build --packages-select vla_mujoco_bridge

# Source the workspace
source install/setup.bash
```

**Expected output:**
```
Starting >>> vla_mujoco_bridge
Finished <<< vla_mujoco_bridge [X.Xs]

Summary: 1 package finished [X.Xs]
```

---

### Step 2: Add to Startup

```bash
# Add workspace sourcing to ~/.bashrc
echo "source ~/projects/humanoid_vla/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## Part 7: Verify Installation (10 minutes)

### Test 1: Standalone Simulation

```bash
cd ~/projects/humanoid_vla

# Test MuJoCo loads G1 model
python3 sim/test_g1.py
```

**Expected:**
- Viewer window opens with G1 robot
- Camera preview window shows egocentric view
- Console shows "actual_fps ~30" every 3 seconds
- Robot falls (expected - no balance controller)
- Press 'q' in camera window to exit

**If viewer doesn't open:** Run in headless mode:
```bash
# Edit test_g1.py, set DISPLAY_PREVIEW = False on line 31
# Or run automated test:
python3 tests/test_l0_standalone.py --duration 10 --headless
```

---

### Test 2: ROS2 Bridge

```bash
# Terminal 1: Bridge
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge bridge_node
```

**Expected:**
- "MuJoCo bridge node ready" message
- Viewer opens (or headless if configured)
- No errors

**Terminal 2: Check topics**
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```

**Expected output:**
```
/camera/image_raw
/joint_commands
/joint_states
/parameter_events
/rosout
```

**Ctrl+C to stop bridge**

---

### Test 3: Automated Verification

```bash
cd ~/projects/humanoid_vla

# Run L0 test
python3 tests/test_l0_standalone.py --duration 10
```

**Expected:**
```
✅ TEST PASSED

Performance Metrics:
  Model load time:        < 500 ms
  Avg camera FPS:         ≥ 28 Hz
  Physics realtime factor: 0.95-1.05
```

---

## Part 8: WSL2 Performance Optimization (Optional)

### Configure WSL Memory

Create/edit `C:\Users\ozkan\.wslconfig` (in Windows, not WSL):

```ini
[wsl2]
memory=8GB
processors=4
swap=4GB
```

**Restart WSL:**
```powershell
# In Windows PowerShell
wsl --shutdown
# Then reopen Ubuntu
```

---

## Troubleshooting

### Issue: "Python not found"

```bash
# Check Python path
which python3
# Should output: /usr/bin/python3

# If not found:
sudo apt install -y python3 python3-pip
```

---

### Issue: "Camera 'ego_camera' not found"

**Problem:** Mesh path incorrect in model file

**Fix:**
```bash
cd ~/projects/humanoid_vla
# Check if meshes exist
ls repos/unitree_mujoco/unitree_robots/g1/meshes/
# Should show .obj files

# Update path in sim/models/g1_29dof.xml (line 2)
vim sim/models/g1_29dof.xml
# Set meshdir to absolute path
```

---

### Issue: "ros2: command not found"

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Add to ~/.bashrc permanently
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

---

### Issue: "No module named 'rclpy'"

```bash
# Install ROS2 Python packages
sudo apt install -y python3-rclpy python3-cv-bridge

# Or reinstall ROS2 desktop
sudo apt install --reinstall ros-humble-desktop
```

---

### Issue: Viewer window doesn't open in WSL

**Problem:** WSL2 GUI support requires WSLg (Windows 11 only)

**Fix 1:** Use headless mode (modify `bridge_node.py` line 88: `launch_viewer=False`)

**Fix 2:** Install X server on Windows
```powershell
# Install VcXsrv from: https://sourceforge.net/projects/vcxsrv/
# Launch XLaunch with default settings
```

```bash
# In WSL, set DISPLAY
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
echo "export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0" >> ~/.bashrc
```

---

## Installation Complete! ✅

### Quick Reference

**Start ROS2 bridge:**
```bash
source /opt/ros/humble/setup.bash
source ~/projects/humanoid_vla/ros2_ws/install/setup.bash
ros2 run vla_mujoco_bridge bridge_node
```

**Run verification tests:**
```bash
cd ~/projects/humanoid_vla
python3 tests/test_l0_standalone.py --duration 60
```

**Full test guide:**
```bash
cat docs/phase_a_testing_guide.md
```

---

## Next Steps

1. ✅ Installation complete
2. 📋 Run verification tests: `python3 tests/test_l0_standalone.py`
3. 📊 Follow comprehensive testing: `docs/phase_a_testing_guide.md`
4. 📝 Document results: `docs/phase_a_baseline.md`
5. 🚀 Ready for Phase B!

---

**Estimated total time:** 2-3 hours
**Questions?** See troubleshooting section above or comprehensive test guide.
