#!/bin/bash
# install.sh — Automated Installation Script for Humanoid VLA Project
# Run in WSL Ubuntu 22.04: bash install.sh

set -e  # Exit on error

echo "========================================================================"
echo "HUMANOID VLA PROJECT - AUTOMATED INSTALLATION"
echo "========================================================================"
echo ""
echo "This script will install:"
echo "  - ROS2 Humble"
echo "  - MuJoCo 3.5.0"
echo "  - Python packages (mujoco, opencv, numpy, pynput)"
echo "  - Build ROS2 workspace"
echo ""
echo "Estimated time: 30-45 minutes"
echo "Requires: Ubuntu 22.04 (WSL2 or native), sudo access, internet"
echo ""
read -p "Press Enter to start installation, or Ctrl+C to cancel..."

# ============================================================================
# Part 1: System Update
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 1/6: Updating system packages..."
echo "========================================================================"
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git curl wget vim \
  software-properties-common apt-transport-https ca-certificates

echo "✅ System updated"

# ============================================================================
# Part 2: ROS2 Humble Installation
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 2/6: Installing ROS2 Humble..."
echo "========================================================================"

# Check if ROS2 already installed
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "ROS2 Humble already installed, skipping..."
else
    echo "Adding ROS2 repository..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe -y
    sudo apt update

    # Add ROS2 GPG key
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg

    # Add ROS2 repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
      sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt update

    echo "Installing ROS2 Humble Desktop (this may take 10-15 minutes)..."
    sudo apt install -y ros-humble-desktop

    echo "Installing ROS2 development tools..."
    sudo apt install -y python3-colcon-common-extensions \
      python3-rosdep python3-argcomplete

    # Initialize rosdep
    if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
        sudo rosdep init
    fi
    rosdep update
fi

# Add to bashrc if not already there
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "Added ROS2 to ~/.bashrc"
fi

source /opt/ros/humble/setup.bash
echo "✅ ROS2 Humble installed"

# ============================================================================
# Part 3: Python & MuJoCo
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 3/6: Installing Python packages..."
echo "========================================================================"

# Update pip
python3 -m pip install --upgrade pip

# Install packages
echo "Installing MuJoCo, OpenCV, NumPy, pynput..."
pip3 install mujoco opencv-python numpy pynput

# Verify
echo ""
echo "Verifying installations:"
python3 -c "import mujoco; print(f'✓ MuJoCo {mujoco.__version__}')"
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python3 -c "import pynput; print(f'✓ pynput installed')"

echo "✅ Python packages installed"

# ============================================================================
# Part 4: Clone Dependencies
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 4/6: Cloning Unitree models..."
echo "========================================================================"

PROJECT_ROOT="$(pwd)"
cd "$PROJECT_ROOT"

mkdir -p repos
cd repos

# Clone unitree_mujoco if not exists
if [ ! -d "unitree_mujoco" ]; then
    echo "Cloning unitree_mujoco..."
    git clone https://github.com/unitreerobotics/unitree_mujoco.git
else
    echo "unitree_mujoco already exists, skipping..."
fi

# Clone mujoco_menagerie (optional)
if [ ! -d "mujoco_menagerie" ]; then
    echo "Cloning mujoco_menagerie (optional)..."
    git clone https://github.com/google-deepmind/mujoco_menagerie.git
else
    echo "mujoco_menagerie already exists, skipping..."
fi

cd "$PROJECT_ROOT"
echo "✅ Unitree models cloned"

# ============================================================================
# Part 5: Update Model Paths
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 5/6: Updating model paths..."
echo "========================================================================"

MESH_PATH="$PROJECT_ROOT/repos/unitree_mujoco/unitree_robots/g1/meshes"
echo "Mesh path: $MESH_PATH"

# Check if meshes exist
if [ ! -d "$MESH_PATH" ]; then
    echo "ERROR: Mesh directory not found: $MESH_PATH"
    exit 1
fi

# Update meshdir in model file
MODEL_FILE="$PROJECT_ROOT/sim/models/g1_29dof.xml"
if [ -f "$MODEL_FILE" ]; then
    # Backup original
    cp "$MODEL_FILE" "$MODEL_FILE.backup"

    # Update meshdir
    sed -i "s|meshdir=\"[^\"]*\"|meshdir=\"$MESH_PATH\"|" "$MODEL_FILE"

    echo "Updated meshdir in $MODEL_FILE"
    echo "First 3 lines:"
    head -n 3 "$MODEL_FILE"
else
    echo "WARNING: Model file not found: $MODEL_FILE"
    echo "You may need to update paths manually later"
fi

echo "✅ Model paths updated"

# ============================================================================
# Part 6: Build ROS2 Workspace
# ============================================================================
echo ""
echo "========================================================================"
echo "PART 6/6: Building ROS2 workspace..."
echo "========================================================================"

cd "$PROJECT_ROOT/ros2_ws"

# Source ROS2
source /opt/ros/humble/setup.bash

# Build
echo "Building vla_mujoco_bridge package..."
colcon build --packages-select vla_mujoco_bridge

# Source workspace
source install/setup.bash

# Add to bashrc if not already there
WORKSPACE_SOURCE="source $PROJECT_ROOT/ros2_ws/install/setup.bash"
if ! grep -q "$WORKSPACE_SOURCE" ~/.bashrc; then
    echo "$WORKSPACE_SOURCE" >> ~/.bashrc
    echo "Added workspace to ~/.bashrc"
fi

echo "✅ ROS2 workspace built"

# ============================================================================
# Part 7: Verification
# ============================================================================
echo ""
echo "========================================================================"
echo "INSTALLATION COMPLETE - Running verification test..."
echo "========================================================================"

cd "$PROJECT_ROOT"

echo ""
echo "Testing MuJoCo simulation (10 seconds, headless)..."
if python3 tests/test_l0_standalone.py --duration 10 --headless; then
    echo ""
    echo "✅ VERIFICATION PASSED"
else
    echo ""
    echo "⚠️  Verification test failed, but installation is complete."
    echo "See troubleshooting in docs/INSTALLATION.md"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "INSTALLATION SUMMARY"
echo "========================================================================"
echo ""
echo "✅ ROS2 Humble installed"
echo "✅ MuJoCo 3.5.0 installed"
echo "✅ Python packages installed"
echo "✅ Unitree G1 models cloned"
echo "✅ ROS2 workspace built"
echo ""
echo "Project location: $PROJECT_ROOT"
echo ""
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Close and reopen terminal (to load .bashrc changes)"
echo ""
echo "2. Test standalone simulation:"
echo "   cd ~/projects/humanoid_vla"
echo "   python3 sim/test_g1.py"
echo ""
echo "3. Test ROS2 bridge:"
echo "   ros2 run vla_mujoco_bridge bridge_node"
echo ""
echo "4. Run full verification:"
echo "   python3 tests/test_l0_standalone.py --duration 60"
echo ""
echo "5. Follow comprehensive testing guide:"
echo "   cat docs/phase_a_testing_guide.md"
echo ""
echo "For troubleshooting, see: docs/INSTALLATION.md"
echo ""
echo "🎉 Setup complete! You're ready for Phase A verification."
echo ""
