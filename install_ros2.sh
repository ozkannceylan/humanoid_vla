#!/usr/bin/env bash
# Phase A — ROS2 Humble installation script for Ubuntu 22.04
# Run with: bash install_ros2.sh
# This script requires sudo and will prompt for your password.

set -e  # Exit on first error

echo "=== [1/6] Setting locale ==="
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "=== [2/6] Enabling universe repository ==="
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y

echo "=== [3/6] Adding ROS2 apt source ==="
sudo apt install -y curl
ROS_APT_SOURCE_VERSION=$(curl -s \
  https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

echo "=== [4/6] Installing ROS2 Humble desktop ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs

echo "=== [5/6] Initialising rosdep ==="
sudo rosdep init || echo "(rosdep already initialised — OK)"
rosdep update

echo "=== [6/6] Adding ROS2 source to ~/.bashrc ==="
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  echo "Added to ~/.bashrc"
else
  echo "Already in ~/.bashrc — skipping"
fi

echo ""
echo "========================================"
echo " ROS2 Humble installation COMPLETE"
echo " Run: source /opt/ros/humble/setup.bash"
echo "========================================"
