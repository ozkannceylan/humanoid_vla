#!/usr/bin/env bash
# Phase A -- ROS2 install script
# Ubuntu 24.04 (Noble)  --> ROS2 Jazzy
# Ubuntu 22.04 (Jammy)  --> ROS2 Humble
# Run with: bash install_ros2.sh

set -e

. /etc/os-release
case "${VERSION_CODENAME}" in
  noble)  ROS_DISTRO="jazzy" ;;
  jammy)  ROS_DISTRO="humble" ;;
  *) echo "Unsupported Ubuntu version: ${VERSION_CODENAME}"; exit 1 ;;
esac
echo "Detected Ubuntu ${VERSION_CODENAME} -> installing ROS2 ${ROS_DISTRO}"

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
  | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

echo "=== [4/6] Installing ROS2 ${ROS_DISTRO} desktop ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  ros-${ROS_DISTRO}-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-std-msgs

echo "=== [5/6] Initialising rosdep ==="
sudo rosdep init || echo "(rosdep already initialised -- OK)"
rosdep update

echo "=== [6/6] Adding ROS2 source to ~/.bashrc ==="
if ! grep -q "source /opt/ros/${ROS_DISTRO}/setup.bash" ~/.bashrc; then
  echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
  echo "Added to ~/.bashrc"
else
  echo "Already in ~/.bashrc -- skipping"
fi

echo ""
echo "========================================"
echo " ROS2 ${ROS_DISTRO} installation COMPLETE"
echo " Run: source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "========================================"
