#!/usr/bin/env bash
BOOSTER_IP="${BOOSTER_IP:?set BOOSTER_IP to the robot address (ask a teammate; never commit it)}"
export DEBIAN_FRONTEND=noninteractive
export TZ=Europe/Amsterdam
set -euo pipefail
apt update && apt install software-properties-common curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
  apt-key add -
sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu jammy main" \
  > /etc/apt/sources.list.d/ros2.list'
apt update
apt install ros-humble-desktop ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-joint-state-publisher-gui -y
# Clone and build the K1 ROS2 package
mkdir -p ~/k1_ws/src && cd ~/k1_ws/src
git clone https://github.com/BoosterRobotics/booster_robotics_sdk_ros2.git
cd ~/k1_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
# Launch the K1 bridge
source ~/k1_ws/install/setup.bash
ros2 launch booster_ros2 k1_bringup.launch.py \
  robot_ip:=${BOOSTER_IP}
# Inspect joint states
# List all available topics
ros2 topic list