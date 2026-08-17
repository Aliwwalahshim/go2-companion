#!/bin/bash
# lidar_view_rviz.sh - live 3D view of the Go2's built-in LiDAR point cloud
# in RViz2 (proper GPU-rendered orbit/zoom/pan, colored by height).
#
# Sends nothing to the robot - RViz only subscribes to /utlidar/cloud
# (sensor_msgs/PointCloud2), same topic lidar_view.py uses. Safe to run
# any time, on its own or alongside anything else.
#
# RMW_IMPLEMENTATION must be rmw_cyclonedds_cpp - this box's default RMW
# (FastRTPS, if left unset) reliably crashes with "bad_alloc" on this
# Jetson. Also: the LiDAR's own broadcast switch must be ON
# (rt/utlidar/switch, plain SDK topic, OFF by default) or /utlidar/cloud
# never publishes - see lidar_view.py's docstring for how to send it ON.
#
#     ./lidar_view_rviz.sh

set -e
cd "$(dirname "$0")"
source /opt/ros/foxy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
rviz2 -d go2_lidar.rviz
