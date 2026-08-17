"""
pose_server.py - ROS 2 odometry node, served over UDP.

RUN THIS IN A foxy-SOURCED SHELL. It imports rclpy.
The SDK/control process reads it through pose_client.py (plain UDP).

    Terminal B (foxy-sourced):  python3 pose_server.py

TOPIC UNCERTAINTY
-----------------
The Go2 firmware has shipped robot odometry under more than one name and
message type. This node subscribes to BOTH and uses whichever is actually
publishing:

    /utlidar/robot_odom   nav_msgs/Odometry     (~150 Hz, always on)
    /utlidar/robot_pose   geometry_msgs/PoseStamped

Confirm what your firmware exposes with:
    ros2 topic list | grep utlidar
    ros2 topic info /utlidar/robot_odom

Override with --odom-topic / --pose-topic if your names differ.

WIRE PROTOCOL (JSON over UDP, request/reply)
--------------------------------------------
    request : anything (a single byte is fine)
    reply   : {"ok": bool,     # False until the first message arrives
               "x": float, "y": float, "yaw": float,   # metres, radians
               "src": str,     # which topic it came from
               "t": float}     # epoch seconds of the last update
"""

import argparse
import json
import math
import socket
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
except ImportError as _exc:
    sys.stderr.write(
        "ERROR: could not import ROS 2 (%s).\n"
        "This server MUST run in a foxy-sourced shell, e.g.:\n"
        "    source /opt/ros/foxy/setup.bash && python3 pose_server.py\n" % _exc)
    sys.exit(1)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47002
DEFAULT_ODOM_TOPIC = "/utlidar/robot_odom"
DEFAULT_POSE_TOPIC = "/utlidar/robot_pose"


def yaw_from_quat(qx, qy, qz, qw):
    """Z-axis rotation from a quaternion (standard ROS convention)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseNode(Node):
    def __init__(self, odom_topic, pose_topic):
        super().__init__("go2_pose_server")
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.stamp = 0.0
        self.src = ""
        self.count = 0
        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_subscription(PoseStamped, pose_topic, self.on_pose, 10)

    def _store(self, pose, src):
        p = pose.position
        q = pose.orientation
        self.x = float(p.x)
        self.y = float(p.y)
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.stamp = time.time()
        self.src = src
        self.count += 1

    def on_odom(self, msg):
        self._store(msg.pose.pose, "robot_odom")

    def on_pose(self, msg):
        # only used if robot_odom is silent on this firmware
        if self.src == "robot_odom" and time.time() - self.stamp < 1.0:
            return
        self._store(msg.pose, "robot_pose")

    def snapshot(self):
        return dict(ok=self.count > 0, x=self.x, y=self.y, yaw=self.yaw,
                    src=self.src, t=self.stamp)


def main():
    ap = argparse.ArgumentParser(description="Go2 pose UDP server")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    ap.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((args.host, args.port))

    rclpy.init(args=None)
    node = PoseNode(args.odom_topic, args.pose_topic)

    print("[pose_server] listening on %s:%d" % (args.host, args.port))
    print("[pose_server] subscribed to %s (Odometry) and %s (PoseStamped)"
          % (args.odom_topic, args.pose_topic))

    last_report = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)

            while True:
                try:
                    _, addr = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    sock.sendto(json.dumps(node.snapshot()).encode("utf-8"), addr)
                except OSError:
                    pass

            now = time.time()
            if now - last_report >= 5.0:
                last_report = now
                if node.count:
                    print("[pose_server] %s  x=%.2f y=%.2f yaw=%.1fdeg  n=%d"
                          % (node.src, node.x, node.y,
                             math.degrees(node.yaw), node.count))
                else:
                    print("[pose_server] no odometry yet - check "
                          "'ros2 topic list | grep utlidar'")
    except KeyboardInterrupt:
        print("\n[pose_server] interrupted.")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()
        print("[pose_server] stopped.")


if __name__ == "__main__":
    main()
