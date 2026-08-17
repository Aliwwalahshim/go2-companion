"""
pose_client.py - UDP client for pose_server.py. NO rclpy.

Safe to import alongside the Unitree SDK (standard library only).

    pc = PoseClient()
    if not pc.wait_ready(5.0):
        print("pose_server.py is not running")
    p = pc.get()
    if p:
        p.x, p.y, p.yaw          # metres, metres, radians
"""

import json
import math
import socket
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47002
DEFAULT_TIMEOUT = 0.08
STALE_AFTER = 1.0          # odometry runs ~150 Hz; 1 s old means it died


class Pose:
    __slots__ = ("x", "y", "yaw", "src", "age")

    def __init__(self, x, y, yaw, src, age):
        self.x, self.y, self.yaw, self.src, self.age = x, y, yaw, src, age

    def __repr__(self):
        return "Pose(x=%.2f, y=%.2f, yaw=%.1fdeg, src=%s, age=%.2fs)" % (
            self.x, self.y, math.degrees(self.yaw), self.src, self.age)


class PoseClient:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 timeout=DEFAULT_TIMEOUT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def get(self):
        """Current (x, y, yaw), or None if unavailable / stale."""
        try:
            self.sock.sendto(b"?", self.addr)
            data, _ = self.sock.recvfrom(65535)
            rep = json.loads(data.decode("utf-8"))
        except (socket.timeout, OSError, ValueError):
            return None
        if not rep.get("ok"):
            return None
        age = time.time() - float(rep.get("t", 0.0))
        if age > STALE_AFTER:
            return None
        return Pose(float(rep["x"]), float(rep["y"]), float(rep["yaw"]),
                    str(rep.get("src", "")), age)

    def wait_ready(self, seconds=5.0):
        """Block until the server delivers a valid pose, or time out."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.get() is not None:
                return True
            time.sleep(0.1)
        return False

    def shutdown(self):
        try:
            self.sock.close()
        except OSError:
            pass


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]. Shared by go_to_room and the bridge."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
