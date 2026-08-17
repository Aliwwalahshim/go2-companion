"""
lidar_client.py - UDP client for lidar_node.py. NO rclpy.

Safe to import alongside the Unitree SDK: this file only uses the standard
library plus cv2/numpy for debug_view(), so it cannot drag CycloneDDS into
the control process.

SEAM (drop-in for the old in-process LidarAvoider)
--------------------------------------------------
    avoider = LidarClient()
    adj = avoider.compute(persons)   # persons = [(dist_m, ang_deg, half_deg)]
    adj.block_forward   # True -> forward speed must be 0
    adj.fwd_scale       # 0..1 multiplier for forward speed
    adj.lat             # sideways dodge velocity (+ = left)
    adj.status, adj.have_scan
    adj.pts             # [(range_m, angle_deg), ...] obstacle points, for display
    cv2.imshow("LiDAR", avoider.debug_view())   # same style as DepthAvoider
    avoider.shutdown()

FAIL-SAFE
---------
If lidar_node.py is not running, every reply times out. compute() then
returns have_scan=False and NO dodge - the caller must treat "no scan" as
"no avoidance available", not as "path clear". follow_operator.py and
go_to_room.py both gate their avoidance on adj.have_scan for this reason.
"""

import json
import math
import socket
import time

import cv2
import numpy as np

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47001
DEFAULT_TIMEOUT = 0.08     # generous at 15 Hz lidar / 20 Hz node compute
STALE_AFTER = 1.0          # a reply older than this is treated as no scan

# debug_view() coloring thresholds - mirror lidar_node.py's BLOCK_DIST/DODGE_DIST
VIS_BLOCK_DIST = 0.45
VIS_DODGE_DIST = 0.90


class Adjust:
    """Same field names the old LidarAvoider returned, so callers are unchanged."""
    __slots__ = ("block_forward", "lat", "fwd_scale", "status", "have_scan", "pts")

    def __init__(self):
        self.block_forward = False
        self.lat = 0.0
        self.fwd_scale = 1.0
        self.status = "no lidar node"
        self.have_scan = False
        self.pts = []


class LidarClient:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 timeout=DEFAULT_TIMEOUT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.latest = Adjust()
        self.fails = 0

    def compute(self, persons=()):
        """Ask the node for an avoidance decision, masking out `persons`."""
        payload = {"persons": [list(p) for p in persons]}
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
            data, _ = self.sock.recvfrom(65535)
            rep = json.loads(data.decode("utf-8"))
        except (socket.timeout, OSError, ValueError):
            self.fails += 1
            # Do NOT keep the last dodge alive on a dead link - fail to
            # "no scan, no dodge" and let the caller decide.
            self.latest.have_scan = False
            self.latest.block_forward = False
            self.latest.lat = 0.0
            self.latest.fwd_scale = 1.0
            self.latest.status = "lidar node unreachable"
            self.latest.pts = []
            return self.latest

        self.fails = 0
        fresh = (time.time() - float(rep.get("t", 0.0))) < STALE_AFTER
        self.latest.block_forward = bool(rep.get("block", False))
        self.latest.lat = float(rep.get("lat", 0.0))
        self.latest.fwd_scale = float(rep.get("scale", 1.0))
        self.latest.status = str(rep.get("status", ""))
        self.latest.have_scan = bool(rep.get("scan", False)) and fresh
        self.latest.pts = rep.get("pts", []) if self.latest.have_scan else []
        if not fresh:
            self.latest.status = "lidar data stale"
        return self.latest

    def debug_view(self, size=420, rng=3.5):
        """Top-down picture of what the built-in LiDAR saw, same visual
        style as DepthAvoider.debug_view() in depth_avoider.py: black
        background, range rings, robot dot at bottom center, obstacle
        points colored by how close they are, dodge arrow, status text."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (20, 20, 24)
        cx, cy = size // 2, size - 30
        scale = (size - 50) / rng

        for m in (1, 2, 3):
            if m <= rng:
                cv2.circle(img, (cx, cy), int(m * scale), (50, 50, 55), 1)
                cv2.putText(img, "%dm" % m, (cx + 4, cy - int(m * scale) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 95), 1)

        a = self.latest
        if a.have_scan and a.pts:
            pts = np.array(a.pts, dtype=np.float32)   # [[range, angle_deg], ...]
            r, ang = pts[:, 0], np.radians(pts[:, 1])
            x = r * np.cos(ang)     # +x forward
            y = r * np.sin(ang)     # +y left
            sx = (cx - y * scale).astype(np.int32)
            sy = (cy - x * scale).astype(np.int32)
            ok = (sx >= 0) & (sx < size) & (sy >= 0) & (sy < size)
            sx, sy, r2 = sx[ok], sy[ok], r[ok]
            near = r2 < VIS_BLOCK_DIST
            mid = (~near) & (r2 < VIS_DODGE_DIST)
            img[sy[~near & ~mid], sx[~near & ~mid]] = (70, 200, 90)   # green: seen
            img[sy[mid], sx[mid]] = (0, 200, 255)                     # yellow: dodge range
            img[sy[near], sx[near]] = (60, 60, 245)                   # red: too close

        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        if a.have_scan and abs(a.lat) > 0.01:
            dx = int(math.copysign(55, a.lat))
            cv2.arrowedLine(img, (cx, cy - 12), (cx - dx, cy - 12),
                            (0, 200, 255), 2, tipLength=0.35)

        col = (60, 60, 245) if a.block_forward else (0, 200, 255)
        cv2.putText(img, a.status, (10, size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        return img

    def alive(self, tries=3):
        """True if lidar_node.py answers at all (used by the pre-flight check)."""
        for _ in range(tries):
            adj = self.compute([])
            if adj.status != "lidar node unreachable":
                return True
            time.sleep(0.1)
        return False

    def shutdown(self):
        try:
            self.sock.close()
        except OSError:
            pass


# Backwards-compatible alias: older code imported LidarAvoider.
LidarAvoider = LidarClient
