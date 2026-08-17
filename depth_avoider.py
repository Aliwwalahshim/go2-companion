#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
depth_avoider.py  --  obstacle avoidance from the RealSense depth camera.

WHY THIS EXISTS
---------------
The Go2's built-in LiDAR avoidance ignores anything below roughly 0.3 m,
so it walks straight into boxes on the floor.  The depth camera sees those
perfectly.  This module turns the depth image into a 3D point cloud, throws
away the floor, and reports whether the path ahead is blocked and which way
to step around it.

No ROS, no extra process, no network.  It runs inside your follow script
using the depth frame you already have.

CAMERA FRAME (documented by Intel, so no axis guessing):
    +Z forward     +X right     +Y down

The floor level is estimated from the image itself every frame and smoothed
over time, so the camera does not need to be perfectly level and you do not
need to measure its mounting height.

USE:
    av = DepthAvoider(intrinsics, mirror=True)
    adj = av.compute(depth_np, exclude_boxes=[(x1, y1, x2, y2)])
    if adj.block_forward: fwd = 0
    else:                 fwd *= adj.fwd_scale
    lat = adj.lat
    cv2.imshow("Depth Avoid", av.debug_view())
"""

import cv2
import numpy as np


# ---------------- tuning ----------------
STEP = 4               # use every Nth pixel (4 -> ~25k points, fast)
Z_MIN, Z_MAX = 0.25, 4.0     # metres: ignore closer/further than this

OBST_MIN_H = 0.06      # metres above floor: below this is floor, not obstacle
OBST_MAX_H = 1.60      # above this is ceiling / doorframe, ignore

CORRIDOR_HALF = 0.28   # metres: half width of the robot's path
SIDE_HALF = 0.75       # metres: how far sideways we look for a gap

BLOCK_DIST = 0.70      # hard stop forward inside this
SLOW_DIST = 1.50       # start slowing here
DODGE_DIST = 1.80      # start stepping sideways here
SIDE_CLEAR = 0.90      # a side counts as open beyond this
DODGE_SPEED = 0.40     # m/s sideways

FWD_FLOOR = 0.45       # never slow below this unless truly blocked
MIN_POINTS = 25        # a sector needs this many points to count as real
BOX_PAD = 12           # pixels of padding around an excluded person box
FLOOR_EMA = 0.25       # smoothing on the floor-level estimate


class Adjust(object):
    """Same shape as the old lidar Adjust, so it drops straight in."""

    def __init__(self):
        self.have_scan = False
        self.block_forward = False
        self.fwd_scale = 1.0
        self.lat = 0.0
        self.status = "no depth"


def _clampf(v, lo, hi):
    return max(lo, min(hi, v))


class DepthAvoider(object):

    def __init__(self, intrinsics, mirror=True, depth_scale=0.001):
        """intrinsics: rs.intrinsics from the aligned depth stream profile."""
        self.w = intrinsics.width
        self.h = intrinsics.height
        self.mirror = mirror
        self.scale = depth_scale

        # precompute the deprojection grid once (this is the expensive bit)
        us = np.arange(0, self.w, STEP, dtype=np.float32)
        vs = np.arange(0, self.h, STEP, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        self.kx = (uu - intrinsics.ppx) / intrinsics.fx
        self.ky = (vv - intrinsics.ppy) / intrinsics.fy
        self.gh, self.gw = self.kx.shape

        self.floor_y = None        # estimated floor level in camera Y
        self.dodge_dir = 0         # hysteresis: -1 left, +1 right, 0 none
        self._dbg = None
        self._last = Adjust()

    # ------------------------------------------------------------------
    def compute(self, depth_np, exclude_boxes=()):
        a = Adjust()
        if depth_np is None:
            self._last = a
            return a

        # --- subsample and deproject to 3D (camera frame) ---
        d = depth_np[::STEP, ::STEP].astype(np.float32) * self.scale

        # --- blank out the operator so we never dodge the person ---
        for box in exclude_boxes:
            x1, y1, x2, y2 = box
            if self.mirror:
                x1, x2 = (self.w - 1) - x2, (self.w - 1) - x1
            x1 = max(0, int(x1) - BOX_PAD) // STEP
            x2 = min(self.w - 1, int(x2) + BOX_PAD) // STEP
            y1 = max(0, int(y1) - BOX_PAD) // STEP
            y2 = min(self.h - 1, int(y2) + BOX_PAD) // STEP
            if x2 > x1 and y2 > y1:
                d[y1:y2, x1:x2] = 0.0

        valid = (d > Z_MIN) & (d < Z_MAX)
        if valid.sum() < 200:
            a.status = "no depth"
            self._last = a
            self._dbg = None
            return a

        z = d[valid]
        x = self.kx[valid] * z
        y = self.ky[valid] * z          # +Y is DOWN

        # --- estimate the floor: the lowest stuff in view (largest Y) ---
        f = np.percentile(y, 92)
        self.floor_y = f if self.floor_y is None else \
            (FLOOR_EMA * f + (1.0 - FLOOR_EMA) * self.floor_y)

        height = self.floor_y - y        # metres above the floor
        obst = (height > OBST_MIN_H) & (height < OBST_MAX_H)

        ox, oz = x[obst], z[obst]
        a.have_scan = True

        # --- three sectors in front ---
        def nearest(mask):
            if mask.sum() < MIN_POINTS:
                return 99.0
            return float(np.percentile(oz[mask], 5))   # 5th pct rejects noise

        centre = nearest(np.abs(ox) < CORRIDOR_HALF)
        left = nearest((ox < -CORRIDOR_HALF) & (ox > -SIDE_HALF))
        right = nearest((ox > CORRIDOR_HALF) & (ox < SIDE_HALF))

        # --- decide ---
        a.block_forward = centre < BLOCK_DIST
        a.fwd_scale = _clampf((centre - BLOCK_DIST) / (SLOW_DIST - BLOCK_DIST),
                              0.0, 1.0)
        # keep making forward progress while stepping around something,
        # otherwise the robot just crabs sideways and never gets past it
        if not a.block_forward:
            a.fwd_scale = max(a.fwd_scale, FWD_FLOOR)

        if centre < DODGE_DIST:
            # pick the side with more room, and stay on it (hysteresis)
            if self.dodge_dir == 0:
                self.dodge_dir = 1 if right > left else -1
            else:
                chosen = right if self.dodge_dir > 0 else left
                other = left if self.dodge_dir > 0 else right
                if chosen < SIDE_CLEAR * 0.7 and other > chosen + 0.4:
                    self.dodge_dir = -self.dodge_dir

            strength = _clampf((DODGE_DIST - centre) /
                               (DODGE_DIST - BLOCK_DIST), 0.35, 1.0)
            a.lat = -self.dodge_dir * DODGE_SPEED * strength
            side = "RIGHT" if self.dodge_dir > 0 else "LEFT"
            a.status = ("BLOCK C=%.2f -> go %s" % (centre, side)
                        if a.block_forward else
                        "dodge %s C=%.2f" % (side, centre))
        else:
            self.dodge_dir = 0
            a.lat = 0.0
            a.status = ("slowing C=%.2f" % centre if a.fwd_scale < 1.0
                        else "clear C=%.2f" % centre)

        self._sectors = (centre, left, right)
        self._pts = (ox, oz)
        self._last = a
        self._dbg = None
        return a

    # ------------------------------------------------------------------
    def debug_view(self, size=420, rng=3.5):
        """Top-down picture of what the depth camera thinks is in the way."""
        if self._dbg is not None:
            return self._dbg

        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (20, 20, 24)
        cx, cy = size // 2, size - 30
        scale = (size - 50) / rng

        # range arcs
        for m in (1, 2, 3):
            if m <= rng:
                cv2.circle(img, (cx, cy), int(m * scale), (50, 50, 55), 1)
                cv2.putText(img, "%dm" % m, (cx + 4, cy - int(m * scale) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 95), 1)

        # the corridor the robot needs
        hw = int(CORRIDOR_HALF * scale)
        cv2.line(img, (cx - hw, cy), (cx - hw, 20), (70, 70, 80), 1)
        cv2.line(img, (cx + hw, cy), (cx + hw, 20), (70, 70, 80), 1)

        a = self._last
        if a.have_scan and hasattr(self, "_pts"):
            ox, oz = self._pts
            if len(ox):
                sx = (cx + ox * scale).astype(np.int32)
                sy = (cy - oz * scale).astype(np.int32)
                ok = (sx >= 0) & (sx < size) & (sy >= 0) & (sy < size)
                sx, sy, oz2 = sx[ok], sy[ok], oz[ok]
                near = oz2 < BLOCK_DIST
                img[sy[~near], sx[~near]] = (70, 200, 90)      # green: seen
                img[sy[near], sx[near]] = (60, 60, 245)        # red: too close

            c, l, r = self._sectors
            cv2.putText(img, "L %.2f   C %.2f   R %.2f"
                        % (min(l, 9.99), min(c, 9.99), min(r, 9.99)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)

        # robot + dodge arrow
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        if abs(a.lat) > 0.01:
            dx = int(np.sign(a.lat) * 55)
            cv2.arrowedLine(img, (cx, cy - 12), (cx - dx, cy - 12),
                            (0, 200, 255), 2, tipLength=0.35)

        col = (60, 60, 245) if a.block_forward else (0, 200, 255)
        cv2.putText(img, a.status, (10, size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if self.floor_y is not None:
            cv2.putText(img, "floor y=%.2f" % self.floor_y, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 130), 1)

        self._dbg = img
        return img
