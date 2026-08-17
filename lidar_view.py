#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
lidar_view.py - live 3D-ish colorized view of the Go2's built-in LiDAR
point cloud (orbit camera, height coloring), similar to the point cloud
screen in the Unitree phone app.

RUN THIS IN A foxy-SOURCED SHELL. It imports rclpy directly, same as
lidar_node.py - never import it from the SDK/control process (rclpy and
the Unitree SDK collide on CycloneDDS inside one process). This script
sends nothing to the robot at all: no SportClient, no MotionSwitcherClient,
just a plain ROS 2 subscription to the LiDAR's own point cloud topic. Safe
to run any time, on its own or alongside anything else.

RMW_IMPLEMENTATION must be set to rmw_cyclonedds_cpp - this box's default
RMW (FastRTPS, if the var is left unset) reliably crashes with
"bad_alloc" on this Jetson. cyclonedds_ws is already built here for
exactly this reason:

    source /opt/ros/foxy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 lidar_view.py

Also: the LiDAR's own broadcast switch must be ON (rt/utlidar/switch,
plain SDK topic, OFF by default) or /utlidar/cloud never publishes at
all. climb_stairs.py / follow_depth_avoid.py don't touch this switch, so
if nothing's shown, send ON once from any eth0-connected SDK session,
e.g. go2_utlidar_switch.py in example/go2/high_level (edit its "OFF"
call to "ON", or send it inline) - this is a sensor-only toggle, not a
motion command.

The real topic name here is /utlidar/cloud - confirmed via
`ros2 topic info /utlidar/cloud` on this unit (Publisher count: 1,
type sensor_msgs/msg/PointCloud2). lidar_node.py in this folder was
"updated" to /rslidar_points at some point and disagrees; on this unit
that appears to be wrong - /rslidar_points isn't in `ros2 topic list`
at all. lidar_avoider.py's original /utlidar/cloud was right. That
mismatch lives in lidar_node.py, not touched here since
follow_depth_avoid.py runs with USE_BUILTIN_AVOID = False and doesn't
depend on it.

VIEW / CONTROLS
----------------
Big panel: a real 3D perspective (orbit camera), points colored by
height (blue=low ... red=high). Small inset, top-left: a flat top-down
minimap for orientation while orbiting.

    left-drag    orbit (yaw / pitch)
    mouse wheel  zoom in/out
    r            reset the view
    q / ESC      quit (Ctrl+C also works)

Points are accumulated over a short rolling window (ACCUM_SEC) so the
picture stays full between sparse scans - the same trick lidar_node.py
uses for obstacle avoidance.

Optional args:
    --topic   PointCloud2 topic (default /utlidar/cloud)
    --range   max range accumulated/shown, metres (default 5.0)
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

DEFAULT_TOPIC = "/utlidar/cloud"
ACCUM_SEC = 0.4
MAX_ACCUM_PTS = 40000

MAIN_W = 800
MAIN_H = 600
LEGEND_W = 50
MINI_SIZE = 160
Z_MIN = -0.4        # colour-scale floor (metres, sensor frame)
Z_MAX = 0.8         # colour-scale ceiling

DILATE_NEAR = np.ones((5, 5), np.uint8)
DILATE_FAR = np.ones((3, 3), np.uint8)
NEAR_CUTOFF_M = 2.5   # camera-space depth split for point-size tiering

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- default orbit camera state ---
DEFAULT_YAW = 200.0     # degrees, around world Z, measured from +x
DEFAULT_PITCH = 28.0    # degrees above the horizon
DEFAULT_DIST = 5.0      # metres from the look-at point
LOOK_AT = np.array([1.5, 0.0, 0.1], dtype=np.float32)
FOV_DEG = 60.0

# precompute the jet colormap lookup table once
_JET_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET
).reshape(256, 3)


def height_to_color(z):
    """z: Nx float array -> Nx3 uint8 BGR via the jet colormap, clipped
    to [Z_MIN, Z_MAX]."""
    t = np.clip((z - Z_MIN) / (Z_MAX - Z_MIN), 0.0, 1.0)
    idx = (t * 255).astype(np.uint8)
    return _JET_LUT[idx]


class CloudBuffer:
    """Rolling accumulation of point clouds. Pure NumPy, no ROS types."""

    def __init__(self):
        self.clouds = deque()      # (timestamp, Nx3 array)
        self.msg_count = 0

    def add_raw(self, raw_bytes, point_step):
        """Parse a PointCloud2 data blob directly with NumPy (fields x,y,z
        float32). Avoids sensor_msgs_py, which is slow and often missing."""
        try:
            arr = np.frombuffer(bytes(raw_bytes), dtype=np.float32)
            step = point_step // 4
            if step >= 3:
                self.clouds.append((time.time(), arr.reshape(-1, step)[:, :3]))
                self.msg_count += 1
        except Exception:
            pass

    def points(self):
        now = time.time()
        while self.clouds and now - self.clouds[0][0] > ACCUM_SEC:
            self.clouds.popleft()
        if not self.clouds:
            return None
        pts = np.concatenate([c[1] for c in self.clouds], axis=0)
        if pts.shape[0] > MAX_ACCUM_PTS:
            pts = pts[-MAX_ACCUM_PTS:]
        return pts


class OrbitCam:
    """Mouse-driven orbit camera: drag to rotate, wheel to zoom."""

    def __init__(self):
        self.reset()
        self.dragging = False
        self.last_xy = (0, 0)

    def reset(self):
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.dist = DEFAULT_DIST

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            self.last_xy = (x, y)
            self.yaw = (self.yaw + dx * 0.4) % 360.0
            self.pitch = max(5.0, min(85.0, self.pitch - dy * 0.4))
        elif event == cv2.EVENT_MOUSEWHEEL:
            step = 1 if cv2.getMouseWheelDelta(flags) > 0 else -1
            self.dist = max(1.0, min(15.0, self.dist * (0.9 ** step)))

    def basis(self):
        """Camera position + right/up/forward unit vectors (world frame:
        x forward, y left, z up)."""
        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        offset = self.dist * np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float32)
        cam_pos = LOOK_AT + offset
        forward = LOOK_AT - cam_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return cam_pos, right, up, forward


def project(pts, cam_pos, right, up, forward, w, h, fov_deg=FOV_DEG):
    """World Nx3 -> (screen Nx2 int32, depth Nx float, mask bool N)."""
    rel = pts - cam_pos
    cx = rel @ right
    cy = rel @ up
    cz = rel @ forward
    mask = cz > 0.15
    focal = (w * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    sx = cx[mask] / cz[mask] * focal + w * 0.5
    sy = -cy[mask] / cz[mask] * focal + h * 0.5
    screen = np.stack([sx, sy], axis=1)
    return screen, cz[mask], mask


def draw_grid_and_axes(img, cam):
    cam_pos, right, up, forward = cam.basis()
    h, wd = img.shape[:2]

    def seg(p0, p1, color, thickness=1):
        pts = np.array([p0, p1], dtype=np.float32)
        screen, depth, mask = project(pts, cam_pos, right, up, forward, wd, h)
        if screen.shape[0] < 2:
            return
        p0s = tuple(np.round(screen[0]).astype(int))
        p1s = tuple(np.round(screen[1]).astype(int))
        cv2.line(img, p0s, p1s, color, thickness, cv2.LINE_AA)

    # floor grid, 1 m spacing, +/-5 m box centered on the robot
    for i in range(-5, 6):
        seg((float(i), -5.0, 0.0), (float(i), 5.0, 0.0), (35, 35, 35))
        seg((-5.0, float(i), 0.0), (5.0, float(i), 0.0), (35, 35, 35))

    # robot marker + forward axis (red = forward/+x, green = left/+y)
    seg((0, 0, 0), (0.6, 0, 0), (0, 0, 220), 2)
    seg((0, 0, 0), (0, 0.6, 0), (0, 220, 0), 2)
    seg((0, 0, 0), (0, 0, 0.6), (220, 0, 0), 2)


def draw_points_3d(img, pts, cam):
    if pts is None or pts.shape[0] == 0:
        return
    cam_pos, right, up, forward = cam.basis()
    h, wd = img.shape[:2]
    screen, depth, mask = project(pts, cam_pos, right, up, forward, wd, h)
    if screen.shape[0] == 0:
        return
    zc = pts[mask][:, 2]
    px = np.round(screen[:, 0]).astype(np.int32)
    py = np.round(screen[:, 1]).astype(np.int32)
    inb = (px >= 0) & (px < wd) & (py >= 0) & (py < h)
    px, py, zc, depth = px[inb], py[inb], zc[inb], depth[inb]
    if px.size == 0:
        return

    colors = height_to_color(zc)
    near = depth < NEAR_CUTOFF_M

    layer = np.zeros_like(img)
    if np.any(~near):
        layer[py[~near], px[~near]] = colors[~near]
        layer = cv2.dilate(layer, DILATE_FAR)
    if np.any(near):
        near_layer = np.zeros_like(img)
        near_layer[py[near], px[near]] = colors[near]
        near_layer = cv2.dilate(near_layer, DILATE_NEAR)
        layer = np.maximum(layer, near_layer)

    idx = np.maximum.reduce([layer[..., 0], layer[..., 1], layer[..., 2]]) > 0
    img[idx] = layer[idx]


def draw_minimap(pts, max_range):
    """Small flat top-down view (robot at bottom center) for orientation."""
    img = np.zeros((MINI_SIZE, MINI_SIZE, 3), np.uint8)
    ppm = MINI_SIZE / (2.0 * max_range)
    cx, cy = MINI_SIZE // 2, MINI_SIZE - 10
    cv2.circle(img, (cx, cy), int(1 * ppm), (40, 40, 40), 1)
    cv2.circle(img, (cx, cy), int(min(max_range, 3) * ppm), (40, 40, 40), 1)
    cv2.line(img, (cx, 0), (cx, MINI_SIZE), (30, 30, 30), 1)

    if pts is not None and pts.shape[0] > 0:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        rng = np.sqrt(x * x + y * y)
        m = rng < max_range
        x, y, z = x[m], y[m], z[m]
        px = (cx - y * ppm).astype(np.int32)
        py = (cy - x * ppm).astype(np.int32)
        inb = (px >= 0) & (px < MINI_SIZE) & (py >= 0) & (py < MINI_SIZE)
        px, py, zc = px[inb], py[inb], z[inb]
        if px.size:
            img[py, px] = height_to_color(zc)

    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (MINI_SIZE - 1, MINI_SIZE - 1), (90, 90, 90), 1)
    cv2.putText(img, "top-down", (4, 12), FONT, 0.35, (180, 180, 180), 1)
    return img


def legend(height=MAIN_H, width=LEGEND_W):
    grad = np.linspace(255, 0, height).astype(np.uint8).reshape(-1, 1)
    grad = np.repeat(grad, width, axis=1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    cv2.putText(bar, "%.1fm" % Z_MAX, (2, 14), FONT, 0.38, (255, 255, 255), 1)
    cv2.putText(bar, "%.1fm" % Z_MIN, (2, height - 6), FONT, 0.38,
                (255, 255, 255), 1)
    return bar


def main():
    ap = argparse.ArgumentParser(
        description="Live 3D-ish colorized view of the Go2's built-in LiDAR point cloud")
    ap.add_argument("--topic", default=DEFAULT_TOPIC,
                    help="PointCloud2 topic (default %s)" % DEFAULT_TOPIC)
    ap.add_argument("--range", type=float, default=5.0, dest="max_range",
                    help="max range accumulated/shown, metres (default 5.0)")
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        print("ERROR: could not import ROS 2 (%s)." % exc)
        print("This must run in a foxy-sourced shell, e.g.:")
        print("    source /opt/ros/foxy/setup.bash")
        print("    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp")
        print("    python3 lidar_view.py")
        return 1

    buf = CloudBuffer()

    class LidarSub(Node):
        def __init__(self):
            super().__init__("go2_lidar_view")
            self.create_subscription(PointCloud2, args.topic, self.cb, 5)

        def cb(self, msg):
            buf.add_raw(msg.data, msg.point_step)

    rclpy.init(args=None)
    node = LidarSub()

    print("[lidar_view] subscribed to %s. Sends nothing to the robot - "
          "view only." % args.topic)
    print("[lidar_view] window open. left-drag=orbit, wheel=zoom, "
          "r=reset view, q/ESC=quit.")

    cam = OrbitCam()
    win = "Go2 LiDAR view (3D)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cam.on_mouse)

    leg = legend()
    last_count = 0
    last_rate_t = time.time()
    rate = 0.0

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)

            pts = buf.points()

            canvas = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
            draw_grid_and_axes(canvas, cam)
            draw_points_3d(canvas, pts, cam)

            mini = draw_minimap(pts, args.max_range)
            canvas[10:10 + MINI_SIZE, 10:10 + MINI_SIZE] = mini

            full = np.hstack([canvas, leg])

            now = time.time()
            if now - last_rate_t >= 1.0:
                rate = (buf.msg_count - last_count) / (now - last_rate_t)
                last_count = buf.msg_count
                last_rate_t = now
            n_pts = 0 if pts is None else pts.shape[0]
            cv2.putText(full, "msgs/s: %.1f  pts: %d  yaw:%.0f pitch:%.0f "
                        "dist:%.1fm  (drag=orbit wheel=zoom r=reset q=quit)"
                        % (rate, n_pts, cam.yaw, cam.pitch, cam.dist),
                        (8, full.shape[0] - 10), FONT, 0.42, (0, 255, 255), 1)

            cv2.imshow(win, full)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            elif key == ord("r"):
                cam.reset()
    except KeyboardInterrupt:
        print("\n[lidar_view] interrupted.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("[lidar_view] stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
