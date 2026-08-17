"""
lidar_node.py - ROS 2 LiDAR obstacle-avoidance node, served over UDP.

RUN THIS IN A foxy-SOURCED SHELL. It imports rclpy.
Never import it from the SDK/control process - rclpy and the Unitree SDK
collide on CycloneDDS inside one process. The control side talks to this
node through lidar_client.py (plain UDP, no rclpy).

    Terminal A (foxy-sourced):  python3 lidar_node.py
    Terminal C (NOT sourced):   python3 claude_go2_master.py eth0

ALGORITHM
---------
Ported from the tested lidar_avoider.py. Everything that was learned the
hard way is kept:

1. ACCUMULATION - one point cloud message is very sparse, so a single
   message misses obstacles. We keep a rolling ~0.5 s buffer.
2. GROUND REMOVAL - the cloud is in the lidar's TILTED sensor frame, so a
   fixed z-band keeps the FLOOR and the robot blocks itself forever. We
   RANSAC-fit the floor plane every cycle and keep only points 0.08-0.60 m
   above it.
3. PERSON MASK - every person the camera reports (distance + bearing +
   angular width) is cut out of the scan, so the lidar never dodges humans.
4. PROGRESSIVE SLOW-DOWN - fwd_scale (1.0 clear .. 0.0 at BLOCK_DIST) so
   the robot brakes smoothly instead of slamming to a stop.
5. ROBUST DISTANCE - sector distance is the k-th closest point, not the
   single closest, so one noise point cannot trigger a phantom dodge.
6. DODGE HYSTERESIS - once dodging left it keeps dodging left unless the
   right is clearly freer, so there is no left/right flip-flop.

WIRE PROTOCOL (JSON over UDP, request/reply)
--------------------------------------------
    request : {"persons": [[dist_m, angle_deg, half_width_deg], ...]}
    reply   : {"block": bool,      # forward speed must be 0
               "lat": float,       # sideways dodge velocity (+ = left)
               "scale": float,     # 0..1 forward-speed multiplier
               "status": str,      # human-readable
               "scan": bool,       # False = no lidar data yet
               "t": float}         # epoch seconds this was computed
"""

import argparse
import json
import socket
import sys
import time
from collections import deque

import numpy as np

# rclpy / sensor_msgs are imported inside main() on purpose. Everything
# above main() is plain NumPy, so the avoidance maths can be imported and
# unit-tested on a machine with no ROS 2 installed.

# ---------------- NETWORK ----------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47001
# Confirmed via `ros2 topic info /utlidar/cloud` on this unit (Publisher
# count: 1, type sensor_msgs/msg/PointCloud2). An earlier "update" to
# /rslidar_points (for a RoboSense RSHELIOS) does not exist on this robot's
# ROS2 graph at all - /utlidar/cloud is what's actually live. See
# lidar_view.py's docstring for how this was confirmed.
DEFAULT_TOPIC = "/utlidar/cloud"

# ---------------- TUNING ----------------
FWD_AXIS = "x"         # set "y" if the cloud's forward axis is y
Z_MIN = -0.20          # fallback z-band only (used if no floor plane found)
Z_MAX = 0.40
MAX_RANGE = 3.0
MIN_RANGE = 0.40       # the lidar sees the robot's own head/body/legs and
                       # close-range noise below ~0.4 m -> must be ignored or
                       # the robot permanently self-blocks

BLOCK_DIST = 0.45      # hard-stop forward motion inside this distance
DODGE_DIST = 0.90      # start side-stepping inside this distance
SLOW_DIST = 1.30       # start scaling forward speed down inside this
DODGE_SPEED = 0.35     # max sideways dodge speed (m/s)
SIDE_CLEAR = 0.60      # a side must be this open before dodging into it
                       # (never side-step into a wall)

PERSON_BAND = 0.40     # +/- metres around each person's distance to mask out
CENTER_HALF_DEG = 25
SIDE_HALF_DEG = 70

ACCUM_SEC = 0.5        # rolling window of cloud messages to accumulate
MAX_ACCUM_PTS = 20000
KTH_NEAREST = 5        # sector distance = k-th closest point
MAX_DBG_PTS = 150      # obstacle points sent back for visualization (UDP size)

# --- ground removal (RANSAC) ---
RANSAC_ITERS = 40
FIT_SAMPLE = 6000      # max points used for the plane fit
GROUND_EPS = 0.06      # inlier distance (m) to count as floor
PLANE_MIN_FRAC = 0.20  # fit must explain at least this fraction of points
SENSOR_H_MIN = 0.10    # plausible sensor height above the fitted floor;
SENSOR_H_MAX = 0.80    # outside this the "plane" is probably a wall -> reject
OBST_H_MIN = 0.08      # obstacle band above the floor (below = floor/noise,
OBST_H_MAX = 0.60      # above = doorframes/ceiling the robot fits under)

COMPUTE_HZ = 20.0
STALE_AFTER = 1.0      # seconds without a cloud msg before we report no scan


def _clampf(v, lo, hi):
    return max(lo, min(hi, v))


def downsample_pts(r, a):
    """(range, angle_deg) arrays -> capped [[range, angle], ...] list for the
    UDP reply, so a client can draw a top-down picture of what the LiDAR
    saw without needing rclpy itself (see lidar_client.py's debug_view)."""
    n = r.size
    if n == 0:
        return []
    if n > MAX_DBG_PTS:
        idx = np.linspace(0, n - 1, MAX_DBG_PTS).astype(np.int32)
        r, a = r[idx], a[idx]
    return np.round(np.stack([r, a], axis=1), 2).tolist()


def robust_dist(sel_r):
    """k-th closest range in a sector; 9.9 if too few points (= clear)."""
    if sel_r.size < KTH_NEAREST:
        return 9.9
    return float(np.partition(sel_r, KTH_NEAREST - 1)[KTH_NEAREST - 1])


def fit_ground(pts):
    """RANSAC-fit the floor plane. Returns (normal, d) with the normal
    oriented so height = p . normal + d is positive above the floor, or
    None if no plausible floor plane is found."""
    n_pts = pts.shape[0]
    if n_pts < 200:
        return None
    if n_pts > FIT_SAMPLE:
        pts = pts[np.random.choice(n_pts, FIT_SAMPLE, replace=False)]
    best_n, best_d, best_cnt = None, 0.0, 0
    for _ in range(RANSAC_ITERS):
        p0, p1, p2 = pts[np.random.choice(pts.shape[0], 3, replace=False)]
        nv = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nv)
        if norm < 1e-6:
            continue
        nv /= norm
        dv = -float(np.dot(nv, p0))
        cnt = int(np.count_nonzero(np.abs(pts @ nv + dv) < GROUND_EPS))
        if cnt > best_cnt:
            best_n, best_d, best_cnt = nv, dv, cnt
    if best_n is None or best_cnt < PLANE_MIN_FRAC * pts.shape[0]:
        return None
    # orient so the sensor origin (0,0,0) sits ABOVE the plane
    if best_d < 0:
        best_n, best_d = -best_n, -best_d
    # sanity: the sensor must be at a plausible height above a FLOOR;
    # otherwise the dominant plane is a wall -> reject
    if not (SENSOR_H_MIN < best_d < SENSOR_H_MAX):
        return None
    return best_n, best_d


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
        """Accumulated points from the last ACCUM_SEC seconds."""
        now = time.time()
        while self.clouds and now - self.clouds[0][0] > ACCUM_SEC:
            self.clouds.popleft()
        if not self.clouds:
            return None
        pts = np.concatenate([c[1] for c in self.clouds], axis=0)
        if pts.shape[0] > MAX_ACCUM_PTS:
            pts = pts[-MAX_ACCUM_PTS:]
        return pts


class Avoidance:
    """Holds the rolling avoidance state between compute() calls."""

    def __init__(self):
        self.dodge_dir = 0.0    # -1 right, 0 none, +1 left (hysteresis memory)
        self.last_plane = None  # last good floor fit, reused when a fit fails

    def compute(self, pts, persons):
        """pts: Nx3 accumulated cloud. persons: [(dist, ang_deg, half_deg)].
        Returns the reply dict."""
        if pts is None or pts.shape[0] == 0:
            return dict(block=False, lat=0.0, scale=1.0,
                        status="no lidar", scan=False, t=time.time(), pts=[])

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        if FWD_AXIS == "y":
            x, y = y, x

        rng = np.sqrt(x * x + y * y)
        ang = np.degrees(np.arctan2(y, x))

        # remove the floor: height above the RANSAC-fitted floor plane.
        # Falls back to the fixed z-band only if no floor was ever fitted.
        plane = fit_ground(pts)
        if plane is not None:
            self.last_plane = plane
        if self.last_plane is not None:
            nvec, doff = self.last_plane
            hgt = pts @ nvec + doff
            zmask = (hgt > OBST_H_MIN) & (hgt < OBST_H_MAX)
            mode = "g"      # ground-plane mode (good)
        else:
            zmask = (z > Z_MIN) & (z < Z_MAX)
            mode = "z"      # fallback z-band (floor may leak through!)

        m = zmask & (rng > MIN_RANGE) & (rng < MAX_RANGE) & (x > 0.0)

        # cut EVERY camera-detected person out of the scan so the lidar
        # only reacts to real obstacles, never humans
        for person in persons:
            try:
                pd, pang, phalf = float(person[0]), float(person[1]), float(person[2])
            except (TypeError, ValueError, IndexError):
                continue
            band = (rng > pd - PERSON_BAND) & (rng < pd + PERSON_BAND)
            cone = np.abs(ang - pang) < phalf
            m &= ~(band & cone)

        if not np.any(m):
            return dict(block=False, lat=0.0, scale=1.0,
                        status="clear n=0 [%s]" % mode, scan=True, t=time.time(), pts=[])

        a = ang[m]
        r = rng[m]

        def sector(lo, hi):
            sel = (a >= lo) & (a < hi)
            return robust_dist(r[sel])

        center = sector(-CENTER_HALF_DEG, CENTER_HALF_DEG)
        left = sector(CENTER_HALF_DEG, SIDE_HALF_DEG)
        right = sector(-SIDE_HALF_DEG, -CENTER_HALF_DEG)

        block = bool(center < BLOCK_DIST)
        fwd_scale = _clampf((center - BLOCK_DIST) / (SLOW_DIST - BLOCK_DIST), 0.0, 1.0)
        lat = 0.0
        status = "clear"

        if center < DODGE_DIST:
            # only dodge into a side that is actually OPEN; keep the current
            # dodge side while it stays open (hysteresis, no flip-flopping)
            can_left = left >= SIDE_CLEAR
            can_right = right >= SIDE_CLEAR
            if self.dodge_dir > 0 and can_left:
                want = 1.0
            elif self.dodge_dir < 0 and can_right:
                want = -1.0
            elif can_left and (left >= right or not can_right):
                want = 1.0
            elif can_right:
                want = -1.0
            else:
                want = 0.0          # boxed in: stand still, don't hit a wall
            self.dodge_dir = want
            if want != 0.0:
                strength = _clampf((DODGE_DIST - center) / (DODGE_DIST - BLOCK_DIST),
                                   0.3, 1.0)
                lat = want * DODGE_SPEED * strength
                status = "dodge %s (C=%.2f)" % ("LEFT" if want > 0 else "RIGHT", center)
            else:
                status = "TRAPPED (C=%.2f L=%.2f R=%.2f)" % (center, left, right)
        else:
            self.dodge_dir = 0.0
            if left < DODGE_DIST and right >= SIDE_CLEAR:
                lat = -DODGE_SPEED * 0.7
                status = "push RIGHT (L=%.2f)" % left
            elif right < DODGE_DIST and left >= SIDE_CLEAR:
                lat = DODGE_SPEED * 0.7
                status = "push LEFT (R=%.2f)" % right

        if status == "clear" and fwd_scale < 1.0:
            status = "slowing (C=%.2f)" % center
        if block:
            status = "BLOCK fwd (C=%.2f)" % center
        # n = obstacle points left after floor+person masking (big = real
        # obstacle, tiny = noise); [g]=ground-plane mode, [z]=z-band fallback
        status += " n=%d [%s]" % (a.size, mode)

        return dict(block=block, lat=float(lat), scale=float(fwd_scale),
                    status=status, scan=True, t=time.time(),
                    pts=downsample_pts(r, a))


def main():
    ap = argparse.ArgumentParser(description="Go2 LiDAR avoidance UDP server")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--topic", default=DEFAULT_TOPIC,
                    help="PointCloud2 topic (default %s)" % DEFAULT_TOPIC)
    args = ap.parse_args()

    # ROS 2 imports live here so the maths above stays testable without ROS.
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        print("ERROR: could not import ROS 2 (%s)." % exc)
        print("This node MUST run in a foxy-sourced shell, e.g.:")
        print("    source /opt/ros/foxy/setup.bash && python3 lidar_node.py")
        return 1

    class LidarSub(Node):
        def __init__(self, topic, buf):
            super().__init__("go2_lidar_node")
            self.buf = buf
            self.create_subscription(PointCloud2, topic, self.cb, 5)

        def cb(self, msg):
            self.buf.add_raw(msg.data, msg.point_step)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((args.host, args.port))

    rclpy.init(args=None)
    buf = CloudBuffer()
    node = LidarSub(args.topic, buf)
    avoid = Avoidance()

    latest = dict(block=False, lat=0.0, scale=1.0,
                  status="starting", scan=False, t=time.time())
    persons = []
    last_compute = 0.0
    last_report = 0.0
    period = 1.0 / COMPUTE_HZ

    print("[lidar_node] listening on %s:%d, subscribed to %s"
          % (args.host, args.port, args.topic))
    print("[lidar_node] waiting for point clouds ...")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)

            # ---- drain UDP requests, reply with the latest computation ----
            while True:
                try:
                    data, addr = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    req = json.loads(data.decode("utf-8"))
                    if isinstance(req, dict) and isinstance(req.get("persons"), list):
                        persons = req["persons"]
                except Exception:
                    pass    # malformed request: still reply with current state
                try:
                    sock.sendto(json.dumps(latest).encode("utf-8"), addr)
                except OSError:
                    pass

            # ---- recompute at a fixed rate ----
            now = time.time()
            if now - last_compute >= period:
                last_compute = now
                latest = avoid.compute(buf.points(), persons)

            # ---- heartbeat so a silent terminal is obviously wrong ----
            if now - last_report >= 5.0:
                last_report = now
                print("[lidar_node] msgs=%d  %s" % (buf.msg_count, latest["status"]))
    except KeyboardInterrupt:
        print("\n[lidar_node] interrupted.")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()
        print("[lidar_node] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
