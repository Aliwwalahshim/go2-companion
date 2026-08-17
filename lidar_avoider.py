"""
lidar_avoider.py - Isolated ROS 2 LiDAR obstacle avoidance (person-masked).

WHAT CHANGED (vs previous version)
----------------------------------
1. ACCUMULATION: a single /utlidar/cloud message is very sparse, so one
   message alone often misses obstacles ("does not recognize things").
   We now keep a rolling ~0.5 s buffer of points -> much denser scan.
1b. GROUND REMOVAL: the cloud is in the lidar's tilted sensor frame, so a
   fixed z-band keeps the FLOOR and the robot blocks itself forever
   ("BLOCK fwd C=0.35 n=2357"). We RANSAC-fit the floor plane every cycle
   and keep only points 0.08-0.60 m above it.
2. PERSON MASK: the camera sends EVERY visible person (distance + bearing
   + angular width). Each one is cut out of the scan, so the lidar never
   dodges humans - only real obstacles. Before, only the operator was
   masked and only in a fixed +/-15 deg cone.
3. PROGRESSIVE SLOW-DOWN: besides block/dodge, we return fwd_scale
   (1.0 = clear ... 0.0 = at BLOCK_DIST) so the robot brakes smoothly
   instead of driving full speed until 30 cm and slamming to a stop.
4. ROBUST DISTANCE: sector distance = 3rd-closest point, not the single
   closest -> one noise point no longer triggers a phantom dodge.
5. DODGE HYSTERESIS: once dodging left, it keeps dodging left unless the
   right side is CLEARLY freer -> no left/right flip-flopping (jerk).

SEAM
----
    avoider = LidarAvoider()
    adj = avoider.compute(persons)   # persons = [(dist_m, angle_deg, half_deg), ...]
    adj.block_forward   # True -> forward speed must be 0
    adj.fwd_scale       # 0..1 multiplier for forward speed
    adj.lat             # sideways dodge velocity (+ = left)
    adj.status, adj.have_scan
    avoider.shutdown()
"""
import multiprocessing as mp
import time

import numpy as np

# ---------------- TUNING ----------------
FWD_AXIS = "x"         # set "y" if the cloud's forward axis is y
Z_MIN = -0.20          # ignore floor points below this
Z_MAX = 0.40           # ignore points above robot body
MAX_RANGE = 3.0
MIN_RANGE = 0.40       # L1 lidar sees the robot's own head/body/legs + close
                       # range noise below ~0.4 m -> must be ignored or it
                       # permanently self-blocks

BLOCK_DIST = 0.45      # hard-stop forward motion inside this distance
DODGE_DIST = 0.90      # start side-stepping inside this distance
SLOW_DIST = 1.30       # start scaling forward speed down inside this
DODGE_SPEED = 0.35     # max sideways dodge speed (m/s)
SIDE_CLEAR = 0.60      # a side must be at least this open before dodging into
                       # it (never side-step into a wall)

PERSON_BAND = 0.40     # +/- metres around each person's distance to mask out
CENTER_HALF_DEG = 25
SIDE_HALF_DEG = 70

ACCUM_SEC = 0.5        # rolling window of cloud messages to accumulate
MAX_ACCUM_PTS = 20000
KTH_NEAREST = 5        # sector distance = k-th closest point (noise rejection)

# --- ground removal ---
# /utlidar/cloud is in the lidar's TILTED sensor frame, so a fixed z-band
# CANNOT separate the floor from obstacles - the floor ahead shows up inside
# the band and permanently blocks the robot. Instead we RANSAC-fit the floor
# plane every cycle and keep only points 0.08-0.60 m ABOVE it.
RANSAC_ITERS = 40
FIT_SAMPLE = 6000      # max points used for the plane fit
GROUND_EPS = 0.06      # inlier distance (m) to count as floor
PLANE_MIN_FRAC = 0.20  # fit must explain at least this fraction of points
SENSOR_H_MIN = 0.10    # plausible sensor height above the fitted floor;
SENSOR_H_MAX = 0.80    # outside this the "plane" is probably a wall -> reject
OBST_H_MIN = 0.08      # obstacle band above the floor (below = floor/noise,
OBST_H_MAX = 0.60      # above = doorframes/ceiling the robot fits under)


def _clampf(v, lo, hi):
    return max(lo, min(hi, v))


class _Adjust:
    __slots__ = ("block_forward", "lat", "fwd_scale", "status", "have_scan")
    def __init__(self):
        self.block_forward = False
        self.lat = 0.0
        self.fwd_scale = 1.0
        self.status = "no lidar"
        self.have_scan = False


class LidarAvoider:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.parent_conn, self.child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_run_ros_node, args=(self.child_conn,), daemon=True)
        self.process.start()
        self.latest = _Adjust()

    def compute(self, persons):
        """persons: list of (distance_m, angle_deg, half_width_deg), one per
        visible person (INCLUDING the operator). All are masked from the scan."""
        try:
            self.parent_conn.send(("persons", list(persons)))
        except Exception:
            pass
        while self.parent_conn.poll():
            try:
                msg = self.parent_conn.recv()
                (self.latest.block_forward, self.latest.lat,
                 self.latest.fwd_scale, self.latest.status,
                 self.latest.have_scan) = msg
            except EOFError:
                break
        return self.latest

    def shutdown(self):
        try:
            self.parent_conn.send("quit")
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
        except Exception:
            pass


def _run_ros_node(conn):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    import numpy as np
    from collections import deque

    class LidarSub(Node):
        def __init__(self):
            super().__init__("go2_lidar_avoider")
            self.clouds = deque()   # (timestamp, Nx3 array)
            self.create_subscription(PointCloud2, "/utlidar/cloud", self.cb, 5)

        def cb(self, msg):
            try:
                arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
                step = msg.point_step // 4
                if step >= 3:
                    self.clouds.append((time.time(), arr.reshape(-1, step)[:, :3]))
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

    def robust_dist(sel_r):
        """k-th closest range in a sector; 9.9 if too few points (= clear)."""
        if sel_r.size < KTH_NEAREST:
            return 9.9
        return float(np.partition(sel_r, KTH_NEAREST - 1)[KTH_NEAREST - 1])

    def fit_ground(pts):
        """RANSAC-fit the floor plane. Returns (normal, d) with the normal
        oriented so height = p . normal + d is positive above the floor,
        or None if no plausible floor plane is found."""
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

    rclpy.init(args=None)
    node = LidarSub()

    persons = []       # [(dist, angle_deg, half_deg), ...] from the camera
    dodge_dir = 0.0    # -1 right, 0 none, +1 left (hysteresis memory)
    last_plane = None  # last good floor fit, reused when a fit fails

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)

        while conn.poll():
            try:
                data = conn.recv()
                if data == "quit":
                    node.destroy_node()
                    rclpy.shutdown()
                    return
                if isinstance(data, tuple) and data[0] == "persons":
                    persons = data[1]
            except EOFError:
                return

        pts = node.points()
        if pts is None or pts.shape[0] == 0:
            conn.send((False, 0.0, 1.0, "no lidar", False))
            continue

        x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
        if FWD_AXIS == "y":
            x, y = y, x

        rng = np.sqrt(x * x + y * y)
        ang = np.degrees(np.arctan2(y, x))

        # remove the floor: height above the RANSAC-fitted floor plane.
        # Falls back to the fixed z-band only if no floor was ever fitted.
        plane = fit_ground(pts)
        if plane is not None:
            last_plane = plane
        if last_plane is not None:
            nvec, doff = last_plane
            hgt = pts @ nvec + doff
            zmask = (hgt > OBST_H_MIN) & (hgt < OBST_H_MAX)
            mode = "g"      # ground-plane mode (good)
        else:
            zmask = (z > Z_MIN) & (z < Z_MAX)
            mode = "z"      # fallback z-band (floor may leak through!)

        m = zmask & (rng > MIN_RANGE) & (rng < MAX_RANGE) & (x > 0.0)

        # cut EVERY camera-detected person out of the scan so the lidar
        # only reacts to real obstacles, never humans
        for (pd, pang, phalf) in persons:
            band = (rng > pd - PERSON_BAND) & (rng < pd + PERSON_BAND)
            cone = np.abs(ang - pang) < phalf
            m &= ~(band & cone)

        if not np.any(m):
            conn.send((False, 0.0, 1.0, "clear n=0 [%s]" % mode, True))
            continue

        a = ang[m]; r = rng[m]

        def sector(lo, hi):
            sel = (a >= lo) & (a < hi)
            return robust_dist(r[sel])

        center = sector(-CENTER_HALF_DEG, CENTER_HALF_DEG)
        left = sector(CENTER_HALF_DEG, SIDE_HALF_DEG)
        right = sector(-SIDE_HALF_DEG, -CENTER_HALF_DEG)

        block = center < BLOCK_DIST
        fwd_scale = _clampf((center - BLOCK_DIST) / (SLOW_DIST - BLOCK_DIST), 0.0, 1.0)
        lat = 0.0
        status = "clear"

        if center < DODGE_DIST:
            # only dodge into a side that is actually OPEN; keep the current
            # dodge side while it stays open (hysteresis, no flip-flopping)
            can_left = left >= SIDE_CLEAR
            can_right = right >= SIDE_CLEAR
            if dodge_dir > 0 and can_left:
                want = 1.0
            elif dodge_dir < 0 and can_right:
                want = -1.0
            elif can_left and (left >= right or not can_right):
                want = 1.0
            elif can_right:
                want = -1.0
            else:
                want = 0.0          # boxed in: stand still, don't hit a wall
            dodge_dir = want
            if want != 0.0:
                strength = _clampf((DODGE_DIST - center) / (DODGE_DIST - BLOCK_DIST), 0.3, 1.0)
                lat = want * DODGE_SPEED * strength
                status = "dodge %s (C=%.2f)" % ("LEFT" if want > 0 else "RIGHT", center)
            else:
                status = "TRAPPED (C=%.2f L=%.2f R=%.2f)" % (center, left, right)
        else:
            dodge_dir = 0.0
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

        try:
            conn.send((block, lat, fwd_scale, status, True))
        except BrokenPipeError:
            break
