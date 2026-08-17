"""
go_to_room.py - Drive to a recorded room pose in a straight line, with
                heading control and LiDAR dodging.

    python3 go_to_room.py kitchen [eth0]

Run in a NON-foxy terminal (it imports the Unitree SDK).
Needs pose_server.py AND lidar_node.py already running in foxy terminals.

WHAT THIS IS AND IS NOT
-----------------------
This is straight-line odometry navigation with reactive obstacle dodging.
It is NOT a path planner: it has no map, so it cannot route around a wall
that fully blocks the direct line. If it gets boxed in it gives up and
stops rather than wandering. Recorded poses are odometry-relative and do
not survive a robot reboot.

SAFETY
------
* Forward speed is never negative - the Go2 has no rear sensing, so the
  robot must never reverse under program control.
* On SIGTERM (which is what the master bridge's `stop` tool sends) the
  finally block still runs and issues StopMove.
* Aborts on: arrival, timeout, no progress (stuck), lost odometry, or
  being boxed in with nowhere to dodge.
"""

import math
import signal
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from pose_client import PoseClient, wrap_pi
from lidar_client import LidarClient
from record_room import load_rooms

# ---------------- TUNING ----------------
ARRIVE_M = 0.35          # close enough to call it arrived
TURN_GATE = 0.60         # rad; above this, turn in place before walking
FWD_KP = 0.55
FWD_MAX = 0.45           # deliberately below the master bridge's clamp
ROT_KP = 1.1
ROT_MAX = 0.7
ROT_DEADBAND = 0.08
LAT_MAX = 0.35

LOOP_HZ = 20.0
TIMEOUT_S = 120.0        # hard ceiling on one navigation run
STUCK_SEC = 12.0         # no progress toward the goal for this long -> abort
STUCK_EPS = 0.15         # metres of improvement that counts as progress
TRAPPED_SEC = 6.0        # boxed in with no dodge available for this long
LOST_POSE_SEC = 2.0      # odometry gone for this long -> abort

# slew rates (units per second): ACC = away from zero, DEC = toward zero
ACC_FWD, DEC_FWD = 0.6, 1.5
ACC_LAT, DEC_LAT = 0.8, 1.5
ACC_ROT, DEC_ROT = 2.0, 3.0
SEND_EPS = 0.02


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def slew(cur, target, acc, dec, dt):
    if target > cur:
        rate = acc if cur >= 0 else dec
        return min(cur + rate * dt, target)
    rate = acc if cur <= 0 else dec
    return max(cur - rate * dt, target)


class Stopped(Exception):
    """Raised by the SIGTERM handler so the finally block runs."""


def main():
    if len(sys.argv) < 2:
        print("usage: python3 go_to_room.py <room> [interface]")
        return 2
    room_name = sys.argv[1]
    interface = sys.argv[2] if len(sys.argv) > 2 else "eth0"

    rooms = load_rooms()
    if room_name not in rooms:
        print("ERROR: no room named '%s'." % room_name)
        print("Known rooms: %s" % (", ".join(sorted(rooms)) or "(none)"))
        return 1
    target = rooms[room_name]
    tx, ty = float(target["x"]), float(target["y"])

    # ---- helper processes must already be up ----
    pc = PoseClient()
    if not pc.wait_ready(5.0):
        print("ERROR: no pose from pose_server.py. Start it in a foxy terminal.")
        return 1
    lc = LidarClient()
    if not lc.alive():
        print("ERROR: lidar_node.py is not answering. Start it in a foxy terminal.")
        print("Refusing to navigate without obstacle avoidance.")
        return 1

    print("Init channel on %s ..." % interface)
    ChannelFactoryInitialize(0, interface)
    time.sleep(0.5)
    robot = SportClient()
    robot.SetTimeout(3.0)
    robot.Init()
    print("SportClient ready.")

    def on_term(signum, frame):
        raise Stopped()
    signal.signal(signal.SIGTERM, on_term)

    moving = False
    cmd_fwd = cmd_lat = cmd_rot = 0.0
    started = time.time()
    last_loop = started
    last_pose_ok = started
    best_dist = float("inf")
    best_at = started
    trapped_since = None
    period = 1.0 / LOOP_HZ
    outcome = "interrupted"

    def send(fwd, lat, rot):
        nonlocal moving
        if abs(fwd) > SEND_EPS or abs(lat) > SEND_EPS or abs(rot) > SEND_EPS:
            try:
                robot.Move(fwd, lat, rot)
                moving = True
            except Exception as exc:
                print("[robot] Move failed: %s" % exc)
        elif moving:
            try:
                robot.StopMove()
            except Exception:
                pass
            moving = False

    print("\n=== GO TO '%s'  (target x=%.2f y=%.2f) ===" % (room_name, tx, ty))
    print("L2+B on the remote is the physical e-stop.\n")

    try:
        while True:
            now = time.time()
            dt = clamp(now - last_loop, 0.0, 0.2)
            last_loop = now

            if now - started > TIMEOUT_S:
                outcome = "timeout after %.0fs" % TIMEOUT_S
                break

            pose = pc.get()
            if pose is None:
                if now - last_pose_ok > LOST_POSE_SEC:
                    outcome = "lost odometry"
                    break
                cmd_fwd = slew(cmd_fwd, 0.0, ACC_FWD, DEC_FWD, dt)
                cmd_lat = slew(cmd_lat, 0.0, ACC_LAT, DEC_LAT, dt)
                cmd_rot = slew(cmd_rot, 0.0, ACC_ROT, DEC_ROT, dt)
                send(cmd_fwd, cmd_lat, cmd_rot)
                time.sleep(period)
                continue
            last_pose_ok = now

            dx = tx - pose.x
            dy = ty - pose.y
            dist = math.hypot(dx, dy)

            if dist < ARRIVE_M:
                outcome = "ARRIVED (%.2f m from target)" % dist
                break

            # stuck detection: measured against the best distance so far
            if dist < best_dist - STUCK_EPS:
                best_dist = dist
                best_at = now
            elif now - best_at > STUCK_SEC:
                outcome = "stuck - no progress for %.0fs (%.2f m short)" % (
                    STUCK_SEC, dist)
                break

            err = wrap_pi(math.atan2(dy, dx) - pose.yaw)

            tgt_rot = 0.0
            tgt_fwd = 0.0
            tgt_lat = 0.0
            if abs(err) > ROT_DEADBAND:
                tgt_rot = clamp(ROT_KP * err, -ROT_MAX, ROT_MAX)
            if abs(err) <= TURN_GATE:
                # only walk once roughly pointed at the target; never reverse
                tgt_fwd = clamp(FWD_KP * dist, 0.0, FWD_MAX)

            adj = lc.compute([])        # no camera here: everything is an obstacle
            if adj.have_scan and tgt_fwd > 0.01:
                if adj.block_forward:
                    tgt_fwd = 0.0
                else:
                    tgt_fwd *= adj.fwd_scale
                tgt_lat = clamp(adj.lat, -LAT_MAX, LAT_MAX)

            # boxed in: blocked with no dodge available
            if adj.have_scan and adj.block_forward and abs(adj.lat) < 0.01:
                trapped_since = trapped_since or now
                if now - trapped_since > TRAPPED_SEC:
                    outcome = "boxed in - blocked with nowhere to dodge"
                    break
            else:
                trapped_since = None

            cmd_fwd = slew(cmd_fwd, max(0.0, tgt_fwd), ACC_FWD, DEC_FWD, dt)
            cmd_lat = slew(cmd_lat, tgt_lat, ACC_LAT, DEC_LAT, dt)
            cmd_rot = slew(cmd_rot, tgt_rot, ACC_ROT, DEC_ROT, dt)
            send(cmd_fwd, cmd_lat, cmd_rot)

            print("\rdist=%5.2f  err=%6.1fdeg  f=%.2f l=%+.2f r=%+.2f  %s      "
                  % (dist, math.degrees(err), cmd_fwd, cmd_lat, cmd_rot,
                     adj.status[:34]), end="", flush=True)
            time.sleep(period)

    except (KeyboardInterrupt, Stopped):
        outcome = "stopped by operator"
    finally:
        try:
            robot.StopMove()
        except Exception:
            pass
        # always lower into a low, stable crouch when the program stops, so
        # the robot never stays standing unattended
        time.sleep(0.3)
        try:
            print("[robot] StandDown -> %s" % robot.StandDown())
        except Exception as exc:
            print("[robot] StandDown failed: %s" % exc)
        pc.shutdown()
        lc.shutdown()
        print("\n=== %s: %s ===" % (room_name, outcome))

    return 0 if outcome.startswith("ARRIVED") else 1


if __name__ == "__main__":
    sys.exit(main())
