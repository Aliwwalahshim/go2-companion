"""
climb_stairs.py - standalone stair-climbing demo for the Unitree Go2.

Enters AI sport mode (stair_mode.py), sends slow, speed-capped forward
Move() commands while the robot's own onboard controller runs whatever
gait it decides fits the terrain (including the stair gait) using its 4D
LiDAR terrain perception, then restores the previous mode. This file does
not implement a gait - it cannot; Go2's SportClient has no such command on
this SDK (see STAIR_MODE_README.md for the research trail). All this does
is unlock AI mode, drive forward slowly while it is active, and hand
control back cleanly.

SAFETY - read before pointing this at a real staircase
--------------------------------------------------------
- Ascent only, by default. Community reports are consistent that this
  robot climbs up reliably and struggles coming down; --allow-descent
  exists to acknowledge that risk but changes nothing about what is sent -
  it is still forward walking, just aimed at a descent. That is on you.
- Forward speed is hard-capped at 0.3 m/s in code (MAX_STAIR_SPEED),
  regardless of what --speed asks for.
- No reverse, ever. There is no rear sensing on this robot. vx is clamped
  to [0, MAX_STAIR_SPEED] no matter what; this script never sends a
  negative x to Move().
- If AI mode cannot be verified active, no Move() command is sent at all -
  stair_mode.enter_stair_mode() raises before this script touches the
  robot.
- The previous mode is always restored on exit, including on Ctrl+C or a
  crash (finally block around the whole drive loop).
- The remote's L2+B is the physical e-stop. A human must be present with
  the remote in hand for every test - this script is not a substitute for
  that, and does not attempt to be.

See STAIR_MODE_README.md for the exact run order, which terminals must be
foxy-sourced (none, for this script), and the network state required
(eth0 must carry only 192.168.123.18/24, and no other script may already
hold the robot - see that file for how to check).

MANUAL MODE (--manual)
-----------------------
If you'd rather drive with the physical remote yourself and just watch a
live camera preview to see whether the robot's camera actually has the
staircase in frame, use --manual instead of letting this script send
Move() commands. In that mode this file sends no movement command at
all - it only enters AI mode (so the remote's own driving unlocks the
stair gait) and opens a cv2 preview window of the RealSense color feed.
Press 'q' or ESC in that window, or Ctrl+C here, to end and restore the
previous mode. The remote's L2+B e-stop still applies.
"""

import argparse
import sys
import time

import stair_mode
from stair_mode import StairModeError

MAX_STAIR_SPEED = 0.3          # m/s - hard cap, independent of --speed
MOVE_INTERVAL = 0.1            # s between Move() sends while climbing


def clamp_forward(v):
    """Never negative (never reverse), never above the stair-mode cap."""
    return max(0.0, min(v, MAX_STAIR_SPEED))


def climb(sport, duration, speed, disable_avoid, dry_run):
    speed = clamp_forward(speed)

    try:
        stair_mode.enter_stair_mode(disable_onboard_avoid=disable_avoid,
                                     dry_run=dry_run)
    except StairModeError as exc:
        print("[climb_stairs] could not verify AI mode - refusing to send "
              "any movement command: %s" % exc)
        return 1

    print("[climb_stairs] driving forward at %.2f m/s for %.1fs. "
          "L2+B on the remote is the e-stop." % (speed, duration))
    try:
        end = time.time() + duration
        while time.time() < end:
            if dry_run:
                print("   [dry-run] Move(%.2f, 0.0, 0.0)" % speed)
            else:
                try:
                    sport.Move(speed, 0.0, 0.0)
                except Exception as exc:
                    print("[climb_stairs] Move failed: %s" % exc)
                    break
            time.sleep(MOVE_INTERVAL)
    finally:
        if not dry_run and sport is not None:
            try:
                sport.StopMove()
            except Exception as exc:
                print("[climb_stairs] StopMove failed: %s" % exc)
        stair_mode.exit_stair_mode()

    return 0


def watch(disable_avoid, dry_run):
    """Manual-drive companion to climb(): enter AI mode, send no Move()
    at all (the physical remote drives), just show a live camera preview
    until 'q'/ESC/Ctrl+C, then restore the previous mode."""
    try:
        stair_mode.enter_stair_mode(disable_onboard_avoid=disable_avoid,
                                     dry_run=dry_run)
    except StairModeError as exc:
        print("[climb_stairs] could not verify AI mode: %s" % exc)
        return 1

    print("[climb_stairs] AI mode active. Drive with the physical remote "
          "now - this script sends no movement commands. L2+B on the "
          "remote is the e-stop. Press 'q' or ESC in the preview window, "
          "or Ctrl+C here, to end and restore the previous mode.")

    if dry_run:
        print("[climb_stairs] DRY RUN - no camera is opened. Sleeping "
              "5s to simulate the watch session, then restoring mode.")
        try:
            time.sleep(5.0)
        except KeyboardInterrupt:
            print("\n[climb_stairs] interrupted.")
        finally:
            stair_mode.exit_stair_mode()
        return 0

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("[climb_stairs] camera preview window open.")

    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            cv2.putText(img, "AI MODE ACTIVE - drive with the remote.",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(img, "L2+B = e-stop.  q/ESC = end + restore mode.",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow("Go2 stair watch", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[climb_stairs] interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        stair_mode.exit_stair_mode()

    return 0


def main():
    ap = argparse.ArgumentParser(description="Go2 AI-mode stair climb demo")
    ap.add_argument("interface", nargs="?", default="eth0")
    ap.add_argument("--speed", type=float, default=0.2,
                    help="requested forward speed in m/s (hard-capped at "
                         "%.1f regardless of this value)" % MAX_STAIR_SPEED)
    ap.add_argument("--duration", type=float, default=8.0,
                    help="seconds to drive forward once AI mode is confirmed")
    ap.add_argument("--allow-descent", action="store_true",
                    help="acknowledge the descent risk described in this "
                         "file's docstring. Does not change any command "
                         "sent - it is still forward walking. Pointing the "
                         "robot at a descent instead of an ascent is on you.")
    ap.add_argument("--disable-avoid", action="store_true",
                    help="also disable the onboard LiDAR obstacle-avoidance "
                         "switch before climbing (unverified community "
                         "report: it may treat a stair riser as a wall and "
                         "refuse to approach it). Restored on exit "
                         "regardless of how this script ends.")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the whole mode switch/verify/restore sequence "
                         "against nothing - no channel is opened, no "
                         "MotionSwitcherClient or SportClient touches the "
                         "network, Move() is only printed")
    ap.add_argument("--manual", action="store_true",
                    help="send no Move() commands at all - enter AI mode, "
                         "show a live camera preview, and let the physical "
                         "remote drive. Ends (and restores mode) on 'q'/ESC "
                         "in the preview window or Ctrl+C.")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive 'Press Enter to continue' "
                         "gate. Only pass this if a human is ALREADY "
                         "standing there with the physical remote in hand, "
                         "ready to e-stop with L2+B - this flag exists so "
                         "that acknowledgment can be given once, "
                         "explicitly, up front (e.g. by whoever is "
                         "operating this script on the human's behalf), "
                         "not so the check can be silently skipped.")
    args = ap.parse_args()

    if args.speed > MAX_STAIR_SPEED:
        print("[climb_stairs] --speed %.2f exceeds the %.1f m/s stair-mode "
              "cap; it will be clamped." % (args.speed, MAX_STAIR_SPEED))

    if args.allow_descent:
        print("[climb_stairs] WARNING: --allow-descent is set. Descending "
              "stairs is this robot's documented weak point - community "
              "reports are consistent that it goes up reliably and "
              "struggles coming down. Proceed only with a spotter and the "
              "remote in hand, aimed at a short, shallow, carpeted flight "
              "if at all possible.")

    print("WARNING: stair climbing must only be attempted with a human "
          "holding the remote, standing where they can reach the robot. "
          "L2+B on the remote is the physical e-stop.")
    if not args.dry_run:
        if args.yes:
            print("--yes given: proceeding without the interactive prompt "
                  "on the assumption a human is already holding the "
                  "remote.")
        else:
            input("Press Enter to continue...")

    if args.manual:
        if not args.dry_run:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            print("Initializing channel on %s ..." % args.interface)
            ChannelFactoryInitialize(0, args.interface)
            time.sleep(0.5)
        return watch(args.disable_avoid, args.dry_run)

    sport = None
    if not args.dry_run:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        print("Initializing channel on %s ..." % args.interface)
        ChannelFactoryInitialize(0, args.interface)
        time.sleep(0.5)
        sport = SportClient()
        sport.SetTimeout(3.0)
        sport.Init()
        print("SportClient ready.")

    return climb(sport, args.duration, args.speed, args.disable_avoid,
                 args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
