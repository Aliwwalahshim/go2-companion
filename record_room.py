"""
record_room.py - Save the robot's current pose under a room name.

Reads the pose socket only. Does NOT touch the SDK, so it is safe to run
in either kind of terminal (it needs pose_server.py running, nothing else).

    python3 record_room.py kitchen
    python3 record_room.py --list
    python3 record_room.py --delete kitchen

Walk the robot to the spot with the remote first, then record. Poses are
odometry-relative, so every room in rooms.json is only valid until the
robot's odometry is reset (a reboot). Re-record after a reboot.
"""

import argparse
import json
import math
import os
import sys

from pose_client import PoseClient

ROOMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rooms.json")


def load_rooms(path=ROOMS_FILE):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError) as exc:
        print("[rooms] could not read %s: %s" % (path, exc))
        return {}


def save_rooms(rooms, path=ROOMS_FILE):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rooms, fh, indent=2, ensure_ascii=False, sort_keys=True)


def list_rooms(rooms):
    if not rooms:
        print("No rooms recorded yet.")
        return
    print("Recorded rooms (%d):" % len(rooms))
    for name in sorted(rooms):
        r = rooms[name]
        print("  %-16s x=%7.2f  y=%7.2f  yaw=%7.1f deg"
              % (name, r["x"], r["y"], math.degrees(r["yaw"])))


def main():
    ap = argparse.ArgumentParser(description="Record a named room pose")
    ap.add_argument("name", nargs="?", help="room name to record")
    ap.add_argument("--list", action="store_true", help="list recorded rooms")
    ap.add_argument("--delete", metavar="NAME", help="delete a recorded room")
    args = ap.parse_args()

    rooms = load_rooms()

    if args.list:
        list_rooms(rooms)
        return 0

    if args.delete:
        if args.delete not in rooms:
            print("No room named '%s'." % args.delete)
            return 1
        del rooms[args.delete]
        save_rooms(rooms)
        print("Deleted '%s'." % args.delete)
        return 0

    if not args.name:
        ap.error("give a room name, or use --list / --delete")

    pc = PoseClient()
    if not pc.wait_ready(5.0):
        print("ERROR: no pose from pose_server.py.")
        print("Start it first, in a foxy-sourced terminal:")
        print("    python3 pose_server.py")
        return 1

    pose = pc.get()
    pc.shutdown()
    if pose is None:
        print("ERROR: pose went stale while recording. Try again.")
        return 1

    existed = args.name in rooms
    rooms[args.name] = {"x": pose.x, "y": pose.y, "yaw": pose.yaw}
    save_rooms(rooms)
    print("%s '%s'  ->  x=%.2f  y=%.2f  yaw=%.1f deg"
          % ("Updated" if existed else "Recorded", args.name,
             pose.x, pose.y, math.degrees(pose.yaw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
