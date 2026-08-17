#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
lidar_watch.py  --  live view of what the LiDAR avoider decides.

Run it in its own terminal (NOT sourced with foxy) while lidar_node.py
is running.  It prints the current decision several times a second so you
can hold a box in one place and read the reaction immediately.

    python3 lidar_watch.py

Box test:
    box ~0.7 m in FRONT  -> expect  BLOCK  (or slowing)
    box ~0.7 m on LEFT   -> expect  push RIGHT
    box ~0.7 m on RIGHT  -> expect  push LEFT
"""

import time
from lidar_client import LidarAvoider


def bar(v, width=20):
    """Little text bar for the lateral dodge command."""
    mid = width // 2
    cells = ["-"] * width
    cells[mid] = "|"
    pos = int(round(mid + v * mid / 0.5))
    pos = max(0, min(width - 1, pos))
    cells[pos] = "#"
    return "".join(cells)


def main():
    av = LidarAvoider()
    print("Watching LiDAR avoider.  Ctrl+C to stop.")
    print("%-8s %-26s %6s %6s %7s  %s"
          % ("scan", "status", "fwdx", "lat", "block", "lat bar  L<--|-->R"))
    print("-" * 92)
    try:
        while True:
            a = av.compute([])          # no people masked: pure obstacle view
            if not a.have_scan:
                print("%-8s %-26s" % ("NONE", "no scan from lidar_node"))
            else:
                print("%-8s %-26s %6.2f %6.2f %7s  %s"
                      % ("ok",
                         a.status[:26],
                         getattr(a, "fwd_scale", 1.0),
                         a.lat,
                         "YES" if a.block_forward else "no",
                         bar(a.lat)))
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            av.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
