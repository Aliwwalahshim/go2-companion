"""
map_bridge.py - ROS 2 side of the dashboard. Subscribes to the RTAB-Map 2D
occupancy grid and writes it out as a PNG + JSON so the (rclpy-free) web
server can serve it.

This is the ONLY new rclpy process. It follows the project rule: rclpy lives
here, in a foxy-sourced terminal, NEVER in the web server or the SDK/control
process. Run it alongside the existing pose_server.py (which supplies the
robot pose over UDP).

    foxy
    python3 map_bridge.py

Writes, on every map update, to /tmp/go2_dashboard/ :
    map.png   - RGBA top-down occupancy grid (free = light, occupied = dark,
                unknown = transparent so the dark UI shows through)
    map.json  - {resolution, origin_x, origin_y, width, height, stamp}

The image is stored with row 0 = TOP = maximum world-y, so the web canvas can
map world (x, y) -> pixel with:
    col = (x - origin_x) / resolution
    row = (height - 1) - (y - origin_y) / resolution
"""

import json
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

OUT_DIR = "/tmp/go2_dashboard"
MAP_PNG = os.path.join(OUT_DIR, "map.png")
MAP_JSON = os.path.join(OUT_DIR, "map.json")
# This installed RTAB-Map (the newer split rtabmap_slam packages) publishes
# the 2D occupancy grid on /map, NOT /rtabmap/grid_map. Override at runtime if
# your build differs:  python3 map_bridge.py --ros-args -p grid_topic:=/rtabmap/grid_map
DEFAULT_GRID_TOPIC = "/map"

# RGBA colours for the three occupancy classes (tuned for a dark UI)
C_FREE = (206, 212, 222, 255)      # light grey - drivable
C_OCC = (12, 14, 20, 255)          # near-black - walls/obstacles
C_UNKNOWN = (0, 0, 0, 0)           # transparent - not yet seen


class MapBridge(Node):
    def __init__(self):
        super().__init__("map_bridge")
        os.makedirs(OUT_DIR, exist_ok=True)
        # VOLATILE + RELIABLE is compatible with either a volatile or a
        # transient_local publisher and still gets every live republish.
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        qos.history = HistoryPolicy.KEEP_LAST
        self.declare_parameter("grid_topic", DEFAULT_GRID_TOPIC)
        topic = self.get_parameter("grid_topic").value
        self._sub = self.create_subscription(
            OccupancyGrid, topic, self._on_grid, qos)
        self._count = 0
        self.get_logger().info("map_bridge: waiting for %s ..." % topic)

    def _on_grid(self, msg):
        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            return
        data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[...] = C_UNKNOWN
        rgba[data == 0] = C_FREE
        rgba[data >= 50] = C_OCC              # 50..100 treated as occupied
        # row 0 of OccupancyGrid is the origin (min y); flip so image top = max y
        rgba = np.flipud(rgba)

        try:
            from PIL import Image
            img = Image.fromarray(rgba, "RGBA")
            tmp_png = MAP_PNG + ".tmp"
            img.save(tmp_png, "PNG")          # explicit - temp name has no .png ext
            os.replace(tmp_png, MAP_PNG)
        except Exception as exc:
            self.get_logger().error("map PNG write failed: %s" % exc)
            return

        meta = {
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
            "width": int(w),
            "height": int(h),
            "stamp": time.time(),
        }
        tmp_json = MAP_JSON + ".tmp"
        with open(tmp_json, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp_json, MAP_JSON)

        self._count += 1
        if self._count % 10 == 1:
            self.get_logger().info(
                "map update #%d: %dx%d @ %.3f m/px" % (self._count, w, h,
                                                       msg.info.resolution))


def main():
    rclpy.init()
    node = MapBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
