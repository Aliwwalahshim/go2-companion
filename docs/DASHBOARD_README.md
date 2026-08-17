# Go2 visual navigation interface

Two interfaces for mapping/navigation on the Go2 EDU, replacing rviz2:

- **Phase A - Foxglove** (off-the-shelf 3D viewer, via rosbridge)
- **Phase B - custom web dashboard** (the competition demo: 2D map, live robot
  pose, one-click "go to room")

All files live in `/home/unitree/unitree_sdk2_python/example/go2/front_camera/`. The dashboard reuses the
existing `pose_server.py` / `pose_client.py` / `go_to_room.py` /
`record_room.py` / `rooms.json` in
`unitree_sdk2_python/example/go2/front_camera/` - none of those were modified.

The `foxy` alias (in `~/.bashrc`) is what "foxy-sourced" means below. Never set
`CYCLONEDDS_URI` by hand - the alias already uses the correct config.

---

## Phase A - Foxglove (via rosbridge)

`foxglove_bridge` does NOT work on Foxy, so we use **rosbridge**, which
Foxglove still supports. It is **already installed** (arm64, via apt -
`ros-foxy-rosbridge-server 1.3.1`). The three `~/*rosbridge*_amd64.deb` files
in the home folder are the wrong architecture and were NOT used.

### Run

```bash
foxy
ros2 launch rosbridge_server rosbridge_websocket_launch.xml   # listens on :9090
```

### Connect from a laptop

Find the Jetson's WiFi IP (it changes - do not hardcode):

```bash
ip -4 addr show wlan0 | grep inet     # e.g. 192.168.8.174
```

In Chrome open https://app.foxglove.dev -> **Open connection -> Rosbridge**,
URL: `ws://<jetson-wifi-ip>:9090`. Then import the layout
`foxglove_go2_layout.json` (Layouts -> Import). It contains:
- a 3D panel: `/rtabmap/cloud_map`, `/rtabmap/grid_map`, TF, robot pose
- a Raw Messages panel on `/utlidar/robot_odom.pose.pose`
- an X/Y plot of the odometry

`/rtabmap/*` only appear once RTAB-Map is running (see Phase B run order).

### Verified

rosbridge end-to-end was confirmed on the Jetson: it serves 109 topics and
live `/utlidar/robot_odom` flows through the websocket (a local ws client
received real poses). See Limitations for the honest bit about WiFi frame rate.

---

## Phase B - the web dashboard

A single dark-themed page: the 2D occupancy map fills the view, the robot's
live position + heading is drawn on it, each recorded room is a button, plus a
big STOP and a "record room here" control.

### Architecture (respects the rclpy/DDS separation)

```
 foxy terminals (rclpy)                 non-foxy terminal (SDK side)
 ----------------------                 ---------------------------
 RTAB-Map  -> /rtabmap/grid_map \
 map_bridge.py -> map.png/json   >----- dashboard_server.py (tornado, NO rclpy)
 pose_server.py -> UDP :47002   /         |  reads pose via pose_client (UDP)
                                          |  reads map.png / map.json (files)
                                          |  launches go_to_room.py / record_room.py
                                          |    as subprocesses (inherit this env)
                                          v
                                   laptop browser over WiFi (wlan0)
```

- `map_bridge.py` is the only new rclpy process. It subscribes to
  `/rtabmap/grid_map` and writes `/tmp/go2_dashboard/map.png` + `map.json`.
- `dashboard_server.py` **never imports rclpy**. It reads those files + the
  pose UDP socket and launches the SDK scripts as subprocesses.
- Web traffic is on **wlan0**; robot control stays on **eth0**
  (`192.168.123.18/24` only). They never mix.

### Run order

```bash
# one-time before mapping (needs sudo):
sudo /usr/sbin/nvpmodel -m 0
sudo sysctl -w net.core.rmem_max=52428800

# Terminal 1 - foxy - RTAB-Map (already-working launch; do not rebuild it)
foxy
ros2 launch /home/unitree/go2_slam/rtabmap_mapping.launch.py

# Terminal 2 - foxy - map bridge (grid -> PNG for the web UI)
foxy
python3 /home/unitree/unitree_sdk2_python/example/go2/front_camera/map_bridge.py

# Terminal 3 - foxy - pose service over UDP (the dashboard reads this)
foxy
python3 /home/unitree/unitree_sdk2_python/example/go2/front_camera/pose_server.py

# Terminal 4 - NOT foxy - the web server
python3 /home/unitree/unitree_sdk2_python/example/go2/front_camera/dashboard_server.py --interface eth0 --port 8080
```

Then open `http://<jetson-wifi-ip>:8080` from a laptop/tablet.

Drive the robot around with the remote to build the map (RTAB-Map only extends
it while moving). Rooms recorded via `record_room.py` (or the dashboard's
"Record room here") show up as buttons; clicking one drives there via
`go_to_room.py`; STOP kills that subprocess (which issues `StopMove` in its
SIGTERM handler).

Note: `go_to_room.py` has its own dependencies (it needs `pose_server.py`, and
it imports `lidar_client` for obstacle dodging - if `lidar_node.py` is not
running it will drive without that dodging). That behaviour is unchanged; the
dashboard just launches it.

---

## Limitations - tested vs only launched

**Verified on the real robot / Jetson:**
- rosbridge serves topics and streams live `/utlidar/robot_odom` over the
  websocket (109 topics; real pose values received by a local ws client).
- `map_bridge.py` receives an OccupancyGrid and writes a correct PNG + JSON
  (confirmed with a synthetic 80x60 grid -> valid 80x60 RGBA PNG, right
  resolution/origin in `map.json`).
- `dashboard_server.py`: serves the page, `/api/state` (rooms from
  `rooms.json`, map metadata, status), `/map.png` (HTTP 200 image/png),
  `/api/goto` rejects unknown rooms (400), `/api/stop` is safe when idle.
- All Python compiles (`py_compile`); all source files are ASCII-only.

**NOT yet exercised end-to-end (be aware before the demo):**
- **Live robot pose drawn on the map** needs `pose_server.py` running; the
  drawing math is in `app.js` but was not visually confirmed in a browser here
  (no laptop in this session). Test it before the demo.
- **An actual "go to room" drive** was not run - it moves the robot and needs
  `pose_server.py` + a real driven map. The subprocess launch/stop wiring is
  tested; the driving itself is `go_to_room.py`'s existing behaviour.
- **Foxglove frame rate over WiFi was not measured** (no laptop). Honest
  expectation: rosbridge is JSON-based, so `/utlidar/cloud` (~15 Hz, ~4200
  pts) and `/rtabmap/cloud_map` (can be large) may stutter over WiFi. If so,
  keep the heavy point clouds hidden in Foxglove and rely on the 2D
  `/rtabmap/grid_map` + pose, or add a throttling relay (republish the cloud
  at 2-5 Hz) for visualisation. The dashboard (Phase B) deliberately sends NO
  point cloud to the browser - only a small PNG - so it stays light over WiFi.
- The map PNG updates only as fast as RTAB-Map republishes `/rtabmap/grid_map`
  (a few Hz at most); the robot pose overlay updates ~5 Hz from the UDP socket.

**What the WiFi link can realistically carry:** the Phase B dashboard is
designed for it - a ~tens-of-KB PNG on map change plus a tiny JSON state poll
at 5 Hz. That is fine over WiFi. Phase A's raw point clouds are the part that
may not be.
