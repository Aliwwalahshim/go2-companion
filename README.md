# Unitree Go2 Companion Stack

A collection of control, perception, audio and navigation tools for the
**Unitree Go2 EDU** quadruped, built on top of the Unitree Python SDK and
ROS 2 Foxy, running on the robot's onboard Jetson.

The robot can be driven by **hand gestures**, **follow a person** while
avoiding obstacles, **speak and listen through its own head hardware over
WebRTC**, take **voice commands offline**, climb stairs via the onboard AI
gait, and be operated from a **web dashboard** that shows a live SLAM map and
one-click "go to room" navigation.

> Runs on a Go2 EDU with the built-in 4D LiDAR L1 and an Intel RealSense
> depth camera, on the onboard Jetson (Ubuntu 20.04, Python 3.8, ROS 2 Foxy).

---

## Features

| Area | What it does | Entry point |
|------|--------------|-------------|
| **Gesture console** | Drive and pose the robot with hand gestures (MediaPipe) | `go2_master.py` |
| **Person follow** | Follows a locked operator with RealSense depth obstacle-avoidance; open-palm = STOP | `follow_depth_avoid.py` |
| **Robot speech** | TTS out of the robot's own head speaker over WebRTC (not the Jetson jack) | `go2_speaker.py` |
| **Robot mic** | Capture the robot mic over WebRTC - record to WAV or stream to a callback | `go2_mic.py` |
| **Voice commands** | Fully offline speech-to-command with Vosk, fed from the robot mic | `go2_voice.py` |
| **Stair mode** | Enter the robot's onboard AI gait for stair climbing, with safety limits | `climb_stairs.py`, `stair_mode.py` |
| **Web dashboard** | Live 2D SLAM map + robot pose + one-click room navigation, in the browser | `dashboard_server.py` |
| **Foxglove** | Ready-made layout for 3D map / pose / odometry viewing over rosbridge | `foxglove_go2_layout.json` |
| **Waypoint nav** | Record named rooms from odometry and drive to them | `record_room.py`, `go_to_room.py` |

See [`docs/`](docs/) for the deep-dive write-ups on the speaker, stair mode and
dashboard.

---

## Requirements

**Hardware**
- Unitree Go2 **EDU** (the head speaker/mic paths need Pro/EDU)
- Built-in 4D LiDAR L1 (used for SLAM) + Intel RealSense D4xx (depth)
- Onboard Jetson (Orin NX here)

**Software**
- Ubuntu 20.04, Python 3.8, ROS 2 Foxy
- `unitree_sdk2py` (Unitree's Python SDK)
- Python packages: see [`requirements.txt`](requirements.txt)
- System: `sudo apt install portaudio19-dev` (for the audio stack),
  `ros-foxy-slam-toolbox ros-foxy-pointcloud-to-laserscan ros-foxy-nav2-map-server`
  and `ros-foxy-rosbridge-server` (for SLAM / Foxglove)
- Vosk model (offline speech), downloaded separately (see below)

```bash
pip3 install --user -r requirements.txt

# offline speech model (not committed - ~40 MB):
mkdir -p ~/vosk_models && cd ~/vosk_models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

---

## Repository layout

```
.
|-- go2_master.py            # gesture control console (speaks each action)
|-- follow_depth_avoid.py    # operator-follow + depth avoidance + stop-sign
|-- gesture_detector.py      # MediaPipe hand/pose gesture recognition
|-- person_detector.py       # YOLOv8 operator lock/track
|-- depth_avoider.py         # RealSense depth -> obstacle avoidance
|
|-- go2_speaker.py           # robot head speaker (WebRTC TTS)
|-- go2_mic.py               # robot mic capture (WebRTC, record + stream)
|-- go2_voice.py             # offline Vosk voice commands (mic -> actions)
|
|-- stair_mode.py            # AI-mode switch wrapper (safe enter/exit)
|-- climb_stairs.py          # stair-climb demo
|
|-- dashboard_server.py      # web dashboard (Tornado, no rclpy)
|-- map_bridge.py            # ROS 2 -> PNG map bridge for the dashboard
|-- static/                  # dashboard HTML/CSS/JS
|-- foxglove_go2_layout.json # Foxglove Studio layout
|
|-- pose_server.py / pose_client.py   # odometry pose over UDP
|-- record_room.py / go_to_room.py    # waypoint record + drive
|-- rooms.json               # example recorded rooms
|
|-- lidar_*.py               # LiDAR client / viewers / avoiders
|-- docs/                    # per-feature deep-dive READMEs
```

---

## Quick start

Each tool is a standalone script. A few common ones:

```bash
# gesture console (robot stands up, driven by hand gestures, speaks actions)
python3 go2_master.py eth0

# follow a person (open palm to STOP)
python3 follow_depth_avoid.py eth0

# make the robot talk / record its mic
python3 go2_speaker.py --say "hello from the robot"
python3 go2_mic.py --seconds 5 --out clip.wav

# offline voice commands
python3 go2_voice.py

# web dashboard (open http://<jetson-wifi-ip>:8080)
python3 dashboard_server.py --interface eth0 --port 8080
```

Full run orders (which terminals are ROS 2-sourced, the network state, etc.)
are in [`docs/DASHBOARD_README.md`](docs/DASHBOARD_README.md),
[`docs/GO2_SPEAKER_README.md`](docs/GO2_SPEAKER_README.md) and
[`docs/STAIR_MODE_README.md`](docs/STAIR_MODE_README.md).

---

## Architecture notes (why it is split the way it is)

- **`rclpy` and the Unitree DDS cannot share one process** - they collide on
  CycloneDDS. ROS 2 nodes (SLAM, `map_bridge.py`, `pose_server.py`) run as
  separate processes and hand data to the SDK/control side over local UDP
  sockets or files. The control scripts never import `rclpy`.
- **Audio is over WebRTC, not ALSA.** The robot's speaker and mic are on the
  robot's main computer, not the Jetson - they are reached with the same
  protocol the mobile app uses. `go2_speaker.py` / `go2_mic.py` run the WebRTC
  stack in a separate worker process to keep it isolated from DDS.
- **The dashboard sends only a small PNG + JSON** to the browser (no point
  clouds), so it stays light over WiFi.

---

## Known limitations (honest)

- **The robot's built-in microphone is not exposed by Unitree.** The WebRTC
  audio-receive channel returns constant filler, not live mic audio (verified:
  loud sound produces zero response). `go2_mic.py` / `go2_voice.py` are correct
  and work with a real source - use a **USB mic on the Jetson** for voice
  commands. The **speaker** works fully.
- **Stair climbing** relies on the robot's onboard AI gait; the SDK
  `SelectMode("ai")` switch can be firmware-dependent.
- 2D SLAM assumes roughly planar motion; the LiDAR-to-base mount offset in the
  SLAM launch is an assumption and should be measured for metric accuracy.
- Recorded rooms are only repeatable once map-based localization is running;
  raw odometry drifts and resets on reboot.

---

## License

MIT - see [LICENSE](LICENSE). Built on the Unitree SDK and open-source ROS 2 /
Foxglove / Vosk / MediaPipe / Ultralytics projects, which retain their own
licenses.
