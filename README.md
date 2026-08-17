# Unitree Go2 Control Stack

Gesture control, a desktop control GUI, and person-following for the
**Unitree Go2 EDU**, running on the robot's onboard Jetson (Ubuntu 20.04,
Python 3.8) with the Unitree Python SDK and an Intel RealSense depth camera.

Everything here has been run and tested on the real robot.

---

## Features

| Tool | What it does |
|------|--------------|
| **`go2_master.py`** | Gesture console - drive and pose the robot with hand gestures (MediaPipe). Each action is announced out loud through the robot's own head speaker (over WebRTC). |
| **`go2_control_gui.py`** | Desktop control panel (PyQt5) - drive the robot, trigger actions, and see the camera feed, from a GUI. |
| **`follow_depth_avoid.py`** | Person follow - locks onto an operator (YOLOv8), follows them, and uses RealSense depth for obstacle avoidance. Hold an open palm to STOP. |

Supporting modules: `gesture_detector.py` (MediaPipe gestures),
`person_detector.py` (YOLOv8 operator lock/track), `depth_avoider.py`
(RealSense depth to obstacle avoidance), `go2_speaker.py` (robot head-speaker
announcements).

---

## Requirements

**Hardware:** Unitree Go2 EDU + Intel RealSense D4xx, onboard Jetson.

**Software:** Ubuntu 20.04, Python 3.8, `unitree_sdk2py` (Unitree SDK), and:

```bash
sudo apt install portaudio19-dev            # for the robot-speaker audio dep
pip3 install --user -r requirements.txt
```

---

## Usage

```bash
# gesture console (robot stands up; control it with hand gestures)
python3 go2_master.py eth0

# desktop control GUI
python3 go2_control_gui.py

# follow a person (open palm = STOP)
python3 follow_depth_avoid.py eth0
```

The robot is controlled over `eth0` (default `192.168.123.18/24`). Keep a hand
on the remote - **L2+B is the physical e-stop**.

---

## Notes

- `go2_master.py` speaks each action through the robot's built-in head speaker
  via WebRTC (the Jetson's own audio jack has no working output).
- Gesture and follow modes use the RealSense camera; run them one at a time
  (the Jetson does not have the RAM to run both perception stacks at once).

## License

MIT - see [LICENSE](LICENSE). Built on the Unitree SDK and open-source
MediaPipe / Ultralytics / RealSense projects, which retain their own licenses.
