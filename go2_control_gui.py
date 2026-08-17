"""
go2_control_gui.py - Dark, modern PyQt5 control panel for the Go2.

Live camera feed + mouse-driven virtual joystick + keyboard driving +
one-click LiDAR-avoidance toggle for stair climbing, all in one window.

CONTROLS
--------
Mouse:
    Drag the joystick to walk / strafe. Drag the rotate dial left/right
    to turn in place (distance from centre = turn speed, springs back
    to centre and stops on release) - or hold the FULL SPEED rotate
    buttons for a fixed-rate spin.
    Click LIDAR AVOID to flip between normal walking (avoidance ON,
    Free gait) and climb mode (avoidance OFF, Classic gait) - same
    combo used in follow_built_in_lidar.py, confirmed on this unit to
    climb real stairs.
    Click ALL COMMANDS for every other SportClient action (tricks,
    gait modes, dances, flips) not on the main panel.
    Click RealSense / Built-in to pick which camera feeds the window -
    RealSense is the USB depth camera (pyrealsense2, local, low latency);
    Built-in is the Go2's own front camera (VideoClient over the DDS
    channel, same source as camera_opencv.py, JPEG-encoded so a little
    softer/more compressed). Switching restarts the video thread, so
    expect a brief freeze.

Keyboard (window must have focus):
    W / S       forward / backward
    A / D       strafe left / right
    Q / E       rotate left / right
    Space       stop (zero velocity, stays standing)
    Esc         E-STOP (zero velocity + low crouch stance)
    C           toggle LIDAR AVOID / climb mode
    F11         toggle fullscreen
    Ctrl+Q      quit (crouches the robot down first)

RUN (one terminal, NOT foxy-sourced):
    python3 go2_control_gui.py eth0

Remote in hand. L2 + B is the physical e-stop. This program also lowers
the robot into a low, stable crouch (StandDown) on its own whenever it
stops - window closed, Esc, Ctrl+Q, or a crash. That's a deliberately
different pose than the dedicated Sit button: StandDown is the more
stable one to leave the robot in unattended.
"""

import signal
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSizePolicy, QScrollArea
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.go2.video.video_client import VideoClient

# ================= TUNING =================
CAM_W, CAM_H, CAM_FPS = 848, 480, 30
FWD_MAX, LAT_MAX, ROT_MAX = 0.6, 0.5, 0.8
ACC, DEC = 1.2, 2.2          # slew rates, units/sec
ROT_ACC, ROT_DEC = 2.5, 3.5
LOOP_HZ = 20.0
SEND_EPS = 0.02

ACCENT = "#00d4ff"
ACCENT_DIM = "#0a3a45"
BG = "#12141c"
PANEL = "#1a1d29"
PANEL2 = "#232838"
TEXT = "#e6e8ef"
SUBTEXT = "#8a8fa3"
DANGER = "#ff4d5e"
OK_GREEN = "#33e08a"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def slew(cur, target, acc, dec, dt):
    if target > cur:
        rate = acc if cur >= 0 else dec
        return min(cur + rate * dt, target)
    rate = acc if cur <= 0 else dec
    return max(cur - rate * dt, target)


# ---------------------------------------------------------------- camera
class CameraThread(QThread):
    """Captures from either camera, one at a time. 'realsense' uses the
    depth camera's color stream (pyrealsense2, local USB). 'builtin' uses
    the Go2's own front camera over the DDS channel (VideoClient,
    JPEG-encoded samples) - same source as camera_opencv.py."""
    frame_ready = pyqtSignal(QImage)

    def __init__(self, source="realsense", video_client=None):
        super().__init__()
        self._running = True
        self.source = source
        self.video_client = video_client
        self.hud = {}   # shared read-only dict of small status strings/numbers

    def run(self):
        if self.source == "builtin":
            self._run_builtin()
        else:
            self._run_realsense()

    def _run_realsense(self):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        try:
            pipeline.start(cfg)
        except Exception as exc:
            print("[camera] realsense failed to start: %s" % exc)
            return
        try:
            for s in pipeline.get_active_profile().get_device().query_sensors():
                try:
                    name = s.get_info(rs.camera_info.name)
                except Exception:
                    continue
                if ("RGB" in name or "Color" in name) and \
                        s.supports(rs.option.auto_exposure_priority):
                    s.set_option(rs.option.auto_exposure_priority, 0)
        except Exception:
            pass

        last_t, n, fps = time.time(), 0, 0.0
        try:
            while self._running:
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                frame = np.asanyarray(cf.get_data())

                n += 1
                now = time.time()
                if now - last_t >= 0.5:
                    fps = n / (now - last_t)
                    n, last_t = 0, now

                self._draw_hud(frame, fps)
                self._emit_frame(frame)
        finally:
            pipeline.stop()

    def _run_builtin(self):
        if self.video_client is None:
            print("[camera] built-in video client not available")
            return
        last_t, n, fps = time.time(), 0, 0.0
        while self._running:
            try:
                code, data = self.video_client.GetImageSample()
            except Exception as exc:
                print("[camera] built-in GetImageSample failed: %s" % exc)
                time.sleep(0.3)
                continue
            if code != 0 or not data:
                time.sleep(0.05)
                continue
            arr = np.frombuffer(bytes(data), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            n += 1
            now = time.time()
            if now - last_t >= 0.5:
                fps = n / (now - last_t)
                n, last_t = 0, now

            self._draw_hud(frame, fps)
            self._emit_frame(frame)

    def _emit_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.frame_ready.emit(qimg)

    def _draw_hud(self, frame, fps):
        h = self.hud
        cv2.putText(frame, "FPS %.0f" % fps, (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 1)
        mode = h.get("mode", "-")
        mcol = (77, 224, 160) if mode == "NORMAL" else (0, 165, 255)
        cv2.putText(frame, "MODE %s" % mode, (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mcol, 1)
        cv2.putText(frame, "f=%+.2f l=%+.2f r=%+.2f" %
                    (h.get("fwd", 0.0), h.get("lat", 0.0), h.get("rot", 0.0)),
                    (14, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 0), 1)

    def stop(self):
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------- joystick
class Joystick(QWidget):
    """Draggable virtual joystick. Emits normalized (x, y) in [-1, 1],
    x = strafe right, y = forward. Snaps back to centre on release."""
    moved = pyqtSignal(float, float)

    def __init__(self, diameter=170, parent=None):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.knob_r = diameter * 0.16
        self.pos = QPoint(int(self.radius), int(self.radius))
        self.dragging = False

    def _clamp_to_circle(self, p):
        c = QPoint(int(self.radius), int(self.radius))
        dx, dy = p.x() - c.x(), p.y() - c.y()
        dist = (dx * dx + dy * dy) ** 0.5
        maxr = self.radius - self.knob_r
        if dist > maxr and dist > 0:
            dx, dy = dx * maxr / dist, dy * maxr / dist
        return QPoint(int(c.x() + dx), int(c.y() + dy))

    def mousePressEvent(self, ev):
        self.dragging = True
        self.pos = self._clamp_to_circle(ev.pos())
        self._emit()
        self.update()

    def mouseMoveEvent(self, ev):
        if self.dragging:
            self.pos = self._clamp_to_circle(ev.pos())
            self._emit()
            self.update()

    def mouseReleaseEvent(self, ev):
        self.dragging = False
        self.pos = QPoint(int(self.radius), int(self.radius))
        self.moved.emit(0.0, 0.0)
        self.update()

    def _emit(self):
        maxr = self.radius - self.knob_r
        x = (self.pos.x() - self.radius) / maxr if maxr else 0.0
        y = -(self.pos.y() - self.radius) / maxr if maxr else 0.0
        self.moved.emit(clamp(x, -1, 1), clamp(y, -1, 1))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT_DIM), 2))
        p.setBrush(QBrush(QColor(PANEL2)))
        p.drawEllipse(1, 1, self.diameter - 2, self.diameter - 2)
        p.setPen(QPen(QColor(SUBTEXT), 1, Qt.DashLine))
        c = self.diameter / 2
        p.drawLine(int(c), 8, int(c), int(self.diameter - 8))
        p.drawLine(8, int(c), int(self.diameter - 8), int(c))
        col = QColor(ACCENT) if self.dragging else QColor(SUBTEXT)
        p.setPen(QPen(col, 2))
        p.setBrush(QBrush(col.darker(140) if self.dragging else QColor(PANEL)))
        r = self.knob_r
        p.drawEllipse(self.pos, int(r), int(r))


# ---------------------------------------------------------------- rotate dial
class RotateDial(QWidget):
    """Horizontal drag strip for turning in place with the mouse. Drag
    left/right; distance from centre sets rotation speed. Springs back
    to centre (rotation stops) on release, same feel as the joystick."""
    moved = pyqtSignal(float)

    def __init__(self, width=280, height=54, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.w, self.h = width, height
        self.knob_r = height * 0.42
        self.x = width / 2.0
        self.dragging = False

    def _clamp_x(self, x):
        lo, hi = self.knob_r, self.w - self.knob_r
        return clamp(x, lo, hi)

    def mousePressEvent(self, ev):
        self.dragging = True
        self.x = self._clamp_x(ev.pos().x())
        self._emit()
        self.update()

    def mouseMoveEvent(self, ev):
        if self.dragging:
            self.x = self._clamp_x(ev.pos().x())
            self._emit()
            self.update()

    def mouseReleaseEvent(self, ev):
        self.dragging = False
        self.x = self.w / 2.0
        self.moved.emit(0.0)
        self.update()

    def _emit(self):
        half = (self.w - 2 * self.knob_r) / 2.0
        v = (self.x - self.w / 2.0) / half if half else 0.0
        self.moved.emit(clamp(v, -1, 1))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT_DIM), 2))
        p.setBrush(QBrush(QColor(PANEL2)))
        p.drawRoundedRect(1, 1, self.w - 2, self.h - 2, self.h / 2, self.h / 2)
        p.setPen(QPen(QColor(SUBTEXT), 1, Qt.DashLine))
        cx = self.w / 2
        p.drawLine(int(cx), 6, int(cx), int(self.h - 6))
        f = p.font(); f.setPointSize(9); p.setFont(f)
        p.setPen(QColor(SUBTEXT))
        p.drawText(10, int(self.h / 2 + 4), "⟲")
        p.drawText(self.w - 22, int(self.h / 2 + 4), "⟳")
        col = QColor(ACCENT) if self.dragging else QColor(SUBTEXT)
        p.setPen(QPen(col, 2))
        p.setBrush(QBrush(col.darker(140) if self.dragging else QColor(PANEL)))
        r = self.knob_r
        p.drawEllipse(QPoint(int(self.x), int(self.h / 2)), int(r), int(r))


# ---------------------------------------------------------------- toggle
class ToggleSwitch(QPushButton):
    def __init__(self, text_on, text_off, parent=None):
        super().__init__(parent)
        self.text_on, self.text_off = text_on, text_off
        self.setCheckable(True)
        self.setFixedHeight(46)
        self.setCursor(Qt.PointingHandCursor)
        self.toggled.connect(self._refresh)
        self._refresh(False)

    def _refresh(self, checked):
        self.setText(self.text_on if checked else self.text_off)
        if checked:
            self.setStyleSheet("""
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff8a3d, stop:1 #ff4d5e); color: white;
                    border-radius: 10px; font-weight: 600; font-size: 14px; }
            """)
        else:
            self.setStyleSheet("""
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1fae6b, stop:1 #33e08a); color: #05130c;
                    border-radius: 10px; font-weight: 600; font-size: 14px; }
            """)
# ---------------------------------------------------------------- commands
# every extra SportClient action, beyond the core drive/stand/sit buttons
# already on the main panel. TRICKS take no arguments and fire once.
# GAIT_TOGGLES take a single bool flag - click turns the behaviour ON,
# click again turns it back OFF.
TRICKS = ["Hello", "Stretch", "Content", "Dance1", "Dance2", "Scrape",
          "FrontFlip", "FrontJump", "FrontPounce", "Heart", "LeftFlip",
          "BackFlip", "RiseSit", "Damp", "SwitchAvoidMode"]
GAIT_PLAIN = ["FreeWalk", "StaticWalk", "TrotRun"]
GAIT_TOGGLES = ["ClassicWalk", "FreeBound", "FreeJump", "FreeAvoid",
                "WalkUpright", "CrossStep", "HandStand"]


class CommandsDialog(QDialog):
    """Every other SportClient command, in one scrollable panel - the
    handful of quick-access buttons on the main window don't come close
    to covering the full API, so this is the 'everything else' drawer."""

    def __init__(self, robot, safe_call, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.safe_call = safe_call
        self.setWindowTitle("ALL COMMANDS")
        self.resize(560, 620)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        outer = QVBoxLayout(self)
        warn = QLabel("Some of these are acrobatic (flips, dances) - make "
                       "sure there's clear space around the robot.")
        warn.setObjectName("sub")
        warn.setWordWrap(True)
        outer.addWidget(warn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        lay.addWidget(self._section("GAITS", GAIT_PLAIN, toggle=False))
        lay.addWidget(self._section("GAIT MODES (toggle)", GAIT_TOGGLES, toggle=True))
        lay.addWidget(self._section("TRICKS & UTILITY", TRICKS, toggle=False))
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn)

    def _section(self, title, names, toggle):
        card = QFrame(); card.setObjectName("panel")
        v = QVBoxLayout(card)
        lbl = QLabel(title); lbl.setObjectName("sub")
        v.addWidget(lbl)
        grid = QGridLayout()
        for i, name in enumerate(names):
            b = QPushButton(name)
            if toggle:
                b.setCheckable(True)
                b.toggled.connect(lambda checked, n=name: self._call_flag(n, checked))
            else:
                b.clicked.connect(lambda _, n=name: self._call_plain(n))
            grid.addWidget(b, i // 3, i % 3)
        v.addLayout(grid)
        return card

    def _call_plain(self, name):
        self.safe_call(lambda: getattr(self.robot, name)(), name)

    def _call_flag(self, name, flag):
        self.safe_call(lambda: getattr(self.robot, name)(flag), "%s(%s)" % (name, flag))


# ---------------------------------------------------------------- main
class MainWindow(QMainWindow):
    def __init__(self, interface):
        super().__init__()
        self.setWindowTitle("GO2 CONTROL")
        self.resize(760, 480)
        self.setStyleSheet(self._stylesheet())

        self.interface = interface
        self.robot = None
        self.lidar_av = None
        self.video_client = None
        self.connected = False
        self.climb_mode = False
        self.camera_source = "realsense"

        self.jx = self.jy = 0.0          # joystick axes
        self.rot_hold = 0.0              # -1 / 0 / +1 from full-speed rotate buttons
        self.rot_dial = 0.0              # -1..1 from the mouse-drag rotate dial
        self.keys = set()
        self.cmd_fwd = self.cmd_lat = self.cmd_rot = 0.0
        self.moving = False
        self.last_loop = time.time()
        self.speed_scale = 1.0

        self._build_ui()
        self._connect_robot()

        self.cam = None
        self._start_camera(self.camera_source)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._control_loop)
        self.timer.start(int(1000 / LOOP_HZ))

        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------- UI ----------------
    def _stylesheet(self):
        return f"""
            QMainWindow {{ background: {BG}; }}
            QWidget {{ color: {TEXT}; font-family: 'DejaVu Sans', sans-serif; }}
            QFrame#panel {{ background: {PANEL}; border-radius: 14px; }}
            QLabel#title {{ font-size: 20px; font-weight: 700; color: {ACCENT}; }}
            QLabel#sub {{ color: {SUBTEXT}; font-size: 12px; }}
            QLabel#status {{ color: {TEXT}; font-size: 13px; }}
            QPushButton {{
                background: {PANEL2}; color: {TEXT}; border: 1px solid #333850;
                border-radius: 10px; padding: 8px; font-size: 13px; font-weight: 500;
            }}
            QPushButton:hover {{ background: #2b3148; border-color: {ACCENT}; }}
            QPushButton:pressed {{ background: {ACCENT_DIM}; }}
            QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; color: {ACCENT}; font-weight: 700; }}
            QPushButton#estop {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #d1293f, stop:1 #ff4d5e);
                color: white; font-weight: 800; font-size: 15px; border: none; border-radius: 12px;
            }}
            QPushButton#estop:hover {{ background: #ff6b78; }}
            QSlider::groove:horizontal {{ background: {PANEL2}; height: 6px; border-radius: 3px; }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 16px; margin: -6px 0; border-radius: 8px; }}
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---- left: video ----
        video_frame = QFrame()
        video_frame.setObjectName("panel")
        vl = QVBoxLayout(video_frame)
        vl.setContentsMargins(12, 12, 12, 12)
        header = QHBoxLayout()
        title = QLabel("GO2 CONTROL")
        title.setObjectName("title")
        self.conn_dot = QLabel("●")
        self.conn_dot.setStyleSheet(f"color: {DANGER}; font-size: 16px;")
        self.conn_text = QLabel("connecting...")
        self.conn_text.setObjectName("sub")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.conn_dot)
        header.addWidget(self.conn_text)
        vl.addLayout(header)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(300, 180)
        self.video_label.setStyleSheet(f"background: black; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("waiting for camera...")
        vl.addWidget(self.video_label, 1)

        legend = QLabel("W/S forward-back   A/D strafe   Q/E rotate   "
                         "Space stop   C climb toggle   F11 fullscreen   "
                         "Ctrl+Q quit   Esc E-STOP")
        legend.setObjectName("sub")
        legend.setWordWrap(True)
        vl.addWidget(legend)
        root.addWidget(video_frame, 3)

        # ---- right: controls ----
        side = QVBoxLayout()
        side.setSpacing(14)

        # status card
        status_card = QFrame(); status_card.setObjectName("panel")
        sc = QVBoxLayout(status_card)
        self.mode_label = QLabel("MODE: NORMAL (avoid ON)")
        self.mode_label.setObjectName("status")
        self.speed_label = QLabel("fwd=0.00  lat=0.00  rot=0.00")
        self.speed_label.setObjectName("status")
        sc.addWidget(self.mode_label)
        sc.addWidget(self.speed_label)
        side.addWidget(status_card)

        # joystick card
        joy_card = QFrame(); joy_card.setObjectName("panel")
        jc = QVBoxLayout(joy_card)
        jc.setAlignment(Qt.AlignCenter)
        jlabel = QLabel("DRIVE"); jlabel.setObjectName("sub")
        jlabel.setAlignment(Qt.AlignCenter)
        jc.addWidget(jlabel)
        self.joystick = Joystick()
        self.joystick.moved.connect(self._on_joystick)
        jrow = QHBoxLayout(); jrow.addStretch(); jrow.addWidget(self.joystick); jrow.addStretch()
        jc.addLayout(jrow)

        rlabel = QLabel("ROTATE - drag left/right"); rlabel.setObjectName("sub")
        rlabel.setAlignment(Qt.AlignCenter)
        jc.addWidget(rlabel)
        self.rotate_dial = RotateDial()
        self.rotate_dial.moved.connect(self._on_rotate_dial)
        drow = QHBoxLayout(); drow.addStretch(); drow.addWidget(self.rotate_dial); drow.addStretch()
        jc.addLayout(drow)

        rot_row = QHBoxLayout()
        self.btn_left = QPushButton("⟲ FULL SPEED")
        self.btn_right = QPushButton("FULL SPEED ⟳")
        for b, sign in ((self.btn_left, 1.0), (self.btn_right, -1.0)):
            b.pressed.connect(lambda s=sign: self._set_rot_hold(s))
            b.released.connect(lambda: self._set_rot_hold(0.0))
            rot_row.addWidget(b)
        jc.addLayout(rot_row)
        side.addWidget(joy_card)

        # speed slider
        speed_card = QFrame(); speed_card.setObjectName("panel")
        spc = QVBoxLayout(speed_card)
        sl_label = QLabel("SPEED"); sl_label.setObjectName("sub")
        spc.addWidget(sl_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(20, 100)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(
            lambda v: setattr(self, "speed_scale", v / 100.0))
        spc.addWidget(self.speed_slider)
        side.addWidget(speed_card)

        # climb toggle
        climb_card = QFrame(); climb_card.setObjectName("panel")
        cc = QVBoxLayout(climb_card)
        cl_label = QLabel("LIDAR AVOIDANCE  (off = stair climb mode)")
        cl_label.setObjectName("sub")
        cl_label.setWordWrap(True)
        cc.addWidget(cl_label)
        self.avoid_toggle = ToggleSwitch("CLIMB MODE (avoid OFF)", "LIDAR AVOID: ON")
        self.avoid_toggle.toggled.connect(self._on_toggle_climb)
        cc.addWidget(self.avoid_toggle)
        side.addWidget(climb_card)

        # camera source picker
        cam_card = QFrame(); cam_card.setObjectName("panel")
        camc = QVBoxLayout(cam_card)
        cam_label = QLabel("CAMERA SOURCE"); cam_label.setObjectName("sub")
        camc.addWidget(cam_label)
        cam_row = QHBoxLayout()
        self.btn_cam_rs = QPushButton("RealSense")
        self.btn_cam_builtin = QPushButton("Built-in")
        for b in (self.btn_cam_rs, self.btn_cam_builtin):
            b.setCheckable(True)
            cam_row.addWidget(b)
        self.btn_cam_rs.setChecked(True)
        self.btn_cam_rs.clicked.connect(lambda: self._switch_camera("realsense"))
        self.btn_cam_builtin.clicked.connect(lambda: self._switch_camera("builtin"))
        camc.addLayout(cam_row)
        side.addWidget(cam_card)

        # action buttons
        act_card = QFrame(); act_card.setObjectName("panel")
        ac = QGridLayout(act_card)
        actions = [
            ("Stand Up", self._act_standup), ("Stand Down", self._act_standdown),
            ("Balance Stand", self._act_balance), ("Sit", self._act_sit),
            ("Recovery Stand", self._act_recovery), ("Stop", self._act_stop),
        ]
        for i, (label, fn) in enumerate(actions):
            b = QPushButton(label)
            b.clicked.connect(fn)
            ac.addWidget(b, i // 2, i % 2)
        side.addWidget(act_card)

        all_cmds_btn = QPushButton("⋯  ALL COMMANDS")
        all_cmds_btn.setStyleSheet(f"""
            QPushButton {{ background: {PANEL2}; border: 1px solid {ACCENT}; color: {ACCENT};
                border-radius: 10px; padding: 10px; font-weight: 600; }}
            QPushButton:hover {{ background: {ACCENT_DIM}; }}
        """)
        all_cmds_btn.clicked.connect(self._open_all_commands)
        side.addWidget(all_cmds_btn)

        # e-stop
        estop = QPushButton("E - S T O P")
        estop.setObjectName("estop")
        estop.setFixedHeight(56)
        estop.clicked.connect(self._estop)
        side.addWidget(estop)

        side.addStretch()
        side_frame = QWidget()
        side_frame.setLayout(side)
        side_frame.setFixedWidth(280)

        # scrollable so the sidebar's own content height never forces the
        # whole window taller - lets the window shrink well below what all
        # the cards would need if stacked with no scrolling
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFixedWidth(300)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setWidget(side_frame)
        root.addWidget(side_scroll, 0)

    # ---------------- robot connection ----------------
    def _connect_robot(self):
        try:
            print("Init channel on %s ..." % self.interface)
            ChannelFactoryInitialize(0, self.interface)
            time.sleep(0.5)
            self.robot = SportClient()
            self.robot.SetTimeout(3.0)
            self.robot.Init()
            try:
                self.robot.StandUp()
            except Exception as exc:
                print("StandUp failed: %s" % exc)
            time.sleep(4.0)
            try:
                self.robot.BalanceStand()
            except Exception as exc:
                print("BalanceStand failed: %s" % exc)
            time.sleep(1.0)
            self._set_gait(free=True)

            self.lidar_av = ObstaclesAvoidClient()
            self.lidar_av.SetTimeout(3.0)
            self.lidar_av.Init()
            self.lidar_av.UseRemoteCommandFromApi(True)
            self.lidar_av.SwitchSet(True)
            time.sleep(0.3)

            try:
                self.video_client = VideoClient()
                self.video_client.SetTimeout(3.0)
                self.video_client.Init()
            except Exception as exc:
                print("[camera] built-in VideoClient init failed: %s" % exc)
                self.video_client = None

            self.connected = True
            self.conn_dot.setStyleSheet(f"color: {OK_GREEN}; font-size: 16px;")
            self.conn_text.setText("connected: %s" % self.interface)
        except Exception as exc:
            print("[robot] connect failed: %s" % exc)
            self.conn_text.setText("connect failed: %s" % exc)

    def _set_gait(self, free):
        try:
            if free:
                try:
                    self.robot.FreeWalk()
                except TypeError:
                    self.robot.FreeWalk(True)
            else:
                self.robot.ClassicWalk(True)
        except Exception as exc:
            print("[gait] switch failed: %s" % exc)

    # ---------------- camera ----------------
    def _start_camera(self, source):
        if self.cam is not None:
            self.cam.stop()
        self.cam = CameraThread(source=source, video_client=self.video_client)
        self.cam.frame_ready.connect(self._on_frame)
        self.cam.start()
        self.camera_source = source

    def _switch_camera(self, source):
        if source == self.camera_source:
            self.btn_cam_rs.setChecked(source == "realsense")
            self.btn_cam_builtin.setChecked(source == "builtin")
            return
        if source == "builtin" and self.video_client is None:
            print("[camera] built-in camera not available (VideoClient failed to init)")
            self.btn_cam_rs.setChecked(True)
            self.btn_cam_builtin.setChecked(False)
            return
        self.video_label.setText("switching camera...")
        self.btn_cam_rs.setChecked(source == "realsense")
        self.btn_cam_builtin.setChecked(source == "builtin")
        self._start_camera(source)

    # ---------------- input handlers ----------------
    def _on_joystick(self, x, y):
        self.jx, self.jy = x, y

    def _set_rot_hold(self, v):
        self.rot_hold = v

    def _on_rotate_dial(self, v):
        # dial reads positive when dragged right (toward the drawn "⟳"),
        # which must turn the robot right - same sign as the E key.
        self.rot_dial = -v

    def _on_toggle_climb(self, checked):
        self.climb_mode = checked
        if checked:
            self._set_gait(free=False)
            if self.lidar_av is not None:
                try:
                    self.lidar_av.SwitchSet(False)
                except Exception as exc:
                    print("[lidar-avoid] off failed: %s" % exc)
            self.mode_label.setText("MODE: CLIMB (avoid OFF, Classic gait)")
        else:
            self._set_gait(free=True)
            if self.lidar_av is not None:
                try:
                    self.lidar_av.SwitchSet(True)
                except Exception as exc:
                    print("[lidar-avoid] on failed: %s" % exc)
            self.mode_label.setText("MODE: NORMAL (avoid ON)")

    def keyPressEvent(self, ev):
        if ev.isAutoRepeat():
            return
        k = ev.key()
        if k == Qt.Key_Escape:
            self._estop()
            return
        if k == Qt.Key_Space:
            self._act_stop()
            return
        if k == Qt.Key_C:
            self.avoid_toggle.setChecked(not self.avoid_toggle.isChecked())
            return
        if k == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        if k == Qt.Key_Q and ev.modifiers() & Qt.ControlModifier:
            self.close()
            return
        self.keys.add(k)

    def keyReleaseEvent(self, ev):
        if ev.isAutoRepeat():
            return
        self.keys.discard(ev.key())

    # ---------------- control loop ----------------
    def _control_loop(self):
        now = time.time()
        dt = clamp(now - self.last_loop, 0.0, 0.2)
        self.last_loop = now

        kx = ky = krot = 0.0
        if Qt.Key_W in self.keys: ky += 1.0
        if Qt.Key_S in self.keys: ky -= 1.0
        if Qt.Key_D in self.keys: kx += 1.0    # D = right
        if Qt.Key_A in self.keys: kx -= 1.0    # A = left
        if Qt.Key_Q in self.keys: krot += 1.0
        if Qt.Key_E in self.keys: krot -= 1.0

        vx = clamp(self.jy + ky, -1, 1)
        # robot body frame: +y is LEFT, so "right" (D key / joystick-right /
        # dial-right) must map to NEGATIVE vy - hence the minus sign here.
        vy = clamp(-(self.jx + kx), -1, 1)
        vrot = clamp(self.rot_hold + self.rot_dial + krot, -1, 1)

        tgt_fwd = vx * FWD_MAX * self.speed_scale
        tgt_lat = vy * LAT_MAX * self.speed_scale
        tgt_rot = vrot * ROT_MAX * self.speed_scale

        self.cmd_fwd = slew(self.cmd_fwd, tgt_fwd, ACC, DEC, dt)
        self.cmd_lat = slew(self.cmd_lat, tgt_lat, ACC, DEC, dt)
        self.cmd_rot = slew(self.cmd_rot, tgt_rot, ROT_ACC, ROT_DEC, dt)

        self._send(self.cmd_fwd, self.cmd_lat, self.cmd_rot)
        self.speed_label.setText("fwd=%+.2f  lat=%+.2f  rot=%+.2f"
                                  % (self.cmd_fwd, self.cmd_lat, self.cmd_rot))
        self.cam.hud.update(fwd=self.cmd_fwd, lat=self.cmd_lat, rot=self.cmd_rot,
                             mode="CLIMB" if self.climb_mode else "NORMAL")

    def _send(self, fwd, lat, rot):
        if not self.connected:
            return
        client = self.lidar_av if (self.lidar_av is not None and not self.climb_mode) else self.robot
        try:
            if abs(fwd) > SEND_EPS or abs(lat) > SEND_EPS or abs(rot) > SEND_EPS:
                client.Move(fwd, lat, rot)
                self.moving = True
            elif self.moving:
                client.Move(0.0, 0.0, 0.0) if client is self.lidar_av else self.robot.StopMove()
                self.moving = False
        except Exception as exc:
            print("[robot] Move failed: %s" % exc)

    # ---------------- actions ----------------
    def _act_standup(self):
        self._safe_call(self.robot.StandUp, "StandUp")

    def _act_standdown(self):
        self._safe_call(self.robot.StandDown, "StandDown")

    def _act_balance(self):
        self._safe_call(self.robot.BalanceStand, "BalanceStand")

    def _act_sit(self):
        self._safe_call(self.robot.Sit, "Sit")

    def _act_recovery(self):
        self._safe_call(self.robot.RecoveryStand, "RecoveryStand")

    def _open_all_commands(self):
        if not self.connected:
            print("[commands] robot not connected yet")
            return
        dlg = CommandsDialog(self.robot, self._safe_call, self)
        dlg.exec_()

    def _act_stop(self):
        self.jx = self.jy = self.rot_hold = self.rot_dial = 0.0
        self.keys.clear()
        self.cmd_fwd = self.cmd_lat = self.cmd_rot = 0.0
        self._send(0.0, 0.0, 0.0)

    def _safe_call(self, fn, name):
        if not self.connected:
            return
        try:
            print("[robot] %s -> %s" % (name, fn()))
        except Exception as exc:
            print("[robot] %s failed: %s" % (name, exc))

    def _estop(self):
        self._act_stop()
        if self.connected:
            try:
                self.robot.StopMove()
            except Exception:
                pass
            time.sleep(0.3)
            # low crouched stance, not Sit - more stable to leave the robot
            # in in an emergency/on exit. The dedicated Sit button still
            # calls Sit() directly if that pose is wanted on purpose.
            self._safe_call(self.robot.StandDown, "StandDown")

    # ---------------- misc ----------------
    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def closeEvent(self, ev):
        self.timer.stop()
        self._estop()
        if self.lidar_av is not None:
            try:
                self.lidar_av.Move(0.0, 0.0, 0.0)
                self.lidar_av.SwitchSet(False)
                self.lidar_av.UseRemoteCommandFromApi(False)
            except Exception:
                pass
        self.cam.stop()
        ev.accept()


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    app = QApplication(sys.argv)
    win = MainWindow(interface)
    # Ctrl+C / SIGTERM must also trigger closeEvent (stop + Sit), not just
    # the window's own close button - Qt's event loop only sees a Python
    # signal when control returns to the interpreter, which the 20Hz
    # control-loop timer already guarantees happens often enough.
    signal.signal(signal.SIGINT, lambda *a: win.close())
    signal.signal(signal.SIGTERM, lambda *a: win.close())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
