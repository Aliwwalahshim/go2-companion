"""
go2_master.py - Unitree Go2 console: one tracked hand does EVERYTHING -
                finger gestures pick the MODE, finger gestures issue the
                COMMANDS.

    python3 go2_master.py --reference       # print the full map, no robot
    python3 go2_master.py --dry-run         # camera + gestures, robot idle
    python3 go2_master.py eth0              # for real

This is a rewrite of the previous go2_master.py. Back the old one up
(cp go2_master.py go2_master_old.py) before replacing it on the Jetson.

DESIGN: ONE TABLE, THREE USES
-----------------------------
BINDINGS below is the single source of truth. It drives the dispatcher,
prints the reference, AND draws the on-screen menu - so the menu can never
disagree with what the robot actually does. Add a row and all three update.

WHY ONE HAND FOR EVERYTHING
----------------------------
Earlier versions tracked both hands and used arm poses for the menu. Two
tracked hands meant two, sometimes disagreeing, finger counts on every
frame - whichever hand MediaPipe happened to return first "won"
arbitrarily, so a held gesture could flicker between two readings and no
hold timer could ever complete cleanly. gesture_detector.py now tracks
exactly ONE hand (`max_num_hands=1`): when several hands are visible,
MediaPipe's own confidence ranking picks the best one, and that is the
only hand ever considered. There is nothing left to disagree with.

Finger counts on that one hand now drive the menu too: 1 finger, 2
(peace), 3, 4, thumbs-up, thumbs-down and a fist pick the mode; the same
vocabulary of finger gestures issues commands once inside a mode. The only gesture left
on the ARM channel is T_POSE, the global "back to the menu" - it stays a
big, whole-body arm pose on purpose, see SAFETY below.

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
1. ONE HAND ONLY. gesture_detector.py tracks a single hand
   (max_num_hands=1) instead of two, so two hands in frame can no longer
   produce conflicting finger counts.
2. MENU NOW USES FINGERS, NOT ARMS. Holding up 1/2/3/4 fingers, a
   thumbs-up, thumbs-down or a fist picks the mode (see BINDINGS["MENU"]).
   Arm poses are no longer needed to navigate the menu - only T_POSE
   (return to menu) and OPEN_PALM (stop) remain global, arm/whole-hand
   safety gestures, reachable from every mode.
7. STAIRS mode added. Thumbs-down from MENU. Just 2 options: Classic walk
   ON (ClassicWalk(True) - the gait Unitree's remote manual calls "Stair
   Climbing Mode 1", confirmed live on this unit to climb a real staircase
   forwards and descend it backwards once selected) and Free walk
   (FreeWalk() - back to the normal gait). No AI-mode / motion_switcher
   involved - this is a plain gait switch like everything else in GAITS.
3. NO REVERSING. The old GESTURE mode mapped RIGHT_UP to Move(-WALK_SPEED,
   0, 0). The Go2 has no rear sensing, so backward speed is now clamped to
   zero everywhere.
4. RightFlip REMOVED. SportClient has FrontFlip, BackFlip and LeftFlip -
   there is no RightFlip. The old alias list failed silently.
5. DEAD ALIASES REMOVED. Handshake / ShakeHands / Love / Greet are not
   SportClient methods either. Every command here is checked against the
   live SportClient at startup and hidden if this firmware lacks it.
6. DEPTH GATE. Walking forward needs a live scan from the RealSense depth
   stream (depth_avoider.py) instead of the old lidar_node.py/lidar_client.py
   UDP link. Override with --no-lidar if you accept driving blind.
7. NO BARE `except:`. Failures are printed, not swallowed.

SAFETY
------
* OPEN PALM = STOP, in every mode, always.
* T-POSE = back to the menu, in every mode.
* Every trigger must be HELD before it fires; acrobatics need longer.
* L2+B on the remote is the physical e-stop. Keep the remote in hand.
"""

import argparse
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from gesture_detector import GestureDetector
from go2_speaker import Go2Speaker

# ---------------- tuning ----------------
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
MIRROR_IMAGE = True          # you see yourself as in a mirror

HOLD_MODE = 1.5              # seconds to hold a gesture to enter a mode
HOLD_EXIT = 3.0              # holding FIST to quit
HOLD_CMD = 0.8               # seconds to hold a hand gesture for a command
HOLD_TRICK = 2.5             # acrobatics: deliberately slow
COOLDOWN_CMD = 2.0
COOLDOWN_TRICK = 6.0
HANDSTAND_HOLD = 4.0         # seconds HandStand(True) holds before auto-OFF

WALK_SPEED = 0.30            # m/s forward. NEVER negative.
TURN_SPEED = 0.50            # rad/s
MOVE_INTERVAL = 0.06         # 16 Hz command rate while driving

# Hand tracking blinks for a frame or two at any realistic distance. Without
# a grace window every blink restarts the hold countdown and no command can
# ever complete.
GESTURE_GRACE = 0.40         # seconds a trigger survives a tracking dropout
# Arm poses that beat any hand gesture happening at the same time. Keep this
# set SMALL - only whole-body poses nobody makes by accident. T_POSE is the
# only arm gesture left, and it necessarily shows open palms, so without
# this it would lose the race to the incidental OPEN_PALM (global STOP)
# every time and could never fire.
DOMINANT_ARM = {"T_POSE"}

# =======================================================================
# THE TABLE - behaviour, reference and on-screen menu all come from here
# =======================================================================
class Bind:
    """One row: hold `trigger` for `hold` seconds -> do `kind`(`payload`)."""
    __slots__ = ("trigger", "label", "kind", "payload", "hold", "note")

    def __init__(self, trigger, label, kind, payload=None,
                 hold=HOLD_CMD, note=""):
        self.trigger = trigger      # arm pose or hand gesture name
        self.label = label          # shown in the menu
        self.kind = kind            # mode | cmd | drive | stop | menu | exit
        self.payload = payload
        self.hold = hold
        self.note = note

    def method(self):
        """The SportClient method this needs, or None."""
        if self.kind == "cmd":
            return self.payload[0]
        return None


# Always active, in every mode.
GLOBAL = [
    Bind("OPEN_PALM", "STOP everything", "stop", hold=0.3,
         note="learn this one first"),
    Bind("T_POSE", "back to the MENU", "menu", hold=1.0),
]

BINDINGS = {
    # Mode selection now runs on the SAME single-hand finger vocabulary as
    # in-mode commands - hold up fingers to choose, instead of an arm pose.
    "MENU": [
        Bind("ONE", "DRIVE mode", "mode", "DRIVE", HOLD_MODE,
             "1 finger"),
        Bind("PEACE", "ACTIONS mode", "mode", "ACTIONS", HOLD_MODE,
             "2 fingers"),
        Bind("THREE", "GAITS mode", "mode", "GAITS", HOLD_MODE,
             "3 fingers"),
        Bind("FOUR", "POSTURE mode", "mode", "POSTURE", HOLD_MODE,
             "4 fingers"),
        Bind("THUMBS_UP", "TRICKS mode (acrobatic)", "mode", "TRICKS",
             HOLD_MODE, "flips live in here"),
        Bind("THUMBS_DOWN", "STAIRS mode", "mode", "STAIRS", HOLD_MODE,
             "thumbs down"),
        Bind("FIST", "sit down and QUIT", "exit", None, HOLD_EXIT),
    ],

    # Continuous driving: the robot moves WHILE the gesture is held.
    "DRIVE": [
        Bind("PEACE", "walk forward", "drive", (WALK_SPEED, 0.0, 0.0),
             note="2 fingers"),
        Bind("POINT_LEFT", "turn left", "drive", (0.0, 0.0, TURN_SPEED)),
        Bind("POINT_RIGHT", "turn right", "drive", (0.0, 0.0, -TURN_SPEED)),
        Bind("THREE", "strafe left", "drive", (0.0, 0.25, 0.0),
             note="3 fingers"),
        Bind("FOUR", "strafe right", "drive", (0.0, -0.25, 0.0),
             note="4 fingers"),
        Bind("FIST", "stand still", "stop"),
        Bind("THUMBS_UP", "stand up", "cmd", ("StandUp",)),
        Bind("THUMBS_DOWN", "lie down", "cmd", ("StandDown",)),
    ],

    "ACTIONS": [
        Bind("ONE", "wave hello", "cmd", ("Hello",), note="1 finger up"),
        Bind("PEACE", "heart gesture", "cmd", ("Heart",)),
        Bind("THREE", "stretch", "cmd", ("Stretch",)),
        Bind("FOUR", "dance 1", "cmd", ("Dance1",)),
        Bind("THUMBS_UP", "dance 2", "cmd", ("Dance2",)),
        Bind("THUMBS_DOWN", "scrape", "cmd", ("Scrape",)),
        Bind("FIST", "happy / content", "cmd", ("Content",)),
    ],

    "GAITS": [
        Bind("ONE", "static walk (slow, stable)", "cmd", ("StaticWalk",)),
        Bind("PEACE", "trot / run", "cmd", ("TrotRun",)),
        Bind("THREE", "free walk", "cmd", ("FreeWalk",)),
        Bind("FOUR", "classic walk ON", "cmd", ("ClassicWalk", True)),
        Bind("THUMBS_UP", "cross-step ON", "cmd", ("CrossStep", True)),
        Bind("THUMBS_DOWN", "cross-step OFF", "cmd", ("CrossStep", False)),
        Bind("FIST", "balanced stand", "cmd", ("BalanceStand",)),
    ],

    # Just the 2 gaits that matter for stairs. Confirmed live on this unit:
    # the handheld remote's RIGHT(long)+START combo (Unitree calls it
    # "Stair Climbing Mode 1" in the printed manual - up forward, down
    # backward) visibly switches the robot to Classic gait, and it climbed
    # a real staircase in that state. ClassicWalk(True) is the plain
    # SportClient call for that same gait - no AI mode / motion_switcher
    # involved at all, which is why the motion_switcher route in
    # stair_mode.py kept failing (error 7004): this was never gated behind
    # a mode switch, it's just a gait selection like any other in GAITS.
    # This has been confirmed to work through the remote's own trigger;
    # calling ClassicWalk(True) here is believed to select the identical
    # gait, but has not yet been separately confirmed climbing a staircase
    # with ClassicWalk(True) sent purely from this console (as opposed to
    # the remote combo) - test that before trusting it unattended.
    "STAIRS": [
        Bind("ONE", "classic walk ON - for stairs (up forward, "
             "down backward)", "cmd", ("ClassicWalk", True),
             note="1 finger"),
        Bind("PEACE", "free walk - normal gait", "cmd", ("FreeWalk",),
             note="2 fingers"),
    ],

    "POSTURE": [
        Bind("THUMBS_UP", "stand up", "cmd", ("StandUp",)),
        Bind("THUMBS_DOWN", "lie down", "cmd", ("StandDown",)),
        Bind("ONE", "sit", "cmd", ("Sit",)),
        Bind("PEACE", "rise from sitting", "cmd", ("RiseSit",)),
        Bind("THREE", "recover after a fall", "cmd", ("RecoveryStand",)),
        Bind("FOUR", "pose mode ON", "cmd", ("Pose", True)),
        Bind("FIST", "damp (joints relax)", "cmd", ("Damp",),
             note="the robot SAGS - only on the ground"),
    ],

    # Acrobatics: longer hold, longer cooldown, spoken warning.
    "TRICKS": [
        Bind("THUMBS_UP", "FRONT FLIP", "cmd", ("FrontFlip",), HOLD_TRICK,
             "large clear area"),
        Bind("THUMBS_DOWN", "BACK FLIP", "cmd", ("BackFlip",), HOLD_TRICK,
             "large clear area"),
        Bind("POINT_LEFT", "LEFT FLIP", "cmd", ("LeftFlip",), HOLD_TRICK,
             "no RightFlip exists in the SDK"),
        Bind("PEACE", "front jump", "cmd", ("FrontJump",), HOLD_TRICK),
        Bind("THREE", "front pounce", "cmd", ("FrontPounce",), HOLD_TRICK),
        Bind("FOUR", "handstand (4s, auto-off)", "cmd", ("HandStand", True),
             HOLD_TRICK),
        Bind("FIST", "handstand OFF", "cmd", ("HandStand", False), HOLD_TRICK),
    ],
}

MODE_ORDER = ["MENU", "DRIVE", "ACTIONS", "GAITS", "STAIRS", "POSTURE", "TRICKS"]
TRICK_MODES = {"TRICKS"}


# =======================================================================
# Reference printing
# =======================================================================
def print_reference(available=None):
    """The full map. `available` = set of SportClient methods, or None to
    print everything regardless of firmware."""
    def ok(b):
        m = b.method()
        return available is None or m is None or m in available

    print("\n" + "=" * 70)
    print(" GO2 CONSOLE - GESTURE REFERENCE (one tracked hand)")
    print("=" * 70)
    print("\nALWAYS ACTIVE (any mode):")
    for b in GLOBAL:
        print("   %-14s  %-28s %s" % (b.trigger, b.label,
                                      "(%s)" % b.note if b.note else ""))

    for mode in MODE_ORDER:
        rows = BINDINGS[mode]
        shown = [b for b in rows if ok(b)]
        hidden = [b for b in rows if not ok(b)]
        print("\n%s  [HAND GESTURES]" % mode)
        print("-" * 70)
        for b in shown:
            call = ""
            if b.kind == "cmd":
                call = "SportClient.%s(%s)" % (
                    b.payload[0],
                    ", ".join(repr(a) for a in b.payload[1:]))
            elif b.kind == "drive":
                vx, vy, vz = b.payload
                call = "Move(%.2f, %.2f, %.2f) while held" % (vx, vy, vz)
            elif b.kind == "mode":
                call = "-> %s mode" % b.payload
            print("   %-14s %-28s %-30s %.1fs" % (b.trigger, b.label, call, b.hold))
            if b.note:
                print("   %-14s    ^ %s" % ("", b.note))
        for b in hidden:
            print("   %-14s %-28s NOT ON THIS FIRMWARE - hidden"
                  % (b.trigger, b.label))
    print("\n" + "=" * 70)
    print("Hand gestures: 1=index up  2=peace  3=three  4=four  5=open palm")
    print("               thumbs up / thumbs down / fist / point left|right")
    print("               (tracked on ONE hand only)")
    print("Arm pose:      T_POSE  -  back to the MENU, from any mode")
    print("=" * 70 + "\n")


# =======================================================================
# Speech
# =======================================================================
class SpeechWorker(threading.Thread):
    """Speaks locally on the Jetson via espeak.

    The Go2 Python SDK has no TTS API - only the A2 and G1 product lines do
    (a2/audio, g1/audio). go2/vui/vui_client.py only controls the status-LED
    ring's switch/volume/brightness, it cannot speak text. So this is the
    Jetson's own speaker output, not a sound coming from the robot chassis.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=4)
        self.enabled = shutil.which("espeak") is not None
        if not self.enabled:
            print("[speech] espeak not found on PATH - text only.")
        self._busy = threading.Event()
        self.start()

    def say(self, text):
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def wait_idle(self, timeout=3.0):
        """Block until the queue is drained and nothing is playing, or
        until timeout. This is a daemon thread, so it dies the instant the
        main thread exits - call this right before quitting so the last
        announcement (e.g. "Console stopped") actually gets spoken instead
        of being silently killed mid-utterance or before it even starts."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.q.empty() and not self._busy.is_set():
                return
            time.sleep(0.05)

    def run(self):
        while True:
            text = self.q.get()
            self._busy.set()
            if self.enabled:
                # This sink auto-suspends when idle, and the first playback
                # after a suspend is sometimes clipped/lost before the
                # hardware finishes waking up. Force it awake and give it a
                # beat before every utterance so a quiet gap between
                # announcements can't silently eat the next one.
                try:
                    subprocess.run(["pactl", "suspend-sink", "@DEFAULT_SINK@", "0"],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=1.0)
                except Exception:
                    pass
                time.sleep(0.15)
                try:
                    subprocess.run(["espeak", text], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=6.0)
                except Exception as exc:
                    print("[speech] espeak failed: %s" % exc)
            self._busy.clear()


# Created in main() once we know we are actually running (not --reference),
# because Go2Speaker() spawns a worker process and opens a WebRTC connection
# to the robot - we do not want that side effect just to print the reference
# table. See main(). Falls back to espeak on the Jetson on its own if the
# robot speaker channel cannot be reached.
_speech = None


def announce(text, speech=None):
    print("[console] %s" % text)
    if _speech is not None:
        _speech.say(speech or text)


# =======================================================================
# Hold tracking
# =======================================================================
class HoldTracker:
    """A trigger must be held for `hold` seconds, then fires ONCE. It
    cannot fire again until it changes and comes back."""

    def __init__(self):
        self.trigger = None
        self.since = 0.0
        self.fired = None

    def update(self, trigger, now, hold):
        if trigger != self.trigger:
            self.trigger = trigger
            self.since = now
            if trigger is None:
                self.fired = None
            return None, 0.0
        if trigger is None or trigger == self.fired:
            return None, 0.0
        remaining = hold - (now - self.since)
        if remaining <= 0.0:
            self.fired = trigger
            return trigger, 0.0
        return None, remaining

    def reset(self):
        self.trigger = None
        self.fired = None


def pick_trigger(hand, arm, table, latched, latched_at, now):
    """Decide which gesture the console is acting on this frame.

    Returns (current, bind, latched, latched_at).

    Two rules make this robust:
      1. Prefer whichever candidate is actually BOUND in this mode, instead
         of blindly preferring the hand.
      2. Keep the last bound trigger alive for GESTURE_GRACE seconds, so a
         one-frame tracking blink does not restart the countdown.

    T_POSE is the one case where the arm channel must WIN outright: making
    a T-pose necessarily shows open palms, so on a frame where the hand
    channel reports the incidental OPEN_PALM (global STOP) instead of
    T_POSE, the arm reading is trusted over the hand reading.
    """
    order = ([arm] if arm in DOMINANT_ARM else []) + [hand, arm]
    for cand in order:
        if cand is not None and cand in table:
            return cand, table[cand], cand, now

    if latched is not None and (now - latched_at) < GESTURE_GRACE:
        bind = table.get(latched)
        if bind is not None:
            return latched, bind, latched, latched_at   # keep the old stamp

    return None, None, None, latched_at


class Stopped(Exception):
    """Raised by SIGTERM so the finally block runs."""


# =======================================================================
# Robot wrapper
# =======================================================================
class Robot:
    def __init__(self, interface, dry_run):
        self.dry = dry_run
        self.moving = False
        self.last_move = 0.0
        self.methods = set()
        self.sport = None
        if dry_run:
            print("[robot] DRY RUN - nothing is sent to the robot.")
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            self.methods = set(m for m in dir(SportClient)
                               if not m.startswith("_"))
            return

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        print("Initializing channel on %s ..." % interface)
        ChannelFactoryInitialize(0, interface)
        time.sleep(0.5)
        self.sport = SportClient()
        self.sport.SetTimeout(3.0)
        self.sport.Init()
        self.methods = set(m for m in dir(SportClient) if not m.startswith("_"))
        print("SportClient ready (%d methods)." % len(self.methods))

    def cmd(self, name, *args):
        if name not in self.methods:
            print("[robot] %s is not on this firmware." % name)
            return False
        if self.dry:
            print("   [dry-run] %s%r" % (name, args))
            return True
        try:
            getattr(self.sport, name)(*args)
            return True
        except Exception as exc:
            print("[robot] %s failed: %s: %s" % (name, type(exc).__name__, exc))
            return False

    def move(self, vx, vy, vyaw):
        vx = max(0.0, vx)                 # NEVER reverse
        now = time.time()
        if now - self.last_move < MOVE_INTERVAL:
            return
        self.last_move = now
        self.moving = True
        if self.dry:
            return
        try:
            self.sport.Move(vx, vy, vyaw)
        except Exception as exc:
            print("[robot] Move failed: %s" % exc)

    def stop_move(self):
        if not self.moving:
            return
        self.moving = False
        if self.dry:
            print("   [dry-run] StopMove()")
            return
        try:
            self.sport.StopMove()
        except Exception as exc:
            print("[robot] StopMove failed: %s" % exc)


# =======================================================================
# Camera
# =======================================================================
class Camera:
    """Wraps RealSense (color + depth) or a plain webcam (color only).

    Depth is only available on RealSense - depth_intrinsics/depth_scale stay
    None on webcam, and callers must treat that as "no depth avoidance".
    """

    def __init__(self, kind="auto", width=CAM_W, height=CAM_H):
        self.pipeline = None
        self.cap = None
        self.align = None
        self.depth_intrinsics = None
        self.depth_scale = None
        if kind in ("auto", "realsense"):
            try:
                import pyrealsense2 as rs
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, width, height,
                                  rs.format.bgr8, CAM_FPS)
                cfg.enable_stream(rs.stream.depth, width, height,
                                  rs.format.z16, CAM_FPS)
                self.align = rs.align(rs.stream.color)
                profile = self.pipeline.start(cfg)
                self.depth_intrinsics = profile.get_stream(rs.stream.color) \
                    .as_video_stream_profile().get_intrinsics()
                self.depth_scale = \
                    profile.get_device().first_depth_sensor().get_depth_scale()
                print("[camera] RealSense %dx%d (color + depth)."
                      % (width, height))
                return
            except Exception as exc:
                if kind == "realsense":
                    raise
                print("[camera] RealSense unavailable (%s) - webcam." % exc)
                self.pipeline = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No camera (tried RealSense and webcam 0).")
        print("[camera] webcam 0 (no depth - obstacle avoidance disabled).")

    def read(self):
        """Returns (color_frame, depth_frame). depth_frame is None on webcam
        or if a depth frame did not arrive this tick."""
        if self.pipeline is not None:
            frames = self.align.process(self.pipeline.wait_for_frames())
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            color = np.asanyarray(cf.get_data()) if cf else None
            depth = np.asanyarray(df.get_data()) if df else None
            return color, depth
        ok, frame = self.cap.read()
        return (frame if ok else None), None

    def close(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


# =======================================================================
# On-screen menu
# =======================================================================
def draw_menu(frame, mode, rows, active_trigger, remaining, banner, cd):
    h, w = frame.shape[:2]
    panel_w = 430
    x0 = w - panel_w - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 10), (w - 10, 60 + 34 * (len(rows) + 3)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    colour = (0, 80, 255) if mode in TRICK_MODES else (0, 255, 120)
    cv2.putText(frame, "MODE: %s" % mode, (x0 + 14, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2)

    y = 85
    for b in rows:
        hot = (b.trigger == active_trigger)
        col = (0, 255, 255) if hot else (210, 210, 210)
        txt = "%-12s %s" % (b.trigger, b.label)
        cv2.putText(frame, txt, (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, col, 2 if hot else 1)
        if hot and remaining > 0:
            bar = int(220 * (1.0 - remaining / max(b.hold, 1e-6)))
            cv2.rectangle(frame, (x0 + 14, y + 6), (x0 + 14 + bar, y + 11),
                          (0, 255, 255), -1)
        y += 34

    y += 6
    for b in GLOBAL:
        cv2.putText(frame, "%-12s %s" % (b.trigger, b.label), (x0 + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 200, 255), 1)
        y += 30

    if cd > 0:
        cv2.putText(frame, "cooldown %.1fs" % cd, (x0 + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if banner:
        cv2.putText(frame, banner, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 3)


# =======================================================================
# Main
# =======================================================================
def main():
    ap = argparse.ArgumentParser(description="Go2 gesture console")
    ap.add_argument("interface", nargs="?", default="eth0")
    ap.add_argument("--reference", action="store_true",
                    help="print the pose/gesture map and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="camera and gestures run; the robot is not commanded")
    ap.add_argument("--camera", default="auto",
                    choices=["auto", "realsense", "webcam"])
    ap.add_argument("--no-lidar", action="store_true",
                    help="allow walking without RealSense depth avoidance "
                         "(drives blind)")
    ap.add_argument("--debug", action="store_true",
                    help="print what is being detected, 4x a second")
    ap.add_argument("--hand-conf", type=float, default=0.5,
                    help="hand detection confidence 0-1 (default 0.5; lower "
                         "it if your hand is not being picked up)")
    ap.add_argument("--no-roi", action="store_true",
                    help="disable wrist-guided cropping and scan the whole "
                         "frame for hands (only useful up close)")
    ap.add_argument("--width", type=int, default=CAM_W)
    ap.add_argument("--height", type=int, default=CAM_H)
    args = ap.parse_args()

    if args.reference:
        print_reference()
        return 0

    # Speech: the robot's own head speaker over WebRTC, with automatic
    # espeak fallback. In --dry-run we deliberately keep speech on the local
    # Jetson (espeak) so a dry run needs no robot/network at all.
    global _speech
    _speech = Go2Speaker(enable_webrtc=not args.dry_run)

    robot = Robot(args.interface, args.dry_run)

    # hide anything this firmware does not have
    for mode in BINDINGS:
        keep = []
        for b in BINDINGS[mode]:
            m = b.method()
            if m is None or m in robot.methods:
                keep.append(b)
            else:
                print("[console] hiding '%s' - %s absent on this firmware."
                      % (b.label, m))
        BINDINGS[mode] = keep

    print_reference(robot.methods)

    cam = Camera(args.camera, args.width, args.height)
    det = GestureDetector(use_pose=True, mirrored=MIRROR_IMAGE,
                          hand_conf=args.hand_conf, use_roi=not args.no_roi)

    # Depth avoidance replaces the old lidar_node.py/LidarClient UDP link:
    # the RealSense depth stream doubles as the "lidar" and forward motion
    # is gated on it the same way forward motion used to be gated on a scan.
    depth_av = None
    if not args.no_lidar:
        if cam.pipeline is not None:
            from depth_avoider import DepthAvoider
            depth_av = DepthAvoider(cam.depth_intrinsics, mirror=MIRROR_IMAGE,
                                    depth_scale=cam.depth_scale)
            print("[depth-avoid] RealSense depth avoidance ready - "
                  "forward motion is protected.")
        else:
            print("[depth-avoid] no RealSense depth stream (webcam in use).")
            print("        Walking forward will be refused.")
            print("        Re-run with --no-lidar to drive blind.")
    else:
        print("[depth-avoid] DISABLED by --no-lidar. The robot will walk BLIND.")

    def on_term(signum, frame):
        raise Stopped()
    signal.signal(signal.SIGTERM, on_term)

    if not args.dry_run:
        announce("Standing up. Keep the area clear.")
        robot.cmd("StandUp")
        time.sleep(4.0)
        robot.cmd("FreeWalk")
        time.sleep(1.0)

    mode = "MENU"
    hold = HoldTracker()
    last_fire = 0.0
    banner, banner_until = "", 0.0
    latched, latched_at = None, 0.0      # grace window for hand dropouts
    last_debug = 0.0
    handstand_until = 0.0        # HandStand(True) deadline; 0.0 = not active
    active_drive = None          # label of the drive in progress, so we
                                 # announce it once when it starts (not per frame)

    announce("Ready. Menu mode.")

    # Pre-upload every phrase this console can speak so the FIRST time each
    # is triggered it plays instantly instead of pausing ~4s to upload. Done
    # in the background, after the startup lines, so it never delays them;
    # clips persist on the robot, so this is a no-op on later runs. Built
    # from the live bindings so it stays in sync with what actually gets said.
    if _speech is not None:
        _phrases = set([
            "Standing up. Keep the area clear.", "Ready. Menu mode.",
            "Handstand done", "Stopped.", "Back to the menu.",
            "Tricks mode. Clear the area.", "Quitting. Lying down.",
            "Console stopped.",
        ])
        for _m in BINDINGS:
            _phrases.add("%s mode." % _m.title())
            for _b in BINDINGS[_m]:
                if _b.label:
                    _phrases.add(_b.label)
        _speech.prewarm(_phrases)

    print("\nESC / q / Ctrl+Q quits (robot lies down). "
          "L2+B on the remote is the e-stop.\n")

    try:
        while True:
            frame, depth = cam.read()
            if frame is None:
                continue
            if MIRROR_IMAGE:
                frame = cv2.flip(frame, 1)

            res = det.update(frame)
            now = time.time()
            rows = BINDINGS[mode]
            cooldown = COOLDOWN_TRICK if mode in TRICK_MODES else COOLDOWN_CMD
            cd_left = max(0.0, cooldown - (now - last_fire))

            table = {b.trigger: b for b in GLOBAL}
            table.update({b.trigger: b for b in rows})

            current, bind, latched, latched_at = pick_trigger(
                res.hand_gesture, res.arm_gesture, table,
                latched, latched_at, now)

            fired, remaining = hold.update(
                current if bind else None, now, bind.hold if bind else 1.0)

            if args.debug:
                if now - last_debug >= 0.25:
                    last_debug = now
                    print("[debug] fps=%4.1f src=%-4s roi=%d hands=%d "
                          "fingers=%d hand=%-11s arm=%-12s -> %-11s hold=%.2f"
                          % (res.fps, res.source, len(res.rois),
                             len(res.hands), res.fingers,
                             res.hand_gesture or "-", res.arm_gesture or "-",
                             current or "-", remaining))

            # ---- continuous driving happens on the LEVEL signal ----
            driving = False
            if mode == "DRIVE" and res.stable:
                db = next((b for b in rows
                           if b.trigger == res.stable and b.kind == "drive"), None)
                if db is not None:
                    vx, vy, vyaw = db.payload
                    if vx > 0.01:
                        if depth_av is None:
                            banner = "walk refused - no depth avoidance"
                            banner_until = now + 1.5
                            vx = 0.0
                        else:
                            adj = depth_av.compute(depth)
                            if not adj.have_scan:
                                banner = "walk refused - no scan"
                                banner_until = now + 1.5
                                vx = 0.0
                            elif adj.block_forward:
                                banner = "obstacle ahead"
                                banner_until = now + 1.5
                                vx = 0.0
                            else:
                                vx *= adj.fwd_scale
                                vy = max(-0.3, min(0.3, vy + adj.lat))
                    if vx > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
                        robot.move(vx, vy, vyaw)
                        driving = True
                        # speak the drive ONCE when it starts (not every frame)
                        if active_drive != db.label:
                            announce(db.label)
                            active_drive = db.label
            if not driving:
                robot.stop_move()
                active_drive = None

            # ---- HandStand auto-revert: HandStand(True) is a self-timed
            # trick, not a held-forever pose - come back down on our own
            # after HANDSTAND_HOLD seconds regardless of what else the
            # console is doing, so the robot never gets left balancing.
            if handstand_until and now >= handstand_until:
                handstand_until = 0.0
                ok = robot.cmd("HandStand", False)
                banner = "handstand OFF" if ok else "handstand OFF FAILED"
                banner_until = now + 2.0
                announce("Handstand done.", "Handstand done")

            # ---- one-shot triggers ----
            if fired and bind is not None:
                if bind.kind == "stop":
                    robot.stop_move()
                    robot.cmd("StopMove")
                    hold.reset()
                    banner = "STOP"
                    banner_until = now + 1.5
                    announce("Stopped.")

                elif bind.kind == "menu":
                    robot.stop_move()
                    mode = "MENU"
                    hold.reset()
                    det.reset()
                    banner = "MENU"
                    banner_until = now + 1.5
                    announce("Back to the menu.")

                elif bind.kind == "mode":
                    robot.stop_move()
                    mode = bind.payload
                    hold.reset()
                    det.reset()
                    banner = mode
                    banner_until = now + 2.0
                    if mode in TRICK_MODES:
                        announce("Tricks mode. Clear the area around the robot.",
                                 "Tricks mode. Clear the area.")
                    else:
                        announce("%s mode." % mode.title())

                elif bind.kind == "exit":
                    announce("Quitting. Lying down.")
                    if _speech is not None:
                        _speech.wait_idle(3.0)
                    robot.stop_move()
                    robot.cmd("StandDown")
                    time.sleep(2.0)
                    break

                elif bind.kind == "cmd":
                    if cd_left > 0:
                        banner = "cooldown %.1fs" % cd_left
                        banner_until = now + 1.0
                    else:
                        robot.stop_move()
                        ok = robot.cmd(*bind.payload)
                        last_fire = now
                        banner = bind.label if ok else "%s FAILED" % bind.label
                        banner_until = now + 2.5
                        announce("%s%s" % (bind.label,
                                           "" if ok else " - not accepted"))
                        if bind.payload == ("HandStand", True):
                            handstand_until = now + HANDSTAND_HOLD if ok else 0.0
                        elif bind.payload == ("HandStand", False):
                            handstand_until = 0.0

                elif bind.kind == "drive":
                    pass        # handled continuously above

            # ---- HUD ----
            det.draw(frame, res)
            if not res.hands:
                cv2.putText(frame, "NO HAND DETECTED - show your hand",
                            (30, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            draw_menu(frame, mode, rows, current,
                      remaining, banner if now < banner_until else "", cd_left)
            if args.dry_run:
                cv2.putText(frame, "DRY RUN", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow("Go2 Console", frame)
            # ESC, 'q', or Ctrl+Q (keycode 17) all quit AND lie the robot
            # down, same as the FIST exit gesture - never leave it standing
            # unattended on quit.
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), 17):
                announce("Quitting. Lying down.")
                if _speech is not None:
                    _speech.wait_idle(3.0)
                robot.stop_move()
                robot.cmd("StandDown")
                time.sleep(2.0)
                break

    except (KeyboardInterrupt, Stopped):
        print("\nInterrupted.")
    finally:
        robot.stop_move()
        robot.cmd("StopMove")
        det.close()
        cam.close()
        cv2.destroyAllWindows()
        announce("Console stopped.")
        if _speech is not None:
            _speech.wait_idle(3.0)
            _speech.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

