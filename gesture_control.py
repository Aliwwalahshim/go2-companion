"""
gesture_detector.py - Hand + arm gesture recognition. PERCEPTION ONLY.

No robot control, no camera handling - you hand it a BGR frame, it tells
you what gesture it sees. Same seam as person_detector.py.

    det = GestureDetector()
    res = det.update(frame)         # every frame
    res.fingers                     # 0-5 on the dominant hand
    res.hand_gesture                # "THUMBS_UP", "OPEN_PALM", ...
    res.arm_gesture                 # "BOTH_ARMS_UP", "T_POSE", ...
    res.confirmed                   # a gesture that has been held long
                                    # enough to act on (edge-triggered,
                                    # fires ONCE) - this is what you use
                                    # to command a robot
    det.draw(frame, res)            # HUD overlay

WHY MEDIAPIPE AND NOT YOLO
--------------------------
YOLO-pose outputs 17 COCO body keypoints and has NO finger joints, so it
cannot count fingers or read a thumb direction at all. MediaPipe Hands
gives 21 landmarks per hand including every finger joint. For this job
MediaPipe is not the weaker option - it is the only one of the two that
can do it.

TWO BUGS THIS FILE FIXES (both were in Body_posses.py)
------------------------------------------------------
1. WRONG LANDMARK INDICES. Body_posses.py used pose landmarks 9 and 10 as
   "wrists". In MediaPipe Pose, 9 and 10 are the MOUTH CORNERS; the wrists
   are 15 and 16. So "arm up" was really asking "is your mouth above your
   shoulder", which sits on the boundary and flickers. Correct indices are
   used here.
2. DISTANCE-DEPENDENT THRESHOLDS. Fixed normalized offsets like 0.15 mean
   completely different things at 1 m and at 3 m from the camera. Every
   threshold here is expressed in SHOULDER WIDTHS (body) or HAND SPANS
   (hands), so it behaves the same at any distance.

ROBUSTNESS NOTES
----------------
* Finger extension is measured as "is the tip further from the wrist than
  the middle joint", not "is the tip higher in the image". That keeps
  working when the hand is rotated or sideways.
* The thumb is tested against the index knuckle instead, because it folds
  sideways rather than curling.
* Nothing fires until a gesture has been held for HOLD_FRAMES consecutive
  frames, and each gesture fires ONCE until it changes. A robot must not
  act on a single flickering frame.
"""

import time

import numpy as np

try:
    import mediapipe as mp
except Exception as exc:      # pragma: no cover - import guard
    mp = None
    _IMPORT_ERR = exc

# ---- MediaPipe Hands landmark ids (21 per hand) ----
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_NAMES = ["index", "middle", "ring", "pinky"]

# ---- MediaPipe Pose landmark ids (33 total) - THE CORRECT ONES ----
# 9 and 10 are the MOUTH CORNERS, not the wrists. This is the bug that
# made Body_posses.py's arm detection unreliable.
P_LEFT_SHOULDER, P_RIGHT_SHOULDER = 11, 12
P_LEFT_ELBOW, P_RIGHT_ELBOW = 13, 14
P_LEFT_WRIST, P_RIGHT_WRIST = 15, 16
P_LEFT_HIP, P_RIGHT_HIP = 23, 24

# ---- tuning ----
EXTEND_RATIO = 1.15      # tip must be this much further from the wrist
                         # than the PIP joint to count as extended
THUMB_RATIO = 1.20       # same idea for the thumb, measured from index MCP
THUMB_DIR_FRAC = 0.25    # thumb tip must be this far above/below the wrist,
                         # as a fraction of hand span, to read as up/down
THUMB_VERT_DOMINANCE = 1.0   # and its vertical travel must beat its sideways
                         # travel by this ratio, i.e. the thumb has to be
                         # within ~45 deg of vertical. Without this, a thumb
                         # just sticking out sideways reads as THUMBS_UP -
                         # an ambiguous thumb must never command the robot.
POINT_DIR_FRAC = 0.35    # how decisively the index must point sideways

ARM_UP_FRAC = 0.35       # wrist this far above the shoulder (shoulder widths)
ARM_DOWN_FRAC = 0.55     # wrist this far below the shoulder
T_POSE_Y_FRAC = 0.30     # wrist within this of shoulder height
T_POSE_X_FRAC = 0.65     # and this far out to the side

# WRIST-GUIDED CROPPING. Standing far enough back for the camera to see
# your whole body makes a hand ~40 px in a 1280 px frame, and MediaPipe's
# palm detector simply will not find it - you get fingers: 0 forever. The
# POSE model still locates the wrists at that distance, so we crop a box
# around each wrist and run hand detection on the crop instead. The hand
# then fills the input, and scanning two small crops is far cheaper than
# scanning the whole frame.
ROI_SCALE = 1.6          # crop size as a multiple of shoulder width. Tighter
                         # is better - it is the hand's FRACTION of the model
                         # input that matters, not its pixel count - but too
                         # tight and wrist jitter throws the hand outside the
                         # box. 1.6 leaves ~2x the hand's width of slack.
ROI_MIN_PX = 224         # upscale a small crop to at least this
ROI_MIN_VIS = 0.30       # wrist visibility needed to trust the crop

HOLD_FRAMES = 3          # frames a gesture must persist before it counts.
                         # Kept low on purpose: the console applies its own
                         # TIME-based hold, and on a Jetson running ~9 fps a
                         # frame-based hold would add most of a second.
COOLDOWN_S = 1.2         # minimum gap between two confirmed gestures
MIN_VISIBILITY = 0.5     # pose landmark confidence floor

HAND_GESTURES = [
    "OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN",
    "POINT_LEFT", "POINT_RIGHT", "POINT_UP", "PEACE", "THREE", "FOUR",
]
ARM_GESTURES = ["BOTH_ARMS_UP", "T_POSE", "ARMS_DOWN",
                "LEFT_UP", "RIGHT_UP", "LEFT_OUT", "RIGHT_OUT"]

# MIRRORING: when the frame is flipped (so the operator sees themselves as
# in a mirror, which is what feels natural), MediaPipe's LEFT_* landmarks
# land on the operator's RIGHT arm. Single-arm names returned below are
# always from the OPERATOR's point of view, and the swap is done for you
# when mirrored=True.


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class HandInfo:
    """One detected hand."""
    __slots__ = ("label", "score", "fingers", "extended", "gesture", "pts", "span")

    def __init__(self, label, score, fingers, extended, gesture, pts, span):
        self.label = label          # "Left" / "Right" (as seen by the camera)
        self.score = score
        self.fingers = fingers      # 0-5
        self.extended = extended    # dict: thumb/index/middle/ring/pinky -> bool
        self.gesture = gesture      # str or None
        self.pts = pts              # (21, 2) pixel coords
        self.span = span            # hand size in pixels, for scaling


class GestureResult:
    __slots__ = ("hands", "fingers", "hand_gesture", "arm_gesture",
                 "confirmed", "stable", "pose_pts", "fps", "rois", "source")

    def __init__(self):
        self.rois = []              # wrist crop boxes actually searched
        self.source = "full"        # "roi" or "full" - which path found hands
        self.hands = []
        self.fingers = 0
        self.hand_gesture = None
        self.arm_gesture = None
        self.confirmed = None       # EDGE: fires once, for one-shot actions
        self.stable = None          # LEVEL: held long enough, stays set while
                                    # held - for continuous driving
        self.pose_pts = None
        self.fps = 0.0


class GestureDetector:
    def __init__(self, max_hands=2, hand_conf=0.5, pose_conf=0.5,
                 use_pose=True, hold_frames=HOLD_FRAMES, mirrored=True,
                 use_roi=True):
        if mp is None:
            raise RuntimeError(
                "mediapipe not available: %s\n"
                "Install with: pip3 install --user mediapipe" % _IMPORT_ERR)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=hand_conf,
            min_tracking_confidence=0.5)
        self.mirrored = mirrored
        self.use_roi = use_roi and use_pose      # cropping needs the wrists
        self.use_pose = use_pose
        self.pose = (mp.solutions.pose.Pose(
            min_detection_confidence=pose_conf,
            min_tracking_confidence=0.5) if use_pose else None)
        self._draw = mp.solutions.drawing_utils
        self._hand_conn = mp.solutions.hands.HAND_CONNECTIONS
        self._pose_conn = mp.solutions.pose.POSE_CONNECTIONS

        self.hold_frames = hold_frames
        self._candidate = None
        self._count = 0
        self._active = None         # gesture currently held (already fired)
        self._stable = None         # gesture held long enough, level signal
        self._last_fire = 0.0
        self._t = time.time()
        self._n = 0
        self._fps = 0.0

    # ---------------- hands ----------------
    def _finger_states(self, pts):
        """Which fingers are extended. Measured radially from the wrist so
        it survives hand rotation, instead of comparing image y."""
        wrist = pts[WRIST]
        span = max(1e-6, _dist(wrist, pts[MIDDLE_MCP]))
        ext = {}
        for name, tip, pip in zip(FINGER_NAMES, FINGER_TIPS, FINGER_PIPS):
            ext[name] = _dist(wrist, pts[tip]) > EXTEND_RATIO * _dist(wrist, pts[pip])
        # The thumb folds sideways rather than curling, so measure it from
        # the index knuckle instead of the wrist.
        ref = pts[INDEX_MCP]
        ext["thumb"] = _dist(ref, pts[THUMB_TIP]) > THUMB_RATIO * _dist(ref, pts[THUMB_IP])
        return ext, span

    def _hand_gesture(self, pts, ext, span):
        n_fing = sum(ext[f] for f in FINGER_NAMES)      # index..pinky
        thumb = ext["thumb"]
        wrist = pts[WRIST]

        # --- thumbs up / down: thumb out, all four fingers curled ---
        if thumb and n_fing == 0:
            dx = pts[THUMB_TIP][0] - wrist[0]
            dy = pts[THUMB_TIP][1] - wrist[1]           # image y grows downward
            clear = (abs(dy) > THUMB_DIR_FRAC * span
                     and abs(dy) > THUMB_VERT_DOMINANCE * abs(dx))
            if clear:
                return "THUMBS_UP" if dy < 0 else "THUMBS_DOWN"
            return "FIST"       # thumb out sideways - not a clear up/down

        if not thumb and n_fing == 0:
            return "FIST"
        if n_fing == 4 and thumb:
            return "OPEN_PALM"

        # --- pointing: index only ---
        if n_fing == 1 and ext["index"]:
            dx = pts[INDEX_TIP][0] - pts[INDEX_MCP][0]
            dy = pts[INDEX_TIP][1] - pts[INDEX_MCP][1]
            if dx < -POINT_DIR_FRAC * span:
                return "POINT_LEFT"
            if dx > POINT_DIR_FRAC * span:
                return "POINT_RIGHT"
            if dy < -POINT_DIR_FRAC * span:
                return "POINT_UP"
            return None

        if n_fing == 2 and ext["index"] and ext["middle"]:
            return "PEACE"
        if n_fing == 3 and ext["index"] and ext["middle"] and ext["ring"]:
            return "THREE"
        if n_fing == 4:
            return "FOUR"
        return None

    # ---------------- arms ----------------
    def _arm_gesture(self, lm, w, h):
        """Arm pose using the CORRECT wrist indices (15/16, not 9/10), with
        every threshold scaled by shoulder width so it is distance-invariant."""
        def ok(i):
            return lm[i].visibility >= MIN_VISIBILITY

        need = [P_LEFT_SHOULDER, P_RIGHT_SHOULDER, P_LEFT_WRIST, P_RIGHT_WRIST]
        if not all(ok(i) for i in need):
            return None

        ls = np.array([lm[P_LEFT_SHOULDER].x * w, lm[P_LEFT_SHOULDER].y * h])
        rs = np.array([lm[P_RIGHT_SHOULDER].x * w, lm[P_RIGHT_SHOULDER].y * h])
        lw = np.array([lm[P_LEFT_WRIST].x * w, lm[P_LEFT_WRIST].y * h])
        rw = np.array([lm[P_RIGHT_WRIST].x * w, lm[P_RIGHT_WRIST].y * h])

        sw = max(1e-6, _dist(ls, rs))          # shoulder width = our unit
        shoulder_y = (ls[1] + rs[1]) / 2.0

        l_up = lw[1] < ls[1] - ARM_UP_FRAC * sw
        r_up = rw[1] < rs[1] - ARM_UP_FRAC * sw
        l_down = lw[1] > ls[1] + ARM_DOWN_FRAC * sw
        r_down = rw[1] > rs[1] + ARM_DOWN_FRAC * sw

        if l_up and r_up:
            return "BOTH_ARMS_UP"

        l_level = abs(lw[1] - shoulder_y) < T_POSE_Y_FRAC * sw
        r_level = abs(rw[1] - shoulder_y) < T_POSE_Y_FRAC * sw
        l_out = abs(lw[0] - ls[0]) > T_POSE_X_FRAC * sw
        r_out = abs(rw[0] - rs[0]) > T_POSE_X_FRAC * sw
        if l_level and r_level and l_out and r_out:
            return "T_POSE"

        if l_down and r_down:
            return "ARMS_DOWN"

        # --- single-arm states (menu navigation) ---
        # Names are from the OPERATOR's point of view; with a mirrored
        # frame MediaPipe's "left" is the operator's right.
        if l_up and not r_up:
            return "RIGHT_UP" if self.mirrored else "LEFT_UP"
        if r_up and not l_up:
            return "LEFT_UP" if self.mirrored else "RIGHT_UP"
        # "out" also has to be level, or an arm hanging down-and-out counts
        if (l_out and l_level) and not (r_out and r_level):
            return "RIGHT_OUT" if self.mirrored else "LEFT_OUT"
        if (r_out and r_level) and not (l_out and l_level):
            return "LEFT_OUT" if self.mirrored else "RIGHT_OUT"
        return None

    # ---------------- debounce ----------------
    def _debounce(self, gesture):
        """A gesture must hold for hold_frames, then fires exactly ONCE.
        It cannot fire again until it changes and comes back."""
        if gesture != self._candidate:
            self._candidate = gesture
            self._count = 1
            self._stable = None          # a new candidate is not stable yet
            if gesture is None:
                self._active = None      # hand dropped: re-arm
            return None

        self._count += 1
        self._stable = (gesture if (gesture is not None
                                    and self._count >= self.hold_frames)
                        else None)
        if gesture is None or self._count < self.hold_frames:
            return None
        if gesture == self._active:
            return None                  # already fired, still being held
        if time.time() - self._last_fire < COOLDOWN_S:
            return None
        self._active = gesture
        self._last_fire = time.time()
        return gesture

    # ---------------- wrist-guided crops ----------------
    def _hand_rois(self, lm, w, h):
        """Boxes around each wrist, sized from shoulder width so they scale
        with distance. Returns [(x0, y0, x1, y1), ...] in pixels."""
        need = [P_LEFT_SHOULDER, P_RIGHT_SHOULDER]
        if not all(lm[i].visibility >= MIN_VISIBILITY for i in need):
            return []
        ls = np.array([lm[P_LEFT_SHOULDER].x * w, lm[P_LEFT_SHOULDER].y * h])
        rs = np.array([lm[P_RIGHT_SHOULDER].x * w, lm[P_RIGHT_SHOULDER].y * h])
        sw = _dist(ls, rs)
        if sw < 8.0:
            return []
        box = ROI_SCALE * sw
        out = []
        for idx in (P_LEFT_WRIST, P_RIGHT_WRIST):
            if lm[idx].visibility < ROI_MIN_VIS:
                continue
            cx, cy = lm[idx].x * w, lm[idx].y * h
            x0 = int(max(0, cx - box / 2))
            y0 = int(max(0, cy - box / 2))
            x1 = int(min(w, cx + box / 2))
            y1 = int(min(h, cy + box / 2))
            if x1 - x0 > 20 and y1 - y0 > 20:
                out.append((x0, y0, x1, y1))
        return out

    def _hands_in_roi(self, frame_bgr, roi):
        """Run hand detection on one crop; landmarks come back in FULL-frame
        pixels. Returns HandInfo or None."""
        import cv2
        x0, y0, x1, y1 = roi
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        if max(ch, cw) < ROI_MIN_PX:                 # upscale so the hand fills it
            s = float(ROI_MIN_PX) / max(ch, cw)
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)),
                              interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hr = self.hands.process(rgb)
        if not hr.multi_hand_landmarks:
            return None
        hl = hr.multi_hand_landmarks[0]
        # normalised crop coords -> full-frame pixels (resize cancels out)
        pts = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0]
                        for p in hl.landmark])
        label, score = "?", 0.0
        if hr.multi_handedness:
            label = hr.multi_handedness[0].classification[0].label
            score = hr.multi_handedness[0].classification[0].score
        ext, span = self._finger_states(pts)
        return HandInfo(label, score, sum(ext.values()), ext,
                        self._hand_gesture(pts, ext, span), pts, span)

    # ---------------- main ----------------
    def update(self, frame_bgr):
        import cv2
        h, w = frame_bgr.shape[:2]
        res = GestureResult()

        # POSE FIRST - it still works at a distance, and it is what tells us
        # where the hands are so we can crop to them.
        pose_lm = None
        if self.use_pose:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pres = self.pose.process(rgb)
            if pres.pose_landmarks:
                res.pose_pts = pres.pose_landmarks
                pose_lm = pres.pose_landmarks.landmark
                res.arm_gesture = self._arm_gesture(pose_lm, w, h)

        if self.use_roi and pose_lm is not None:
            res.rois = self._hand_rois(pose_lm, w, h)

        if res.rois:
            res.source = "roi"
            for roi in res.rois:
                hnd = self._hands_in_roi(frame_bgr, roi)
                if hnd is not None:
                    res.hands.append(hnd)

        if not res.hands:
            # No usable crop (no pose, or nothing found in it): fall back to
            # scanning the whole frame. Works when you are close to the lens.
            res.source = "full"
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hres = self.hands.process(rgb)
            if hres.multi_hand_landmarks:
                handed = hres.multi_handedness or []
                for i, hl in enumerate(hres.multi_hand_landmarks):
                    pts = np.array([[p.x * w, p.y * h] for p in hl.landmark])
                    ext, span = self._finger_states(pts)
                    label, score = "?", 0.0
                    if i < len(handed):
                        label = handed[i].classification[0].label
                        score = handed[i].classification[0].score
                    res.hands.append(HandInfo(
                        label, score, sum(ext.values()), ext,
                        self._hand_gesture(pts, ext, span), pts, span))

        if res.hands:
            # the biggest hand is the one being deliberately shown
            main = max(res.hands, key=lambda x: x.span)
            res.fingers = main.fingers
            res.hand_gesture = main.gesture

        # a hand gesture outranks an arm gesture (it is more deliberate)
        res.confirmed = self._debounce(res.hand_gesture or res.arm_gesture)
        res.stable = self._stable

        self._n += 1
        now = time.time()
        if now - self._t >= 0.5:
            self._fps = self._n / (now - self._t)
            self._n, self._t = 0, now
        res.fps = self._fps
        return res

    def reset(self):
        self._candidate = None
        self._count = 0
        self._active = None
        self._stable = None

    # ---------------- drawing ----------------
    def draw(self, frame, res, action_text=""):
        import cv2
        if res.pose_pts is not None:
            self._draw.draw_landmarks(frame, res.pose_pts, self._pose_conn)
        # show where hand detection is actually looking
        for (x0, y0, x1, y1) in res.rois:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 160, 0), 2)
        for hnd in res.hands:
            for (x, y) in hnd.pts.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 220, 255), -1)
            x0, y0 = hnd.pts[WRIST].astype(int)
            cv2.putText(frame, "%s %d" % (hnd.label[:1], hnd.fingers),
                        (x0 - 20, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 220, 255), 2)

        cv2.putText(frame, "fingers: %d" % res.fingers, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if res.hand_gesture:
            cv2.putText(frame, res.hand_gesture, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if res.arm_gesture:
            cv2.putText(frame, "arms: %s" % res.arm_gesture, (15, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, "FPS %.1f  [%s roi=%d]"
                    % (res.fps, res.source, len(res.rois)),
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        if action_text:
            cv2.putText(frame, action_text, (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return frame

    def close(self):
        try:
            self.hands.close()
            if self.pose:
                self.pose.close()
        except Exception:
            pass


# ---------------- standalone preview (no robot) ----------------
if __name__ == "__main__":
    import cv2
    import sys

    src = 0
    if len(sys.argv) > 1:
        src = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Could not open camera %s" % src)
        sys.exit(1)

    det = GestureDetector()
    print("Gesture preview. Show your hand. ESC to quit.")
    print("Try: open palm, fist, thumbs up, thumbs down, point, peace.")
    last = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # mirror, so it feels natural
        r = det.update(frame)
        if r.confirmed:
            last = "CONFIRMED: %s" % r.confirmed
            print(last)
        det.draw(frame, r, last)
        cv2.imshow("Gesture preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    det.close()
    cv2.destroyAllWindows()
