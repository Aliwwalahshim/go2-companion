"""
gesture_detector.py - single-hand finger gestures + one arm pose (T_POSE).

WHY SINGLE-HAND
---------------
The previous version tracked both hands (MediaPipe `max_num_hands=2`). Two
hands in frame means two, sometimes disagreeing, finger counts every frame -
whichever hand MediaPipe happened to list first "won" arbitrarily, so the
same real gesture could be read differently frame to frame. This detector
tracks exactly ONE hand (`max_num_hands=1`). When more than one hand is
visible, MediaPipe's own detection-confidence ranking picks the best one and
that is the only hand ever considered - there is no left/right hand voting
to go wrong.

WHAT THIS PROVIDES
------------------
* Finger-count / shape gestures from the one tracked hand: FIST, POINT_UP,
  POINT_LEFT, POINT_RIGHT, PEACE, THREE, FOUR, THUMBS_UP, THUMBS_DOWN,
  OPEN_PALM. These drive BOTH the menu (mode selection) and in-mode
  commands in go2_master.py.
* One whole-body arm pose, T_POSE, from MediaPipe Pose. This is the single
  global "back to the menu" gesture - it stays a big, room-readable arm
  pose on purpose, since it must be reachable even while driving/tricking
  and must not accidentally trigger a hand gesture at the same time (a
  T-pose necessarily shows open palms, so OPEN_PALM racing T_POSE is
  handled by go2_master's DOMINANT_ARM set, not here).
* An optional wrist-guided ROI crop (`use_roi=True`, default): once Pose
  has located a wrist, hand detection runs on a crop around it instead of
  the full frame. This is what makes finger gestures readable at
  across-the-room distance instead of only up close. Disable with
  `use_roi=False` to always scan the full frame.
"""

import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---- Pose thresholds (shoulder-width units - camera/distance independent)
UP_K = 0.40
LEVEL_K = 0.45
OUT_K = 0.90
SHOULDER_VIS = 0.30
WRIST_VIS = 0.15
MIN_SHOULDER_W = 0.02

# ---- Finger-state thresholds
FINGER_STRAIGHT_K = 0.55   # tip must be this much farther from wrist than pip
THUMB_STRAIGHT_K = 0.35
POINT_AXIS_RATIO = 1.2     # how much the horizontal axis must dominate to
                           # read as LEFT/RIGHT instead of UP

# ---- Debounce for the continuous ("stable") driving signal
STABLE_FRAMES = 4

# ---- ROI
ROI_MARGIN = 1.6           # ROI half-size, in shoulder widths
ROI_MIN_PX = 140


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


class Result:
    __slots__ = ("hand_gesture", "arm_gesture", "stable", "fingers",
                 "hands", "rois", "fps", "source",
                 "_hand_landmarks", "_pose_landmarks", "frame_shape",
                 "_roi_draw_info")

    def __init__(self):
        self.hand_gesture = None
        self.arm_gesture = None
        self.stable = None
        self.fingers = 0
        self.hands = []
        self.rois = []
        self.fps = 0.0
        self.source = "-"
        self._hand_landmarks = None
        self._pose_landmarks = None
        self.frame_shape = None
        self._roi_draw_info = ((0, 0), None)


class GestureDetector:
    def __init__(self, use_pose=True, mirrored=True, hand_conf=0.5,
                 use_roi=True):
        self.use_pose = use_pose
        self.mirrored = mirrored
        self.use_roi = use_roi

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,                 # <-- the single-hand fix
            min_detection_confidence=hand_conf,
            min_tracking_confidence=max(0.3, hand_conf - 0.1))

        self.pose = None
        if use_pose:
            self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)

        self._stable_gesture = None
        self._stable_count = 0
        self._stable_value = None

        self._last_t = time.time()
        self._fps = 0.0

    # -- public API --------------------------------------------------
    def reset(self):
        """Clear debounce state. Call this on every mode change so an old
        held gesture cannot instantly re-fire in the new mode."""
        self._stable_gesture = None
        self._stable_count = 0
        self._stable_value = None

    def close(self):
        self.hands.close()
        if self.pose is not None:
            self.pose.close()

    def update(self, frame):
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        res = Result()
        res.fps = self._fps
        res.frame_shape = frame.shape

        pose_landmarks = None
        if self.pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pr = self.pose.process(rgb)
            if pr.pose_landmarks is not None:
                pose_landmarks = pr.pose_landmarks.landmark
                res._pose_landmarks = pose_landmarks
                res.arm_gesture = _classify_arm(pose_landmarks)

        roi = None
        if self.use_roi and pose_landmarks is not None:
            roi = _wrist_roi(pose_landmarks, frame.shape)
            if roi is not None:
                res.rois.append(roi)

        if roi is not None:
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            hand_lms, handed = _run_hands(self.hands, crop)
            res.source = "roi"
            proc_shape = crop.shape
            if not hand_lms:
                # fall back to full frame this one frame - a fast hand can
                # slip outside a stale wrist ROI.
                hand_lms, handed = _run_hands(self.hands, frame)
                res.source = "full"
                proc_shape = frame.shape
            offset = (x0, y0)
        else:
            hand_lms, handed = _run_hands(self.hands, frame)
            res.source = "full"
            offset = (0, 0)
            proc_shape = frame.shape

        if hand_lms:
            res.hands = [hand_lms]
            res._hand_landmarks = hand_lms
            fingers, gesture = _classify_hand(hand_lms, self.mirrored, proc_shape)
            res.fingers = fingers
            res.hand_gesture = gesture

        # dominant channel for continuous ("level") driving: hand first,
        # T_POSE can also drive the "stable" concept if ever needed.
        level = res.hand_gesture
        if level != self._stable_gesture:
            self._stable_gesture = level
            self._stable_count = 1
        else:
            self._stable_count += 1
        if level is not None and self._stable_count >= STABLE_FRAMES:
            self._stable_value = level
        elif level is None:
            self._stable_value = None
        res.stable = self._stable_value

        self._roi_offset = offset if hand_lms else None
        self._roi_full = (roi is None) or (res.source == "full")
        res._roi_draw_info = (offset, roi)
        return res

    def draw(self, frame, res):
        h, w = frame.shape[:2]
        for (x0, y0, x1, y1) in res.rois:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 160, 255), 1)

        if res._pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame,
                _as_landmark_list(res._pose_landmarks),
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1,
                                       circle_radius=1),
                mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1))

        if res._hand_landmarks is not None:
            offset, roi = res._roi_draw_info
            ox, oy = offset
            rw = (roi[2] - roi[0]) if roi is not None and res.source == "roi" else w
            rh = (roi[3] - roi[1]) if roi is not None and res.source == "roi" else h
            pts = []
            for lm in res._hand_landmarks:
                px = int(ox + lm.x * rw)
                py = int(oy + lm.y * rh)
                pts.append((px, py))
            for a, b in mp_hands.HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 255, 255), -1)

        y = 26
        if res.hand_gesture:
            cv2.putText(frame, "hand: %s" % res.hand_gesture, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 26
        if res.arm_gesture:
            cv2.putText(frame, "arm: %s" % res.arm_gesture, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 60), 2)


# =======================================================================
# helpers
# =======================================================================
def _run_hands(hands_model, bgr):
    if bgr is None or bgr.size == 0:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands_model.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None
    # max_num_hands=1, so there is at most one - but stay defensive.
    lms = result.multi_hand_landmarks[0].landmark
    handed = None
    if result.multi_handedness:
        handed = result.multi_handedness[0].classification[0].label
    return lms, handed


def _as_landmark_list(landmarks):
    from mediapipe.framework.formats import landmark_pb2
    lst = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        p = lst.landmark.add()
        p.x, p.y, p.z, p.visibility = lm.x, lm.y, lm.z, lm.visibility
    return lst


def _wrist_roi(pose_landmarks, shape):
    h, w = shape[:2]
    ls = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs_ = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if min(ls.visibility, rs_.visibility) < SHOULDER_VIS:
        return None
    sw = abs(ls.x - rs_.x) * w
    if sw < 5:
        return None

    lw = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    # pick whichever wrist is raised higher (more likely to be gesturing)
    wrist = lw if (lw.visibility >= WRIST_VIS and lw.y < rw.y) else rw
    if wrist.visibility < WRIST_VIS:
        wrist = lw if lw.visibility >= WRIST_VIS else None
    if wrist is None:
        return None

    cx, cy = wrist.x * w, wrist.y * h
    half = max(ROI_MIN_PX, sw * ROI_MARGIN)
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(w, cx + half))
    y1 = int(min(h, cy + half))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return (x0, y0, x1, y1)


def _classify_arm(landmarks):
    """Only T_POSE matters to go2_master now (the global 'back to menu'
    gesture) - mode selection moved to hand/finger gestures."""
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs_ = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    if min(ls.visibility, rs_.visibility) < SHOULDER_VIS:
        return None
    if min(lw.visibility, rw.visibility) < WRIST_VIS:
        return None
    sw = abs(ls.x - rs_.x)
    if sw < MIN_SHOULDER_W:
        return None
    l_level = abs(lw.y - ls.y) < LEVEL_K * sw
    r_level = abs(rw.y - rs_.y) < LEVEL_K * sw
    l_out = l_level and abs(lw.x - ls.x) > OUT_K * sw
    r_out = r_level and abs(rw.x - rs_.x) > OUT_K * sw
    if l_out and r_out:
        return "T_POSE"
    return None


# Landmark indices (MediaPipe Hands)
_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_PIP = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
_MCP = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
WRIST = 0


def _finger_extended(lm, name):
    wrist = lm[WRIST]
    tip = lm[_TIP[name]]
    pip = lm[_PIP[name]]
    mcp = lm[_MCP[name]]
    span = _dist(wrist, mcp) + 1e-6
    if name == "thumb":
        return _dist(wrist, tip) > _dist(wrist, pip) + THUMB_STRAIGHT_K * span
    return _dist(wrist, tip) > _dist(wrist, pip) + FINGER_STRAIGHT_K * span * 0.3 \
        and tip.y < pip.y - 0.02


def _classify_hand(lm, mirrored, shape=None):
    ext = {name: _finger_extended(lm, name) for name in _TIP}
    count = sum(1 for name in ("thumb", "index", "middle", "ring", "pinky")
               if ext[name])
    four_count = sum(1 for name in ("index", "middle", "ring", "pinky")
                     if ext[name])

    if four_count == 0 and not ext["thumb"]:
        return 0, "FIST"

    if four_count == 0 and ext["thumb"]:
        wrist = lm[WRIST]
        tip = lm[_TIP["thumb"]]
        if tip.y < wrist.y - 0.05:
            return 1, "THUMBS_UP"
        if tip.y > wrist.y + 0.05:
            return 1, "THUMBS_DOWN"
        return 1, "FIST"

    if four_count == 1 and ext["index"] and not ext["thumb"]:
        wrist = lm[WRIST]
        tip = lm[_TIP["index"]]
        # Landmarks are normalized to the width/height of whatever image was
        # actually processed (a square ROI crop, or the raw 16:9 frame on a
        # fallback). Scaling by that shape puts dx/dy in real pixel-ish
        # units so a non-square source cannot make horizontal pointing look
        # "flatter" than it really is.
        if shape is not None:
            h, w = shape[0], shape[1]
        else:
            h = w = 1.0
        dx = (tip.x - wrist.x) * w
        dy = (tip.y - wrist.y) * h
        # Single threshold, no gap between the UP and LEFT/RIGHT checks:
        # every angle lands in exactly one bucket, so a diagonal point can
        # no longer fall through to a default reading.
        if abs(dx) > abs(dy) * POINT_AXIS_RATIO:
            # go2_master.py always hands us an already-mirrored (selfie-view)
            # frame when mirrored=True, so increasing x already IS the
            # subject's own right side - no extra inversion needed there.
            # Only a raw, un-flipped frame needs inverting.
            right = (dx > 0) if mirrored else (dx < 0)
            return 1, "POINT_RIGHT" if right else "POINT_LEFT"
        return 1, "POINT_UP"

    if four_count == 2 and ext["index"] and ext["middle"]:
        return 2, "PEACE"

    if four_count == 3 and ext["index"] and ext["middle"] and ext["ring"]:
        return 3, "THREE"

    if four_count == 4 and not ext["thumb"]:
        return 4, "FOUR"

    if four_count == 4 and ext["thumb"]:
        return 5, "OPEN_PALM"

    return count, None

