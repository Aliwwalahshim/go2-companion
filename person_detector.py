"""
person_detector.py - YOLO person detection + single-operator lock/re-acquire.

PURPOSE
-------
Perception only. No robot control. It finds people in a frame, locks onto
ONE operator, tracks them frame to frame, and re-acquires them when they
leave the view and come back.

SEAM (how the follow code uses it)
----------------------------------
    det = PersonDetector("yolov8n.engine")   # or "yolov8n.pt"
    det.reset()                              # call when FOLLOW mode starts
    res = det.update(color_frame)            # every frame
    if res:                                  # res.cx, res.cy, res.bbox, res.state
        ...steer toward res.cx, read depth at (res.cx, res.cy)...

    det.people                               # ALL people seen in the last frame
                                             # (list of PersonBox) - use this to
                                             # draw boxes on everyone and to mask
                                             # every person out of the LiDAR scan.

STATES returned in res.state:
    LOCKED      - tracking the operator normally
    REACQUIRED  - operator came back after being lost, re-locked
    (res is None while SEARCHING - operator not currently visible)

HONEST LIMITATION
-----------------
Re-acquire is GEOMETRIC (last-known position + similar size), not
appearance-based re-ID. If the operator is gone for a while and several
people are present, it may re-lock the wrong person. That is acceptable
for a controlled demo; true re-ID would need a heavier model.
"""

import time

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    _IMPORT_ERR = exc

PERSON_CLASS = 0        # COCO 'person'

# --- tracker tuning (all in fractions of frame width, so resolution-independent) ---
MATCH_GATE = 0.30       # max centre move between frames to count as same person.
                        # Widened so fast left/right walking does not drop the
                        # lock between frames.
SIZE_RATIO = (0.5, 2.0) # allowed box-size change to count as same person
REACQUIRE_RADIUS = 0.45 # how near last-known centre a returning person must
                        # appear - wider so stepping aside and back re-locks you
REACQUIRE_GRACE = 100000  # LOCK ONCE: keep the SAME operator locked for the
                        # whole session and never auto-switch to another person.
                        # (Restart the script to pick a new operator.)
SMOOTH = 0.6            # lock position smoothing (0..1, higher = snappier)
CONFIRM_FRAMES = 4      # candidate must be seen this many consecutive frames
                        # before it becomes / re-becomes the operator (stops a
                        # one-frame chair detection from stealing the lock)
CONFIRM_GATE = 0.10     # max centre drift (fraction of width) between those frames


class OperatorResult:
    __slots__ = ("cx", "cy", "bbox", "conf", "state")
    def __init__(self, cx, cy, bbox, conf, state):
        self.cx, self.cy, self.bbox, self.conf, self.state = cx, cy, bbox, conf, state


class PersonBox:
    """One detected person (any person, not just the operator)."""
    __slots__ = ("cx", "cy", "w", "h", "conf", "bbox")
    def __init__(self, cx, cy, w, h, conf, bbox):
        self.cx, self.cy, self.w, self.h, self.conf, self.bbox = cx, cy, w, h, conf, bbox


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", imgsz=320, conf=0.4):
        if YOLO is None:
            raise RuntimeError(
                f"ultralytics not available: {_IMPORT_ERR}\n"
                f"Install with: pip3 install --user ultralytics")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.model_path = model_path
        # lock state
        self.lock = None          # dict(cx, cy, w, h) in pixels, or None
        self.lost = 0
        self.pending = None       # candidate being confirmed: dict(cx, cy, count)
        self.people = []          # every person in the last frame (PersonBox list)
        self._fps = 0.0
        self._t = time.time()
        self._n = 0

    def reset(self):
        """Call when entering FOLLOW so it re-acquires cleanly."""
        self.lock = None
        self.lost = 0
        self.pending = None
        self.people = []

    def _confirm(self, cand, frame_w):
        """Track a lock candidate across frames; True once it has been seen
        CONFIRM_FRAMES consecutive frames without jumping around."""
        cx, cy = cand[0], cand[1]
        gate = CONFIRM_GATE * frame_w
        if self.pending is not None:
            moved = ((cx - self.pending["cx"]) ** 2
                     + (cy - self.pending["cy"]) ** 2) ** 0.5
            if moved <= gate:
                self.pending["cx"] = cx
                self.pending["cy"] = cy
                self.pending["count"] += 1
            else:
                self.pending = dict(cx=cx, cy=cy, count=1)
        else:
            self.pending = dict(cx=cx, cy=cy, count=1)
        return self.pending["count"] >= CONFIRM_FRAMES

    @property
    def fps(self):
        return self._fps

    def is_engine(self):
        return str(self.model_path).endswith(".engine")

    def _detect(self, frame):
        """Return list of (cx, cy, w, h, conf, bbox) for people only."""
        res = self.model(frame, classes=[PERSON_CLASS], imgsz=self.imgsz,
                         conf=self.conf, verbose=False)
        out = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                out.append((cx, cy, x2 - x1, y2 - y1, float(b.conf[0]),
                            (int(x1), int(y1), int(x2), int(y2))))
        return out

    def update(self, frame):
        """Run detection + operator lock/re-acquire. Returns OperatorResult|None."""
        h, w = frame.shape[:2]
        gate = MATCH_GATE * w
        reacq = REACQUIRE_RADIUS * w

        dets = self._detect(frame)
        self.people = [PersonBox(*d) for d in dets]

        # fps
        self._n += 1
        now = time.time()
        if now - self._t >= 0.5:
            self._fps = self._n / (now - self._t)
            self._n = 0
            self._t = now

        if not dets:
            self.lost += 1
            self.pending = None        # confirmation must be consecutive
            if self.lock and self.lost > REACQUIRE_GRACE:
                self.lock = None       # give up; next person becomes operator
            return None

        # ---- no operator locked yet: take the largest person, but only
        #      after it has been stable for CONFIRM_FRAMES frames ----
        if self.lock is None:
            cand = max(dets, key=lambda d: d[2] * d[3])
            if not self._confirm(cand, w):
                return None
            cx, cy, bw, bh, cf, bbox = cand
            self.pending = None
            self.lock = dict(cx=cx, cy=cy, w=bw, h=bh)
            self.lost = 0
            return OperatorResult(int(cx), int(cy), bbox, cf, "LOCKED")

        # ---- operator locked: match to the best candidate ----
        lx, ly, lw, lh = self.lock["cx"], self.lock["cy"], self.lock["w"], self.lock["h"]
        best, best_score = None, None
        for d in dets:
            cx, cy, bw, bh, cf, bbox = d
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            ratio = (bw * bh) / max(1.0, lw * lh)
            # while visible use the tight gate; while searching use the wider radius
            limit = gate if self.lost == 0 else reacq
            if dist <= limit and SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]:
                if best_score is None or dist < best_score:
                    best, best_score = d, dist

        if best is None:
            # operator not matched this frame
            self.lost += 1
            self.pending = None
            if self.lost > REACQUIRE_GRACE:
                self.lock = None
            return None

        if self.lost > 0:
            # operator was missing: require a few consistent frames before
            # trusting the re-acquire, so a chair / passer-by near the
            # last-known position cannot steal the lock in one frame
            if not self._confirm(best, w):
                self.lost += 1
                if self.lost > REACQUIRE_GRACE:
                    self.lock = None
                    self.pending = None
                return None
            self.pending = None

        cx, cy, bw, bh, cf, bbox = best
        reacquired = self.lost > 0
        # smooth the lock so tracking is steady
        self.lock["cx"] = SMOOTH * cx + (1 - SMOOTH) * lx
        self.lock["cy"] = SMOOTH * cy + (1 - SMOOTH) * ly
        self.lock["w"] = SMOOTH * bw + (1 - SMOOTH) * lw
        self.lock["h"] = SMOOTH * bh + (1 - SMOOTH) * lh
        self.lost = 0
        return OperatorResult(int(cx), int(cy), bbox, cf,
                              "REACQUIRED" if reacquired else "LOCKED")
