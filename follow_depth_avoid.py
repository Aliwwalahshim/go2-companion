"""
follow_depth_avoid.py - Operator following with TWO layers of avoidance.

    Layer 1  Go2 built-in LiDAR avoidance  -> tall obstacles (walls, chairs)
    Layer 2  RealSense depth avoidance     -> LOW obstacles (boxes on floor)

The built-in LiDAR ignores anything under about 0.3 m, which is why the
robot kept hitting the cardboard box.  The depth camera covers exactly that
gap.  Together they cover both.

TWO WINDOWS OPEN:
    "Operator Follow"  - the camera view with the tracking boxes
    "Depth Avoid"      - top-down map of what the depth camera sees ahead

Behaviour you asked for: when something blocks the path, the robot steps
sideways around it while still turning toward you, so it comes back to you
after passing the obstacle.

STOP SIGN: hold an open palm up toward the camera (the OPEN_PALM gesture -
all five fingers out) and the robot halts in place - forward/lateral/turn
all go to zero - for as long as you hold it. Lower your hand and it resumes
following automatically. This is read only off the locked operator's own
crop (see gesture_detector.GestureDetector), so a bystander's hand next to
them won't trigger it. Depth avoidance keeps running throughout; it just has
nothing to correct while the commanded velocity is already zero.

STAIR GESTURES: same crop, same reader, two more single-hand gestures.
Hold up 1 finger (POINT_UP) and the robot switches to Classic gait
(SportClient.ClassicWalk(True) - the gait confirmed on this unit to climb
real stairs, see go2_master.py's STAIRS mode) and the depth-avoidance layer
(the "depth lidar") is disabled, so it will not read a stair riser as an
obstacle and refuse to approach it. Hold up 2 fingers (PEACE) to switch
back to Free gait and turn depth avoidance back on. The default at startup
is always Free gait with depth avoidance on - you must explicitly ask for
climb mode every time, it is never the default.

RUN (one terminal, NOT sourced with foxy):
    python3 follow_depth_avoid.py eth0

Remote in hand.  L2 + B is the physical e-stop.
"""

import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

from person_detector import PersonDetector
from depth_avoider import DepthAvoider
from gesture_detector import GestureDetector

# ================= TUNING =================
CAM_W, CAM_H, CAM_FPS = 848, 480, 30
MIRROR_IMAGE = True

MODEL_PATH = "yolov8n.engine"
IMGSZ = 320
DET_CONF = 0.50
FIX_EXPOSURE = True

USE_BUILTIN_AVOID = True   # Go2 LiDAR, for tall obstacles
USE_DEPTH_AVOID = True     # RealSense, for low obstacles

# --- stop-sign gesture (open palm = halt) ---
USE_STOP_GESTURE = True
STOP_GESTURE = "OPEN_PALM"
STOP_BOX_MARGIN = 0.25      # grow the operator bbox this much before reading
                            # the gesture, so a hand near the box edge isn't clipped

# --- stair gestures (1 finger = climb mode, 2 fingers = back to normal) ---
CLIMB_ON_GESTURE = "POINT_UP"    # 1 finger: ClassicWalk + depth-avoid OFF
CLIMB_OFF_GESTURE = "PEACE"      # 2 fingers: FreeWalk + depth-avoid ON

# --- follow control ---
STOP_DISTANCE_M = 1.7
MIN_SAFE_M = 0.9
DIST_DEADBAND_M = 0.2
ROT_KP = 1.0
ROT_MAX = 0.7
ROT_DEADBAND = 0.10
ROT_SIGN = 1.0
FWD_KP = 0.8
FWD_MAX = 0.5              # gentler: avoidance needs time to react
LAT_MAX = 0.35
LOST_GRACE_FRAMES = 15
HOLD_DECAY = 0.94
DEPTH_WINDOW = 5
MOVE_INTERVAL = 0.06

# --- smoothing ---
EMA_ALPHA_WALK = 0.2
ACC_FWD, DEC_FWD = 0.6, 1.2
ACC_LAT, DEC_LAT = 0.9, 1.6
ACC_ROT, DEC_ROT = 2.0, 3.0
SEND_EPS = 0.02


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def slew(cur, target, acc, dec, dt):
    if target > cur:
        rate = acc if cur >= 0 else dec
        return min(cur + rate * dt, target)
    rate = acc if cur <= 0 else dec
    return max(cur - rate * dt, target)


def median_depth(depth_np, x, y):
    half = DEPTH_WINDOW // 2
    vals = []
    h, w = depth_np.shape[:2]
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px = clamp(x + dx, 0, w - 1)
            py = clamp(y + dy, 0, h - 1)
            d = depth_np[py, px] * 0.001
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def load_detector():
    import os
    path = MODEL_PATH
    if path.endswith(".engine") and not os.path.exists(path):
        print("[detector] %s not found - falling back to yolov8n.pt" % path)
        path = "yolov8n.pt"
    det = PersonDetector(path, imgsz=IMGSZ, conf=DET_CONF)
    print("[detector] loaded %s (imgsz=%d, conf=%.2f)" % (path, IMGSZ, DET_CONF))
    return det


def tune_color_sensor(profile):
    if not FIX_EXPOSURE:
        return
    try:
        for s in profile.get_device().query_sensors():
            try:
                name = s.get_info(rs.camera_info.name)
            except Exception:
                continue
            if "RGB" in name or "Color" in name:
                if s.supports(rs.option.auto_exposure_priority):
                    s.set_option(rs.option.auto_exposure_priority, 0)
                    print("[camera] short exposure enabled")
    except Exception as exc:
        print("[camera] exposure tuning skipped: %s" % exc)


def start_builtin_avoid():
    av = ObstaclesAvoidClient()
    av.SetTimeout(3.0)
    av.Init()
    print("[lidar-avoid] UseRemoteCommandFromApi -> %s"
          % av.UseRemoteCommandFromApi(True))
    print("[lidar-avoid] SwitchSet(True) -> %s" % av.SwitchSet(True))
    time.sleep(0.4)
    try:
        code, on = av.SwitchGet()
        print("[lidar-avoid] enabled = %s" % on)
        if not on:
            print("[lidar-avoid] WARNING: did not enable.")
    except Exception as exc:
        print("[lidar-avoid] SwitchGet failed: %s" % exc)
    return av


def stop_builtin_avoid(av):
    if av is None:
        return
    try:
        av.Move(0.0, 0.0, 0.0)
        av.SwitchSet(False)
        av.UseRemoteCommandFromApi(False)
        print("[lidar-avoid] switched off, remote restored.")
    except Exception as exc:
        print("[lidar-avoid] shutdown warning: %s" % exc)


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"

    print("Init channel on %s ..." % interface)
    ChannelFactoryInitialize(0, interface)
    time.sleep(0.5)

    robot = SportClient()
    robot.SetTimeout(3.0)
    robot.Init()
    print("SportClient ready.")
    try:
        robot.StandUp()
    except Exception as exc:
        print("StandUp failed: %s" % exc)
    time.sleep(4.0)

    # settle into a balanced stance, then engage a walking gait
    try:
        print("[gait] BalanceStand -> %s" % robot.BalanceStand())
    except Exception as exc:
        print("[gait] BalanceStand failed: %s" % exc)
    time.sleep(1.0)
    # Free gait is always the starting state - climb mode (Classic gait) is
    # only ever entered live, via the POINT_UP gesture below, never at
    # startup.
    try:
        try:
            code = robot.FreeWalk()
        except TypeError:
            code = robot.FreeWalk(True)
        print("[gait] FreeWalk -> %s" % code)
    except Exception as exc:
        print("[gait] FreeWalk failed: %s" % exc)
    time.sleep(1.0)

    lidar_av = None
    if USE_BUILTIN_AVOID:
        try:
            lidar_av = start_builtin_avoid()
        except Exception as exc:
            print("[lidar-avoid] unavailable: %s" % exc)

    det = load_detector()

    stop_gesture_det = None
    if USE_STOP_GESTURE:
        # pose off, ROI off - we hand it an already-tight crop of the locked
        # operator ourselves, so it only ever reads THEIR hand.
        stop_gesture_det = GestureDetector(use_pose=False, mirrored=MIRROR_IMAGE,
                                           use_roi=False)
        print("[stop-sign] enabled: hold an open palm up to halt the robot.")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
    cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(cfg)
    tune_color_sensor(profile)
    det.reset()

    # depth avoider needs the ALIGNED intrinsics (colour stream geometry)
    depth_av = None
    if USE_DEPTH_AVOID:
        intr = profile.get_stream(rs.stream.color) \
                      .as_video_stream_profile().get_intrinsics()
        dscale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_av = DepthAvoider(intr, mirror=MIRROR_IMAGE, depth_scale=dscale)
        print("[depth-avoid] ready (%dx%d, scale %.4f)"
              % (intr.width, intr.height, dscale))

    moving = False
    last_move = 0.0
    last_err = 0.0
    lost = 0
    stopped_by_gesture = False
    stopped_prev = False
    climb_mode = False          # POINT_UP -> True (Classic gait, depth-avoid off)
    depth_avoid_enabled = True  # only meaningful while depth_av is not None
    smooth_offset = 0.0
    smooth_dist = 0.0
    hold_fwd = 0.0
    hold_rot = 0.0
    cmd_fwd = cmd_lat = cmd_rot = 0.0
    last_loop = time.time()

    def report_err(what, exc):
        nonlocal last_err
        now = time.time()
        if now - last_err >= 2.0:
            last_err = now
            print("[robot] %s failed: %s" % (what, exc))

    def send_cmd(fwd, lat, rot):
        nonlocal moving, last_move
        now = time.time()
        if now - last_move < MOVE_INTERVAL:
            return
        last_move = now
        if abs(fwd) > SEND_EPS or abs(lat) > SEND_EPS or abs(rot) > SEND_EPS:
            try:
                if lidar_av is not None:
                    lidar_av.Move(fwd, lat, rot)
                else:
                    robot.Move(fwd, lat, rot)
                moving = True
            except Exception as exc:
                report_err("Move", exc)
        elif moving:
            try:
                if lidar_av is not None:
                    lidar_av.Move(0.0, 0.0, 0.0)
                else:
                    robot.StopMove()
            except Exception as exc:
                report_err("Stop", exc)
            moving = False

    def hard_stop():
        nonlocal moving, cmd_fwd, cmd_lat, cmd_rot
        cmd_fwd = cmd_lat = cmd_rot = 0.0
        try:
            if lidar_av is not None:
                lidar_av.Move(0.0, 0.0, 0.0)
            robot.StopMove()
        except Exception:
            pass
        moving = False
        # always lower into a low, stable crouch when the program stops, so
        # the robot never stays standing unattended
        time.sleep(0.3)
        try:
            print("[robot] StandDown -> %s" % robot.StandDown())
        except Exception as exc:
            print("[robot] StandDown failed: %s" % exc)

    def set_climb_mode(on):
        """POINT_UP -> on=True: Classic gait, depth-avoid off, so a stair
        riser is never read as an obstacle to stop for. PEACE -> on=False:
        back to Free gait, depth-avoid back on. Never the default - only
        reachable live, by gesture."""
        nonlocal depth_avoid_enabled
        if on:
            try:
                print("[climb] ClassicWalk(True) -> %s" % robot.ClassicWalk(True))
            except Exception as exc:
                report_err("ClassicWalk", exc)
            depth_avoid_enabled = False
            print("[climb] depth-avoid OFF - climb mode.")
        else:
            try:
                try:
                    code = robot.FreeWalk()
                except TypeError:
                    code = robot.FreeWalk(True)
                print("[climb] FreeWalk -> %s" % code)
            except Exception as exc:
                report_err("FreeWalk", exc)
            depth_avoid_enabled = True
            print("[climb] depth-avoid ON - normal follow.")

    layers = []
    if lidar_av is not None:
        layers.append("LIDAR")
    if depth_av is not None:
        layers.append("DEPTH")
    mode_txt = "AVOID: " + ("+".join(layers) if layers else "NONE")
    stop_txt = (", open palm = STOP, 1 finger = climb mode ON, "
                "2 fingers = climb mode OFF") if stop_gesture_det is not None else ""
    print("\n=== OPERATOR FOLLOW (%s%s) ===  ESC quits, L2+B = e-stop\n"
          % (mode_txt, stop_txt))

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf = frames.get_color_frame()
            dfr = frames.get_depth_frame()
            if not cf or not dfr:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(dfr.get_data())

            frame = cv2.flip(color, 1) if MIRROR_IMAGE else color.copy()
            res = det.update(frame)

            tracked = []
            for p in det.people:
                dcx = int((CAM_W - 1) - p.cx) if MIRROR_IMAGE else int(p.cx)
                dcx = clamp(dcx, 0, CAM_W - 1)
                dcy = clamp(int(p.cy), 0, CAM_H - 1)
                tracked.append((p, median_depth(depth, dcx, dcy)))

            status = "SEARCHING"
            dist = 0.0
            tgt_fwd = tgt_lat = tgt_rot = 0.0
            holding = False

            if res is None:
                lost += 1
                stopped_by_gesture = False
                if lost <= LOST_GRACE_FRAMES:
                    holding = True
                    hold_fwd *= HOLD_DECAY
                    hold_rot *= HOLD_DECAY
                    tgt_fwd, tgt_rot = hold_fwd, hold_rot
                else:
                    hold_fwd = hold_rot = 0.0
            else:
                lost = 0
                status = res.state
                for (p, d) in tracked:
                    if p.bbox == res.bbox:
                        dist = d
                        break

                raw_offset = (res.cx - CAM_W / 2) / (CAM_W / 2)
                smooth_offset = (EMA_ALPHA_WALK * raw_offset
                                 + (1.0 - EMA_ALPHA_WALK) * smooth_offset)
                if abs(smooth_offset) > ROT_DEADBAND:
                    tgt_rot = clamp(ROT_SIGN * ROT_KP * smooth_offset,
                                    -ROT_MAX, ROT_MAX)

                if dist > 0:
                    smooth_dist = (EMA_ALPHA_WALK * dist
                                   + (1.0 - EMA_ALPHA_WALK) * smooth_dist)
                    err = smooth_dist - STOP_DISTANCE_M
                    if err > DIST_DEADBAND_M and smooth_dist > MIN_SAFE_M:
                        tgt_fwd = clamp(FWD_KP * err, 0.0, FWD_MAX)

                # ---- stop-sign gesture: open palm halts the robot ----
                stopped_by_gesture = False
                if stop_gesture_det is not None:
                    x1, y1, x2, y2 = res.bbox
                    bw, bh = x2 - x1, y2 - y1
                    mx, my = int(bw * STOP_BOX_MARGIN), int(bh * STOP_BOX_MARGIN)
                    cx0 = clamp(x1 - mx, 0, CAM_W - 1)
                    cy0 = clamp(y1 - my, 0, CAM_H - 1)
                    cx1 = clamp(x2 + mx, cx0 + 1, CAM_W)
                    cy1 = clamp(y2 + my, cy0 + 1, CAM_H)
                    crop = frame[cy0:cy1, cx0:cx1]
                    g = stop_gesture_det.update(crop)
                    stopped_by_gesture = (g.stable == STOP_GESTURE)

                    if g.stable == CLIMB_ON_GESTURE and not climb_mode:
                        climb_mode = True
                        set_climb_mode(True)
                    elif g.stable == CLIMB_OFF_GESTURE and climb_mode:
                        climb_mode = False
                        set_climb_mode(False)

                if stopped_by_gesture:
                    tgt_fwd = tgt_lat = tgt_rot = 0.0
                    hold_fwd = hold_rot = 0.0
                else:
                    hold_fwd, hold_rot = tgt_fwd, tgt_rot

            if stopped_by_gesture != stopped_prev:
                print("[stop-sign] operator raised palm -> HALT."
                      if stopped_by_gesture else
                      "[stop-sign] palm lowered -> resuming follow.")
                stopped_prev = stopped_by_gesture

            # ---- LAYER 2: depth avoidance for LOW obstacles ----
            adj = None
            if depth_av is not None:
                # never treat the tracked people as obstacles
                boxes = [p.bbox for (p, _) in tracked]
                adj = depth_av.compute(depth, exclude_boxes=boxes)

                if depth_avoid_enabled and adj.have_scan and tgt_fwd > 0.01:
                    if adj.block_forward:
                        tgt_fwd = 0.0
                        hold_fwd = 0.0
                    else:
                        tgt_fwd *= adj.fwd_scale
                    # step around it; rotation toward you keeps running,
                    # so the robot comes back to you after passing
                    tgt_lat = clamp(adj.lat, -LAT_MAX, LAT_MAX)

            # ---- slew-rate limit ----
            now = time.time()
            dt = clamp(now - last_loop, 0.0, 0.2)
            last_loop = now
            cmd_fwd = slew(cmd_fwd, tgt_fwd, ACC_FWD, DEC_FWD, dt)
            cmd_lat = slew(cmd_lat, tgt_lat, ACC_LAT, DEC_LAT, dt)
            cmd_rot = slew(cmd_rot, tgt_rot, ACC_ROT, DEC_ROT, dt)
            send_cmd(cmd_fwd, cmd_lat, cmd_rot)

            # ---- drawing ----
            for (p, d) in tracked:
                if res is not None and p.bbox == res.bbox:
                    continue
                x1, y1, x2, y2 = p.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                cv2.putText(frame, "person %.1fm" % d if d > 0 else "person",
                            (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (180, 180, 180), 1)

            if res is not None:
                x1, y1, x2, y2 = res.bbox
                col = (60, 60, 245) if stopped_by_gesture else \
                      ((0, 255, 0) if status == "LOCKED" else (0, 200, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.circle(frame, (res.cx, res.cy), 5, col, -1)
                if dist > 0:
                    cv2.putText(frame, "%.2f m" % dist, (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            if stopped_by_gesture:
                tag = "STOP SIGN - HALTED"
                tcol = (60, 60, 245)
            else:
                tag = {"LOCKED": "TRACKING OPERATOR",
                       "REACQUIRED": "RE-ACQUIRED",
                       "SEARCHING": "SEARCHING..."}[status]
                tcol = (0, 255, 0) if status == "LOCKED" else (0, 200, 255)
            cv2.putText(frame, tag, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tcol, 2)
            cv2.putText(frame, "FPS %.1f" % det.fps, (15, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(frame, "cmd f=%.2f l=%.2f r=%.2f"
                        % (cmd_fwd, cmd_lat, cmd_rot), (15, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
            cv2.putText(frame, "lost %d%s" % (lost, "  HOLDING" if holding else ""),
                        (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255) if holding else (200, 200, 0), 1)
            if adj is not None and adj.have_scan:
                if depth_avoid_enabled:
                    c = (60, 60, 245) if adj.block_forward else (0, 200, 255)
                    txt = "DEPTH: " + adj.status
                else:
                    c = (0, 165, 255)
                    txt = "DEPTH: OFF (climb mode)"
                cv2.putText(frame, txt, (15, 142),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            climb_txt = "  CLIMB MODE (Classic gait)" if climb_mode else ""
            cv2.putText(frame, mode_txt + climb_txt, (15, CAM_H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255) if climb_mode else (0, 140, 255), 1)

            cv2.imshow("Operator Follow", frame)
            if depth_av is not None:
                cv2.imshow("Depth Avoid", depth_av.debug_view())

            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hard_stop()
        if climb_mode:
            try:
                try:
                    robot.FreeWalk()
                except TypeError:
                    robot.FreeWalk(True)
                print("[climb] FreeWalk restored on exit.")
            except Exception as exc:
                print("[climb] could not restore FreeWalk on exit: %s" % exc)
        stop_builtin_avoid(lidar_av)
        if stop_gesture_det is not None:
            stop_gesture_det.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped. Bye.")


if __name__ == "__main__":
    main()
