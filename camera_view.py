"""
camera_view.py - plain live camera viewer, no robot control at all.

Opens the RealSense color stream and shows it in a window. Sends no
commands to the robot, switches no mode, and does not touch
MotionSwitcherClient or SportClient - safe to run no matter what mode the
robot is currently in. Use this to watch what the robot's front camera
sees (e.g. while driving with the physical remote) without depending on
AI mode being reachable.

Press 'q' or ESC in the window, or Ctrl+C here, to quit.
"""

import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("[camera_view] camera preview window open. 'q'/ESC/Ctrl+C to quit.")
    print("[camera_view] this sends nothing to the robot - drive with the "
          "remote. L2+B on the remote is the e-stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            cv2.putText(img, "CAMERA VIEW ONLY - no commands sent to robot.",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(img, "Drive with the remote. q/ESC to quit.",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow("Go2 camera view", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[camera_view] interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
