"""
go2_speaker_test.py - standalone check for go2_speaker.Go2Speaker.

Speaks three lines through the robot's head speaker:
  1. a fresh phrase (first time -> uploaded to the robot)
  2. the SAME phrase again (should replay from cache, no re-upload)
  3. a second fresh phrase
Times each say() call to show it returns immediately (non-blocking).

    python3 go2_speaker_test.py            # through the robot speaker
    python3 go2_speaker_test.py --no-webrtc  # force the espeak fallback
"""
import argparse
import time

from go2_speaker import Go2Speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.123.161")
    ap.add_argument("--no-webrtc", action="store_true")
    args = ap.parse_args()

    print("[test] connecting ...")
    sp = Go2Speaker(robot_ip=args.ip, enable_webrtc=not args.no_webrtc)
    print("[test] is_available (robot speaker):", sp.is_available())

    phrase_a = "Standing up. Keep the area clear."
    phrase_b = "Sitting down now."

    for label, text, wait in [
            ("A fresh ", phrase_a, 5.0),
            ("A cached", phrase_a, 5.0),
            ("B fresh ", phrase_b, 5.0)]:
        t0 = time.time()
        sp.say(text)
        dt = time.time() - t0
        print("[test] say(%s) returned in %.3fs -> %r" % (label, dt, text))
        time.sleep(wait)

    print("[test] closing ...")
    sp.close()
    print("[test] done. Did you hear all three lines from the robot head?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
