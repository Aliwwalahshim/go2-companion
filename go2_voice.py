"""
go2_voice.py - offline spoken-command recognition for the Go2.

Streams the robot's microphone (go2_mic, over WebRTC) into Vosk (fully
offline, on-device speech recognition) and fires a callback when a known
command phrase is heard. No cloud, no internet at run time.

    def on_command(cmd):          # cmd is one of your command strings
        robot.cmd(...)
    vc = VoiceCommands(on_command, commands=["stand up", "sit down", ...])
    if vc.available():
        vc.start()
        ...                       # listens until you stop
        vc.close()

HOW THE PIECES FIT
------------------
- go2_mic streams 48 kHz mono s16 PCM from the robot mic to a callback.
- Vosk wants 16 kHz mono s16, so each chunk is resampled 48k->16k (stdlib
  audioop, stateful) and pushed onto a queue.
- A recognizer thread drains the queue into a Vosk KaldiRecognizer built with
  a GRAMMAR limited to your command phrases (huge accuracy win for a fixed
  vocabulary), and calls on_command when a phrase is recognized.
- go2_mic's WebRTC runs in its own worker process, so this whole thing is safe
  to run alongside a DDS SportClient in the SAME process (put your robot calls
  in on_command). rclpy must still never share the process - not used here.

CLI
---
    python3 go2_voice.py                       # print recognized commands
    python3 go2_voice.py --seconds 30          # listen 30s then exit
    python3 go2_voice.py --speak               # also confirm via head speaker
    python3 go2_voice.py --commands "stand up,sit down,hello,stop"

Model defaults to /home/unitree/vosk_models/vosk-model-small-en-us-0.15
(override with --model).
"""

import argparse
import audioop
import json
import os
import queue
import sys
import threading
import time

from go2_mic import Go2Mic

DEFAULT_MODEL = "/home/unitree/vosk_models/vosk-model-small-en-us-0.15"
MIC_RATE = 48000        # what go2_mic delivers
ASR_RATE = 16000        # what Vosk wants

DEFAULT_COMMANDS = [
    "stand up", "sit down", "lie down", "come here", "stop",
    "hello", "turn left", "turn right", "walk forward", "dance",
    "shake hands", "well done",
]


class VoiceCommands:
    def __init__(self, on_command, commands=None, model_path=DEFAULT_MODEL,
                 robot_ip="192.168.123.161"):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)                        # silence Vosk's own logging
        if not os.path.isdir(model_path):
            raise RuntimeError("Vosk model not found at %s" % model_path)
        self.on_command = on_command
        self.commands = commands or list(DEFAULT_COMMANDS)
        self._model = Model(model_path)
        # restrict recognition to our phrases (+ [unk] for everything else)
        grammar = json.dumps(self.commands + ["[unk]"])
        self._rec = KaldiRecognizer(self._model, ASR_RATE, grammar)
        self._rec.SetWords(False)

        self.mic = Go2Mic(robot_ip=robot_ip)
        self._q = queue.Queue(maxsize=100)
        self._ratecv_state = None
        self._stop = threading.Event()
        self._thread = None

    def available(self):
        return self.mic.is_available()

    # mic callback (runs on go2_mic's pipe-reader thread - keep it light)
    def _mic_cb(self, pcm48):
        try:
            pcm16, self._ratecv_state = audioop.ratecv(
                pcm48, 2, 1, MIC_RATE, ASR_RATE, self._ratecv_state)
        except Exception:
            return
        try:
            self._q.put_nowait(pcm16)
        except queue.Full:
            pass

    def _recognize_loop(self):
        while not self._stop.is_set():
            try:
                pcm = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                final = self._rec.AcceptWaveform(pcm)
            except Exception:
                continue
            if final:
                try:
                    text = json.loads(self._rec.Result()).get("text", "").strip()
                except ValueError:
                    text = ""
                if text:
                    self._dispatch(text)

    def _dispatch(self, text):
        # fire the first command phrase that appears in the recognized text
        for cmd in self.commands:
            if cmd in text:
                try:
                    self.on_command(cmd)
                except Exception as exc:
                    print("[voice] handler error: %s" % exc)
                return

    def start(self):
        if not self.available():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._recognize_loop,
                                        daemon=True)
        self._thread.start()
        return self.mic.start_stream(self._mic_cb)

    def stop(self):
        self._stop.set()
        try:
            self.mic.stop_stream()
        except Exception:
            pass

    def close(self):
        self.stop()
        try:
            self.mic.close()
        except Exception:
            pass


# =======================================================================
# CLI demo
# =======================================================================
def main():
    ap = argparse.ArgumentParser(description="Go2 offline voice commands (Vosk)")
    ap.add_argument("--ip", default="192.168.123.161")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="listen this long then exit (0 = until Ctrl+C)")
    ap.add_argument("--commands", default="",
                    help="comma-separated command phrases (default: a built-in "
                         "demo set)")
    ap.add_argument("--speak", action="store_true",
                    help="confirm each recognized command through the robot's "
                         "head speaker (opens a second WebRTC connection)")
    args = ap.parse_args()

    commands = None
    if args.commands.strip():
        commands = [c.strip().lower() for c in args.commands.split(",")
                    if c.strip()]

    speaker = None
    if args.speak:
        try:
            from go2_speaker import Go2Speaker
            speaker = Go2Speaker(robot_ip=args.ip)
        except Exception as exc:
            print("[voice] --speak unavailable: %s" % exc)

    def on_command(cmd):
        print("[voice] COMMAND: %s" % cmd)
        sys.stdout.flush()
        if speaker is not None and speaker.is_available():
            speaker.say(cmd)

    try:
        vc = VoiceCommands(on_command, commands=commands, model_path=args.model,
                           robot_ip=args.ip)
    except Exception as exc:
        print("[voice] setup failed: %s" % exc)
        return 1

    print("[voice] mic available:", vc.available())
    if not vc.available():
        return 1
    print("[voice] listening. Say one of: %s" % ", ".join(vc.commands))
    if not vc.start():
        print("[voice] could not start the mic stream")
        vc.close()
        return 1

    try:
        if args.seconds > 0:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        vc.close()
        if speaker is not None:
            speaker.close()
        print("\n[voice] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
