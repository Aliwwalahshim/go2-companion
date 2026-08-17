"""
go2_mic.py - capture audio from the Go2's OWN microphone over WebRTC.

WHY THIS FILE EXISTS
--------------------
The Go2's mic (like its head speaker) is NOT an ALSA device on this Jetson.
Verified: `arecord -l` shows only the Tegra APE internal DMA crossbar (no mic
codec), and recording from the default device yields pure digital silence
(peak 0). The Jetson is an add-on computer; the real microphone is wired to
the robot's MAIN computer (192.168.123.161). The only way to reach it is the
same WebRTC path the head speaker uses (see go2_speaker.py) - the mobile
app's protocol, keyless on this LAN.

WHAT IT DOES
------------
Two ways to consume the mic:

1. Record a fixed clip to a WAV file:
       mic = Go2Mic()
       res = mic.record("/tmp/clip.wav", seconds=5)   # -> dict or None
       mic.close()

2. STREAM live audio to a callback (e.g. feed speech recognition):
       def on_audio(pcm):       # pcm = 48 kHz mono s16 bytes
           ...                  # runs on a background thread - keep it light
       mic = Go2Mic()
       mic.start_stream(on_audio)
       ...                      # audio flows until you stop
       mic.stop_stream()
       mic.close()

ARCHITECTURE
------------
The WebRTC/aiortc event loop runs in a SEPARATE worker process (its stdout,
and the driver's noisy logging, go to /dev/null; structured status comes back
on stderr). Same pattern as go2_speaker.py - keeps aiortc out of the DDS
control process, so this is safe to import alongside
ChannelFactoryInitialize(0, "eth0"). For streaming, raw PCM is sent from the
worker to the parent over a dedicated binary pipe (length-prefixed frames),
separate from the JSON status channel, so the driver's stdout spam can never
corrupt the audio.

Audio is always delivered as 48 kHz, mono, signed-16-bit little-endian PCM
(Go2Mic.SAMPLE_RATE).

CLI
---
    python3 go2_mic.py --seconds 5 --out /tmp/clip.wav      # record a clip
    python3 go2_mic.py --stream --seconds 10                # live VU meter
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROBOT_IP = "192.168.123.161"


# =======================================================================
# Parent-side handle
# =======================================================================
class Go2Mic:
    SAMPLE_RATE = 48000          # Hz, mono, s16 LE - what callbacks receive

    def __init__(self, robot_ip=ROBOT_IP, connect_timeout=30.0):
        self.robot_ip = robot_ip
        self._proc = None
        self._available = False
        self._send_lock = threading.Lock()
        self._result_event = threading.Event()
        self._last_result = None
        # streaming
        self._pcm_r = None
        self._stream_cb = None
        self._stream_thread = None
        self._stream_stop = threading.Event()
        self._start_worker(connect_timeout)
        if not self._available:
            print("[go2_mic] robot mic unavailable (WebRTC audio channel "
                  "did not come up).")

    def _start_worker(self, timeout):
        try:
            pcm_r, pcm_w = os.pipe()
            self._proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 "--worker", "--ip", self.robot_ip, "--pcm-fd", str(pcm_w)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, universal_newlines=True, bufsize=1,
                pass_fds=(pcm_w,))
            os.close(pcm_w)          # parent keeps only the read end
            self._pcm_r = pcm_r
        except Exception as exc:
            print("[go2_mic] could not spawn worker: %s" % exc)
            self._proc = None
            return

        ready = threading.Event()
        state = {"available": False}

        def reader():
            for line in self._proc.stderr:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue          # ignore warnings / non-JSON noise
                st = obj.get("status")
                if st == "ready":
                    state["available"] = True
                    ready.set()
                elif st == "unavailable":
                    state["available"] = False
                    ready.set()
                elif st == "recorded" or st == "error":
                    self._last_result = obj
                    self._result_event.set()

        threading.Thread(target=reader, daemon=True).start()
        ready.wait(timeout)
        self._available = state["available"]
        if not self._available:
            self._stop_worker()

    def _stop_worker(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except Exception:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _worker_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _send(self, msg):
        with self._send_lock:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()

    # ---- public API ----
    def is_available(self):
        return bool(self._available and self._worker_alive())

    def record(self, path, seconds=5.0, timeout=None):
        """Record `seconds` of the robot mic to `path` (48 kHz mono WAV).
        Blocks until done. Returns a result dict or None on failure."""
        if not self.is_available():
            return None
        path = os.path.abspath(path)
        if timeout is None:
            timeout = seconds + 20.0
        self._result_event.clear()
        self._last_result = None
        try:
            self._send({"cmd": "record", "path": path,
                        "seconds": float(seconds)})
        except Exception:
            self._available = False
            return None
        if not self._result_event.wait(timeout):
            return None
        res = self._last_result
        if not res or res.get("status") != "recorded":
            return None
        return res

    # ---- streaming ----
    def start_stream(self, callback):
        """Begin continuous mic streaming. `callback(pcm_bytes)` is called
        from a background thread with 48 kHz mono s16 PCM chunks as they
        arrive (see SAMPLE_RATE). Keep the callback light - do heavy work
        (ASR, etc.) on your own queue/thread. Returns True if streaming
        started. Call stop_stream() to end."""
        if not self.is_available() or self._pcm_r is None:
            return False
        if self._stream_thread is not None:
            return True                     # already streaming
        self._stream_cb = callback
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._pcm_reader, daemon=True)
        self._stream_thread.start()
        try:
            self._send({"cmd": "stream_on"})
        except Exception:
            self._available = False
            return False
        return True

    def stop_stream(self):
        if self._stream_thread is None:
            return
        try:
            self._send({"cmd": "stream_off"})
        except Exception:
            pass
        self._stream_stop.set()
        self._stream_cb = None
        self._stream_thread = None
        # the reader thread may be blocked in os.read until the next frame or
        # until the worker exits; it is a daemon and unblocks on close().

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = os.read(self._pcm_r, n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None             # worker closed the pipe
            buf += chunk
        return buf

    def _pcm_reader(self):
        while not self._stream_stop.is_set():
            hdr = self._read_exact(4)
            if hdr is None:
                break
            n = int.from_bytes(hdr, "little")
            if n <= 0 or n > 4000000:
                break
            data = self._read_exact(n)
            if data is None:
                break
            cb = self._stream_cb
            if cb is not None:
                try:
                    cb(data)
                except Exception:
                    pass

    def close(self):
        self.stop_stream()
        if self._worker_alive():
            try:
                self._send({"cmd": "quit"})
                self._proc.wait(timeout=3.0)
            except Exception:
                pass
        self._stop_worker()
        if self._pcm_r is not None:
            try:
                os.close(self._pcm_r)
            except Exception:
                pass
            self._pcm_r = None


# =======================================================================
# Worker process (aiortc event loop; stdout -> /dev/null)
# =======================================================================
def _worker_main(robot_ip, pcm_fd):
    import asyncio
    import logging
    import warnings
    import wave

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    status = sys.stderr
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)

    def emit(obj):
        status.write(json.dumps(obj) + "\n")
        status.flush()

    def write_pcm(data):
        # length-prefixed frame on the dedicated binary pipe; single writer,
        # so ordering is safe. Loop in case a big write is partial.
        payload = len(data).to_bytes(4, "little") + data
        while payload:
            try:
                n = os.write(pcm_fd, payload)
            except OSError:
                return
            payload = payload[n:]

    import av
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection, WebRTCConnectionMethod)

    # Shared capture state; the driver's on("track") loop pushes frames here
    # via our callback whenever the audio channel is on.
    cap = {"record": False, "stream": False, "pcm": None, "frames": 0}
    resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)

    async def on_frame(frame):
        if not (cap["record"] or cap["stream"]):
            return
        cap["frames"] += 1
        try:
            for f in resampler.resample(frame):
                pcm = bytes(f.planes[0])[: f.samples * 2]
                if cap["record"] and cap["pcm"] is not None:
                    cap["pcm"].extend(pcm)
                if cap["stream"]:
                    write_pcm(pcm)
        except Exception:
            pass

    async def do_record(path, seconds):
        import audioop
        cap["pcm"] = bytearray()
        cap["frames"] = 0
        cap["record"] = True
        await asyncio.sleep(seconds)
        cap["record"] = False
        pcm = bytes(cap["pcm"])
        cap["pcm"] = None
        if not pcm:
            emit({"status": "error", "error": "no audio frames received"})
            return
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes(pcm)
        except Exception as exc:
            emit({"status": "error", "error": "write failed: %s" % exc})
            return
        emit({"status": "recorded", "path": path,
              "seconds": len(pcm) / 2.0 / 48000.0,
              "peak": audioop.max(pcm, 2), "rms": audioop.rms(pcm, 2),
              "frames": cap["frames"]})

    async def run():
        try:
            conn = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
            await asyncio.wait_for(conn.connect(), timeout=25.0)
            conn.audio.add_track_callback(on_frame)
            conn.audio.switchAudioChannel(True)   # start streaming the mic
        except Exception as exc:
            emit({"status": "unavailable", "error": str(exc)})
            return
        emit({"status": "ready", "available": True})

        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def stdin_reader():
            for line in sys.stdin:
                loop.call_soon_threadsafe(q.put_nowait, line.strip())
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=stdin_reader, daemon=True).start()

        while True:
            line = await q.get()
            if line is None:
                break
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            cmd = msg.get("cmd")
            try:
                if cmd == "quit":
                    break
                elif cmd == "record":
                    await do_record(msg.get("path", "/tmp/go2_mic.wav"),
                                    float(msg.get("seconds", 5.0)))
                elif cmd == "stream_on":
                    cap["stream"] = True
                    emit({"status": "streaming", "rate": 48000})
                elif cmd == "stream_off":
                    cap["stream"] = False
                    emit({"status": "stream_stopped"})
            except Exception as exc:
                emit({"status": "error", "error": str(exc)})

        try:
            conn.audio.switchAudioChannel(False)
        except Exception:
            pass
        try:
            await conn.disconnect()
        except Exception:
            pass

    asyncio.run(run())


# =======================================================================
# CLI
# =======================================================================
def main():
    ap = argparse.ArgumentParser(description="Capture the Go2 mic over WebRTC")
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default="/tmp/go2_mic.wav")
    ap.add_argument("--stream", action="store_true",
                    help="live VU-meter demo of the streaming callback API")
    ap.add_argument("--worker", action="store_true",
                    help="INTERNAL: run the WebRTC worker process")
    ap.add_argument("--pcm-fd", type=int, default=-1,
                    help="INTERNAL: fd for streamed PCM")
    args = ap.parse_args()

    if args.worker:
        _worker_main(args.ip, args.pcm_fd)
        return 0

    mic = Go2Mic(robot_ip=args.ip)
    print("[go2_mic] available:", mic.is_available())
    if not mic.is_available():
        return 1

    if args.stream:
        import audioop
        print("[go2_mic] streaming %.1fs (live level, talk to the robot) ..."
              % args.seconds)

        def on_audio(pcm):
            rms = audioop.rms(pcm, 2)
            bars = int(min(40, rms / 50))
            sys.stdout.write("\r[go2_mic] level |%-40s| %5d"
                             % ("#" * bars, rms))
            sys.stdout.flush()

        mic.start_stream(on_audio)
        time.sleep(args.seconds)
        mic.stop_stream()
        mic.close()
        print("\n[go2_mic] streaming stopped.")
        return 0

    print("[go2_mic] recording %.1fs to %s ..." % (args.seconds, args.out))
    res = mic.record(args.out, seconds=args.seconds)
    mic.close()
    if res is None:
        print("[go2_mic] recording FAILED")
        return 1
    print("[go2_mic] saved %s  (%.2fs, %d frames, peak %d/32768, rms %d)"
          % (res["path"], res["seconds"], res["frames"], res["peak"],
             res["rms"]))
    if res["peak"] < 50:
        print("[go2_mic] WARNING: near-silence - mic may not have picked "
              "anything up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
