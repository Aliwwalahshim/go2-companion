"""
go2_speaker.py - speak through the Go2's OWN head speaker over WebRTC.

WHY THIS FILE EXISTS
--------------------
The Go2 DDS SDK exposes no audio/TTS client (only vui_client.py for LED /
volume / brightness). The robot's built-in head speaker is reachable only
over the same WebRTC channel the mobile app uses. This module wraps that,
verified working on this EDU unit (firmware 1.1.16, keyless LAN con_notify
handshake - no per-device AES key needed on the local network).

WHAT IT DOES
------------
    sp = Go2Speaker()          # connects in the background
    sp.say("Sitting down")     # espeak -> WAV -> robot head speaker
    sp.play_wav("/path.wav")   # play an existing file on the robot
    sp.is_available()          # True if the WebRTC audio channel is up
    sp.close()

- Synchronous, non-blocking API. say()/play_wav() return immediately; the
  work happens in a separate worker PROCESS, so a 3s announcement never
  stalls a 30 FPS control loop.
- The WebRTC/aiortc event loop and the noisy upload logging live entirely
  in that child process (its stdout is sent to /dev/null). This is the
  project's proven "separate process" pattern (cf. pose_server/lidar_node)
  and it keeps the aiortc stack out of the DDS process, so this module is
  safe to import alongside ChannelFactoryInitialize(0, "eth0").
- AUTOMATIC FALLBACK: if the WebRTC audio channel cannot be established for
  any reason, say() falls back to local espeak on the Jetson and play_wav()
  to aplay. Speech is never the thing that crashes a robot script.

HOW say() STAYS FAST
--------------------
Uploading a clip to the robot is chunked and takes a few seconds, so each
distinct phrase is uploaded ONCE, named deterministically ("tts_"+hash),
and thereafter replayed by id instantly - including across restarts, since
the worker rebuilds its id cache from the clips already on the robot at
startup. A gesture console with a fixed vocabulary ("standing", "sitting",
...) pays the upload cost only the first time each phrase is ever spoken.
Call purge() (or run with --purge) to clear those clips off the robot.

RUN MODES
---------
    python3 go2_speaker.py                 # self-test: speak a line, exit
    python3 go2_speaker.py --say "hello"   # speak one line and exit
    python3 go2_speaker.py --purge         # delete all tts_/usr_ clips
    python3 go2_speaker.py --worker --ip 192.168.123.161   # internal only

See GO2_SPEAKER_README.md for install steps and integration notes.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

ROBOT_IP = "192.168.123.161"


# =======================================================================
# WAV helpers (stdlib only - no ffmpeg needed for plain WAV)
# =======================================================================
def make_wav(text, path, rate=44100, espeak_args=None):
    """espeak -> temp WAV, normalized to `rate`/mono/16-bit at `path`.
    Returns the clip duration in seconds."""
    import wave
    import audioop
    raw = path + ".raw.wav"
    cmd = ["espeak"]
    if espeak_args:
        cmd += list(espeak_args)
    cmd += ["-w", raw, text]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    with wave.open(raw, "rb") as w:
        ch, width, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    os.remove(raw)
    if ch == 2:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    if fr != rate:
        frames, _ = audioop.ratecv(frames, width, 1, fr, rate, None)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return len(frames) / float(width * rate)


def wav_duration(path):
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 4.0


# =======================================================================
# Parent-side handle
# =======================================================================
class Go2Speaker:
    def __init__(self, robot_ip=ROBOT_IP, enable_webrtc=True,
                 connect_timeout=30.0, espeak_args=None):
        self.robot_ip = robot_ip
        self.espeak_args = list(espeak_args) if espeak_args else []
        self._proc = None
        self._available = False
        self._send_lock = threading.Lock()
        self._idle_event = threading.Event()
        self._espeak = shutil.which("espeak") is not None
        self._aplay = shutil.which("aplay") is not None
        if enable_webrtc:
            self._start_worker(connect_timeout)
        if not self._available:
            why = "WebRTC audio unavailable" if enable_webrtc else "WebRTC disabled"
            print("[go2_speaker] %s - using local espeak fallback." % why)

    # ---- lifecycle ----
    def _start_worker(self, timeout):
        try:
            self._proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 "--worker", "--ip", self.robot_ip],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
        except Exception as exc:
            print("[go2_speaker] could not spawn worker: %s" % exc)
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
                elif st == "idle":
                    self._idle_event.set()

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

    def say(self, text):
        if not text:
            return
        if self.is_available():
            try:
                self._send({"cmd": "say", "text": text})
                return
            except Exception:
                self._available = False   # pipe broke - fall through
        self._espeak_say(text)

    def play_wav(self, path):
        path = os.path.abspath(path)
        if self.is_available():
            try:
                self._send({"cmd": "play", "path": path})
                return
            except Exception:
                self._available = False
        self._aplay_wav(path)

    def prewarm(self, texts):
        """Upload these phrases to the robot NOW (in the background, without
        playing them) so the first real say() of each is an instant cache
        hit instead of paying the multi-second upload then. Non-blocking;
        uploads happen only while nothing is actually being spoken, so they
        never delay a real announcement. No-op (already cached) on later
        runs, since clips persist on the robot."""
        if not self.is_available():
            return
        for t in texts:
            if not t:
                continue
            try:
                self._send({"cmd": "prewarm", "text": t})
            except Exception:
                self._available = False
                return

    def purge(self):
        """Delete all tts_/usr_ clips this module uploaded to the robot."""
        if self.is_available():
            try:
                self._send({"cmd": "purge"})
            except Exception:
                pass

    def wait_idle(self, timeout=8.0):
        """Block until all queued speech has finished playing, or timeout.
        The worker processes its queue strictly in order, so the 'idle'
        reply only comes back after every prior say()/play_wav() is done."""
        if not self.is_available():
            return
        self._idle_event.clear()
        try:
            self._send({"cmd": "drain"})
        except Exception:
            return
        self._idle_event.wait(timeout)

    def close(self):
        if self._worker_alive():
            # drain first so a final announcement isn't cut off mid-word
            self.wait_idle(timeout=8.0)
            try:
                self._send({"cmd": "quit"})
                self._proc.wait(timeout=3.0)
            except Exception:
                pass
        self._stop_worker()

    # ---- local fallbacks (background threads - never block caller) ----
    def _espeak_say(self, text):
        if not self._espeak:
            print("[go2_speaker] (no espeak) %s" % text)
            return
        cmd = ["espeak"] + self.espeak_args + [text]
        threading.Thread(
            target=lambda: subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            daemon=True).start()

    def _aplay_wav(self, path):
        if not self._aplay:
            print("[go2_speaker] (no aplay) cannot play %s" % path)
            return
        threading.Thread(
            target=lambda: subprocess.run(
                ["aplay", path], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL),
            daemon=True).start()


# =======================================================================
# Worker process (runs the aiortc event loop; stdout -> /dev/null)
# =======================================================================
def _worker_main(robot_ip):
    import asyncio
    import logging
    import tempfile
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    # Save the real stderr for status JSON, then send stdout (and every
    # library print / chunk dump) to /dev/null at the fd level.
    status = sys.stderr
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)

    def emit(obj):
        status.write(json.dumps(obj) + "\n")
        status.flush()

    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection, WebRTCConnectionMethod)
    from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
    try:
        from unitree_webrtc_connect.constants import AUDIO_API
        _UPLOAD_ID = AUDIO_API["UPLOAD_AUDIO_FILE"]
    except Exception:
        _UPLOAD_ID = 2001

    async def fast_upload(hub, wav_path, name, chunk=16384, gap=0.02):
        """Upload a WAV far faster than the library's upload_audio_file,
        which sends 4 KB chunks with a 0.1s pause each. We send 16 KB chunks
        with a 0.02s gap over the same audiohub data-channel request - same
        protocol, ~15-20x less dead time. This is the main lever on the
        'robot speaks a few seconds late' delay."""
        with open(wav_path, "rb") as f:
            data = f.read()
        md5 = hashlib.md5(data).hexdigest()
        b64 = base64.b64encode(data).decode("utf-8")
        blocks = [b64[i:i + chunk] for i in range(0, len(b64), chunk)]
        total = len(blocks)
        resp = None
        for i, blk in enumerate(blocks, 1):
            param = {
                "file_name": name, "file_type": "wav",
                "file_size": len(data),
                "current_block_index": i, "total_block_number": total,
                "block_content": blk, "current_block_size": len(blk),
                "file_md5": md5, "create_time": int(time.time() * 1000),
            }
            resp = await hub.data_channel.pub_sub.publish_request_new(
                "rt/api/audiohub/request",
                {"api_id": _UPLOAD_ID,
                 "parameter": json.dumps(param, ensure_ascii=True)})
            if gap:
                await asyncio.sleep(gap)
        return resp

    async def get_list(hub):
        resp = await hub.get_audio_list()
        if not isinstance(resp, dict):
            return []
        ds = resp.get("data", {}).get("data", "{}")
        try:
            return json.loads(ds).get("audio_list", [])
        except ValueError:
            return []

    async def build_cache(hub):
        cache = {}
        for a in await get_list(hub):
            nm, uid = a.get("CUSTOM_NAME"), a.get("UNIQUE_ID")
            if nm and uid and (nm.startswith("tts_") or nm.startswith("usr_")):
                cache[nm] = [uid, 4.0]
        return cache

    async def _upload_and_find(hub, wav_path, name, fast):
        try:
            if fast:
                await fast_upload(hub, wav_path, name)
            else:
                await hub.upload_audio_file(wav_path)
        except Exception:
            return None
        for a in await get_list(hub):
            if a.get("CUSTOM_NAME") == name:
                return a.get("UNIQUE_ID")
        return None

    async def ensure_uploaded(hub, cache, wav_path, name, duration):
        """Upload wav_path (basename must be `name`) if not already on the
        robot, then return its uuid. Tries the fast uploader first and falls
        back to the library's slower one if the clip didn't land. Caches."""
        if name in cache:
            return cache[name][0]
        uid = await _upload_and_find(hub, wav_path, name, True)
        if uid is None:
            uid = await _upload_and_find(hub, wav_path, name, False)
        if uid is None:
            return None
        cache[name] = [uid, duration]
        return uid

    async def do_say(hub, cache, text):
        name = "tts_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if name in cache:
            uid, dur = cache[name]
            await hub.play_by_uuid(uid)
            await asyncio.sleep(max(dur, 0.5))
            return
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, name + ".wav")
        try:
            dur = make_wav(text, wav)
            uid = await ensure_uploaded(hub, cache, wav, name, dur)
            if uid is not None:
                await hub.play_by_uuid(uid)
                await asyncio.sleep(max(dur, 0.5))
        finally:
            try:
                os.remove(wav)
                os.rmdir(tmp)
            except Exception:
                pass

    async def do_play(hub, cache, path):
        if not os.path.isfile(path):
            return
        name = "usr_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
        dur = wav_duration(path)
        uid = await ensure_uploaded(hub, cache, path, name, dur)
        # ensure_uploaded needs the basename to equal `name`; copy if not
        if uid is None and name not in cache:
            tmp = tempfile.mkdtemp()
            staged = os.path.join(tmp, name + ".wav")
            shutil.copyfile(path, staged)
            uid = await ensure_uploaded(hub, cache, staged, name, dur)
            try:
                os.remove(staged)
                os.rmdir(tmp)
            except Exception:
                pass
        if uid is not None:
            await hub.play_by_uuid(uid)
            await asyncio.sleep(max(dur, 0.5))

    async def do_prewarm(hub, cache, text):
        if not text:
            return
        name = "tts_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if name in cache:
            return          # already on the robot - nothing to do
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, name + ".wav")
        try:
            dur = make_wav(text, wav)
            await ensure_uploaded(hub, cache, wav, name, dur)
        except Exception:
            pass
        finally:
            try:
                os.remove(wav)
                os.rmdir(tmp)
            except Exception:
                pass

    async def do_purge(hub, cache):
        for a in await get_list(hub):
            nm, uid = a.get("CUSTOM_NAME"), a.get("UNIQUE_ID")
            if uid and nm and (nm.startswith("tts_") or nm.startswith("usr_")):
                try:
                    await hub.delete_record(uid)
                except Exception:
                    pass
        cache.clear()

    async def run():
        try:
            conn = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
            await asyncio.wait_for(conn.connect(), timeout=25.0)
            hub = WebRTCAudioHub(conn)
            cache = await build_cache(hub)
        except Exception as exc:
            emit({"status": "unavailable", "error": str(exc)})
            return
        # The robot's audio hub defaults to a looping play mode, which makes
        # every clip repeat until the next one is triggered. Force play-once.
        try:
            await hub.set_play_mode("no_cycle")
        except Exception:
            pass
        emit({"status": "ready", "available": True})

        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def stdin_reader():
            for line in sys.stdin:
                loop.call_soon_threadsafe(q.put_nowait, line.strip())
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=stdin_reader, daemon=True).start()

        # Real say()/play() commands (in q) always take priority; queued
        # prewarm uploads only run when nothing is waiting to be spoken, so
        # they fill the cache in the background without delaying a real
        # announcement.
        prewarm_q = []

        while True:
            if not q.empty():
                line = q.get_nowait()
            elif prewarm_q:
                await do_prewarm(hub, cache, prewarm_q.pop(0))
                continue
            else:
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
                elif cmd == "say":
                    await do_say(hub, cache, msg.get("text", ""))
                elif cmd == "play":
                    await do_play(hub, cache, msg.get("path", ""))
                elif cmd == "prewarm":
                    prewarm_q.append(msg.get("text", ""))
                elif cmd == "purge":
                    await do_purge(hub, cache)
                    prewarm_q = []
                elif cmd == "drain":
                    # reached only after all earlier jobs finished (FIFO)
                    emit({"status": "idle"})
            except Exception as exc:
                emit({"status": "error", "error": str(exc)})

        try:
            await conn.disconnect()
        except Exception:
            pass

    asyncio.run(run())


# =======================================================================
# CLI
# =======================================================================
def main():
    ap = argparse.ArgumentParser(description="Go2 head-speaker TTS over WebRTC")
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--say", help="speak one line and exit")
    ap.add_argument("--play", help="play a WAV file and exit")
    ap.add_argument("--purge", action="store_true",
                    help="delete all tts_/usr_ clips off the robot and exit")
    ap.add_argument("--no-webrtc", action="store_true",
                    help="skip WebRTC, force local espeak (test the fallback)")
    ap.add_argument("--worker", action="store_true",
                    help="INTERNAL: run the WebRTC worker process")
    args = ap.parse_args()

    if args.worker:
        _worker_main(args.ip)
        return 0

    sp = Go2Speaker(robot_ip=args.ip, enable_webrtc=not args.no_webrtc)
    print("[go2_speaker] available (robot speaker):", sp.is_available())
    try:
        if args.purge:
            sp.purge()
            time.sleep(2.0)
        elif args.play:
            sp.play_wav(args.play)
            time.sleep(wav_duration(args.play) + 3.0)
        else:
            sp.say(args.say or "Robot speaker is online.")
            time.sleep(6.0)
    finally:
        sp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
