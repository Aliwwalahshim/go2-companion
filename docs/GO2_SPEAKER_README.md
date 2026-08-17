# go2_speaker.py - speak through the Go2's own head speaker

Play TTS and WAV files through the **robot's built-in head speaker** over
WebRTC, instead of out of the Jetson's own audio jack (`espeak`).

## Phase 1 report (verified on this robot, not assumed)

1. **Firmware / scope.** Target was Go2 EDU, firmware ~1.1.16. I could not
   read an exact version string over DDS (no SDK call exposes one), but the
   robot self-reports its motion mode as `mcf` (the newer unified motion
   framework that only exists on 1.1.15+) and runs the full modern
   `webrtc_signal_server` / `webrtc_bridge` service set - both consistent
   with 1.1.16. **In scope: yes.** Confirmed a different, stronger way than a
   version match: `RobotStateClient.ServiceList()` shows an **`audio_hub`**
   service actually present on this unit.
2. **AES-128 key: NOT needed here.** Firmware >= 1.1.15 *can* require a
   per-device cloud key (`data2=3` handshake). On this robot, over the wired
   LAN, the **keyless `con_notify` handshake (`192.168.123.161:9991`)
   succeeds** and the data channel verifies OK - tested directly. So
   `unitree-fetch-aes-key` and Unitree account credentials were **not
   required**. (If a future firmware update breaks the keyless path, that
   CLI is the fallback: it needs your Unitree app email/password to fetch the
   key, which then goes into `UnitreeWebRTCConnection(..., aes_128_key=...)`.)
3. **Chosen fork.** `legion1581`'s work, shipped on PyPI as
   **`unitree-webrtc-connect` (2.1.2, 2026-05-17)**. The `go2_webrtc_connect`
   repo's own current examples import `unitree_webrtc_connect`; the repo was
   consolidated into that one maintained package. The `phospho-app` fork is
   the older pre-rename code - not used. `requires-python = ">=3.8"`.
4. **The exact audio API.** Not a live PCM stream (that's the G1 path). This
   robot uses `WebRTCAudioHub` over the WebRTC **data channel** (topic
   `rt/api/audiohub/request`). The working sequence, taken from the library's
   own `webrtc_audio_player.py` and reproduced here:
   - `upload_audio_file(path)` - WAV (or MP3, auto-converted). **WAV must be
     44100 Hz** (the library resamples MP3 to 44100; we generate WAV at
     44100 mono/16-bit to match). Uploaded base64 in ~4 KB chunks.
   - the upload response does **not** contain a usable id, so you then call
     `get_audio_list()` and match your clip by `CUSTOM_NAME` to get its
     `UNIQUE_ID`.
   - `play_by_uuid(uuid)` plays it out the head speaker.
   - `delete_record(uuid)` removes it from robot storage.
5. **Python 3.8:** supported (`>=3.8`); Jetson is 3.8.10. `pyaudio` built a
   wheel cleanly once `portaudio19-dev` was installed. **RAM:** measured
   15 GB total / ~11 GB free - the "tight on RAM" worry did not apply.

## Install

```bash
# 1. system dep for pyaudio (needs sudo - run it yourself)
sudo apt-get install -y portaudio19-dev

# 2. the WebRTC driver (no sudo)
pip3 install --user unitree-webrtc-connect
```

`espeak` (already on this Jetson) is used to synthesize the WAV locally - no
cloud TTS. `ffmpeg` is **not** required (we feed WAV, never MP3).

## Run

```bash
cd /home/unitree/unitree_sdk2_python/example/go2/front_camera

# quick self-test - speaks one line out the robot head
python3 go2_speaker.py --say "Robot speaker is online."

# fuller test: fresh phrase, same phrase (cached), second phrase, timing
python3 go2_speaker_test.py

# force the espeak-on-Jetson fallback (proves graceful degradation)
python3 go2_speaker_test.py --no-webrtc

# clear all clips this module uploaded to the robot
python3 go2_speaker.py --purge
```

No ROS2, no foxy sourcing. `eth0` must carry **only** `192.168.123.18/24`
(the RoboSense `192.168.1.102` second address makes every call fail with
error 3102). The robot is reached at `192.168.123.161`.

## API

```python
from go2_speaker import Go2Speaker

sp = Go2Speaker()              # connects in a background worker process
sp.is_available()              # True if the robot speaker channel is up
sp.say("Sitting down")         # non-blocking; returns immediately
sp.play_wav("/path/clip.wav")  # non-blocking
sp.wait_idle(8.0)              # block until queued speech has finished
sp.purge()                     # delete uploaded clips off the robot
sp.close()                     # drain, then shut the worker down
```

- **Non-blocking:** `say()` / `play_wav()` return in ~0 ms (measured 0.000 s);
  the WebRTC work runs in a **separate process**, so a 3 s announcement never
  stalls a 30 FPS loop.
- **Automatic fallback:** if the WebRTC channel can't be established, `say()`
  falls back to local `espeak` and `play_wav()` to `aplay`. Speech never
  crashes the caller.
- **Fast repeats:** each distinct phrase uploads once (named `tts_<hash>`)
  and replays by id afterward - instant, and persistent across restarts
  because the worker rebuilds its id cache from clips already on the robot.

## Integration into go2_master.py (Phase 4 - NOT applied yet)

`go2_master.py` already routes every announcement through one function and a
single speech object whose interface (`say`, `wait_idle`) `Go2Speaker`
matches exactly. Minimal diff:

```python
# near the top, with the other imports
from go2_speaker import Go2Speaker

# replace this line (currently ~line 364):
#   _speech = SpeechWorker()
_speech = Go2Speaker()          # robot head speaker, espeak auto-fallback

# announce() (~line 367) needs NO change - _speech.say() is the same call.
# The existing _speech.wait_idle(3.0) calls at the exit paths still work.

# add one line in main()'s finally block, after the last wait_idle:
    _speech.close()             # drains, then stops the worker process
```

Because `Go2Speaker` falls back to `espeak` internally, this keeps working
even if WebRTC is down - behaviourally a superset of the current
`SpeechWorker`. The old `SpeechWorker` class can stay in the file unused, or
be deleted. **This change is not applied** - the standalone module is
confirmed working on the real robot (all test lines heard clearly), so it is
ready to wire in on request.

## Honest limitations

- **First-utterance latency.** A phrase never spoken before is uploaded in
  ~4 KB chunks at ~0.1 s/chunk, so the upload runs at very roughly 2-3x the
  clip's real-time length before it plays (a ~3 s line ~= a few seconds of
  upload). **Cached replays are immediate.** For a fixed-vocabulary gesture
  console this means only the *first* time each phrase is ever used is slow;
  everything after is instant, including across restarts. If first-time
  latency ever matters, pre-warm by `say()`-ing the vocabulary once at
  startup, or pre-`upload` the clips.
- **Volume.** Not set by this module. Robot speaker volume is adjustable via
  the DDS `VuiClient.SetVolume()` / `GetVolume()` (already in the SDK) or the
  Unitree app; it was left at whatever the robot was already set to (heard
  clearly in testing). Wiring `SetVolume` in is easy if you want it, but it
  touches a different (DDS) service and was kept out of scope.
- **Coexistence with DDS: verified OK.** A single process ran
  `ChannelFactoryInitialize(0, "eth0")` + `SportClient` (DDS) *and* a live
  `Go2Speaker` (WebRTC) at the same time; a DDS call returned normally with
  WebRTC connected, and speech played - no CycloneDDS/aiortc conflict,
  because the WebRTC stack is isolated in the child process.
- **Not tested:** behaviour if the robot drops off the network mid-session
  (the worker would error on the next play; `is_available()` would still
  report True until a send fails - callers get the espeak fallback only on
  the *next* call after the pipe breaks). Long-run clip accumulation on the
  robot is bounded for a fixed vocabulary but unbounded if you feed it
  endless unique strings - call `purge()` periodically in that case.
- **What the robot cannot do here:** there is still no *streaming* TTS - each
  utterance is a discrete upload+play, not a spoken stream, so you cannot
  barge-in / interrupt a clip cleanly mid-word from this API (you'd play a
  new clip over it).
```
