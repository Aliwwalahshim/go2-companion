# Go2 stair climbing - stair_mode.py / climb_stairs.py

## Phase 1 research findings (read this before the run order)

### Is there a stair-climbing command in this SDK? No.

Verified at runtime on this machine, not assumed:

```
print([m for m in dir(SportClient) if not m.startswith('_')])
```

lists no `ClimbStair`, `StairMode`, `ForwardDownStair`, or `SwitchGait`.
`gait_type` is a **read-only** field on the `rt/sportmodestate` telemetry
topic (`SportModeState_.gait_type`, confirmed in
`unitree_sdk2py/idl/unitree_go/msg/dds_/_SportModeState_.py`) - status
readback, not something you can write to. `SwitchGait` exists only on
`unitree_sdk2py.a2.sport.sport_client` and `unitree_sdk2py.as2.sport.sport_client`
(the A2/AS2 robots) - not on Go2's `go2.sport.sport_client`. No fabricated
methods are used anywhere in this deliverable.

### Does a motion_switcher client exist on this SDK? Yes - already installed, wrong import path was the problem.

The earlier claim that `unitree_sdk2py.go2.motion_switcher` raises
`ModuleNotFoundError` is correct, but that's the wrong path. The real one,
already present in this exact installed copy
(`pip3 show unitree-sdk2py` -> version 1.0.1, at
`/home/unitree/unitree_sdk2_python`, upstream repo
`unitreerobotics/unitree_sdk2_python`, currently at commit `37116c5`), is:

```python
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
```

`MotionSwitcherClient` exposes `CheckMode()`, `SelectMode(nameOrAlias)`,
`ReleaseMode()` (`motion_switcher_client.py`), and it is already used by
several official examples shipped in this same repo, including
`example/go2/low_level/go2_stand_example.py` and
`example/motionSwitcher/motion_switcher_example.py`, which is Unitree's own
worked example of switching to AI mode:

```python
msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()
ret = msc.SelectMode("ai")     # "normal" | "advanced" | "ai" | "ai-w" (wheeled robots)
```

`CheckMode()` returns `(code, {"name": "ai" | "normal" | "advanced" | ""})`
- `go2_stand_example.py` polls `result['name']` to detect whether a
high-level mode is currently claimed. **No SDK upgrade is needed** - the
capability was already sitting in the installed package under a path
nobody had looked in yet. This is the cleanest possible route and is what
`stair_mode.py` uses.

### go2_webrtc_connect / phospho-app fork / unitree_webrtc_connect

`legion1581/go2_webrtc_connect` (and the `phospho-app` fork, and the
successor `legion1581/unitree_webrtc_connect`) speak the mobile app's
WebRTC protocol and do support Go2 AIR/PRO/EDU on current firmware. This is
a real alternative path to AI mode switching that does not depend on the
DDS SportClient at all. However, since this SDK already has a working,
documented, first-party `MotionSwitcherClient.SelectMode("ai")` call, there
is no reason to add a second transport (WebRTC, a different auth/session
model, another dependency tree) for the same one RPC call. Worth keeping in
mind if this project ever needs something WebRTC-only exposes (e.g. video
over Wi-Fi without Ethernet), but not adopted here.

One caveat found during research and worth flagging: for the cheaper **Go2
AIR** variant, `ai_sport` / `motion_switcher` services are reportedly not
present in the stock rootfs and need a custom firmware/package to unlock
(per community docs referencing AIR specifically) - this does not apply to
**Go2 EDU**, which is what this project targets, and where
`motion_switcher` is already a shipped, unmodified system service (as
evidenced by `msc.Init()`/`SelectMode()` working against stock firmware in
Unitree's own example scripts, which assume no jailbreak).

### robot-com-projects/go2_ros2_sdk

A fork of `abizovnuralem/go2_ros2_sdk`. Targets ROS2 Humble/Iron/Rolling,
not Foxy - not a drop-in fit for this project's ROS2 Foxy environment
without porting work. It does not expose AI sport mode, motion switching,
or stair climbing; it focuses on teleop, SLAM, and general navigation over
WebRTC/CycloneDDS. Low independent activity (it's a thin fork). Not
adopted.

### jizhang-cmu/autonomy_stack_go2

The `foxy-humble` branch is explicitly Foxy-compatible (matches this
project's ROS2 distro) and uses only the Go2's built-in L1 lidar + IMU - no
extra sensors required. But it implements its own SLAM/waypoint/collision-
avoidance stack, not the Unitree AI-mode stair gait, and its own docs state
the terrain traversability analysis "cannot distinguish low obstacles" and
has a **minimum obstacle height of 0.3 m** - meaning its own avoidance logic
would most likely treat a staircase as an obstacle to route around, not
climb. Not usable as-is for this task; it solves a different problem
(point-to-point autonomous navigation, not stair traversal).

### Existing open-source Go2 stair-climbing implementations

**None found.** Every SDK-level reference to "stair climbing" in the
Go2 ecosystem traces back to the same thing: the onboard `gait_type == 3`
(climb stair) status value, selected automatically by the robot's own
controller while it is in AI mode - not a community-built gait or a
documented API call anyone has wrapped in code. A relevant, if informal,
data point: a MYBOTSHOP forum thread ("No more stair climbing mode after
update?") confirms that Unitree itself folded the old dedicated stair mode
into AI mode in a firmware update, and that a moderator's response there
matches this project's framing exactly - "you do not need to go into a
separate mode now [for stairs], it's combined into AI mode." That same
thread contains an unverified report that leaving the onboard obstacle-
avoidance switch on can make the robot stop in front of stairs instead of
climbing them; `stair_mode.py`'s `disable_onboard_avoid` option exists
because of that report, off by default, and is documented as unverified
below.

### Conclusion: is AI mode reachable on this firmware, and by what call?

**Yes, reachable, by:**

```python
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
code, _ = msc.SelectMode("ai")
code, result = msc.CheckMode()   # result["name"] == "ai" confirms it
```

This is exactly what `stair_mode.enter_stair_mode()` does, plus retrying
`CheckMode()` for up to 3s before giving up, and refusing to let a caller
send any `Move()` if it never confirms.

---

## What was built

- **`stair_mode.py`** - `enter_stair_mode()`, `exit_stair_mode()`,
  `is_stair_mode_active()`, and a `StairMode` context manager. Verifies the
  switch with `CheckMode()` before returning; raises `StairModeError` and
  sends nothing if it can't confirm. Restoration (`ReleaseMode()` +
  re-selecting whatever mode was active before) always happens in
  `exit_stair_mode()`, and the context manager guarantees that runs even on
  an exception or Ctrl+C.
- **`climb_stairs.py`** - standalone demo. Enters stair mode, sends capped
  slow forward `Move()` commands for a fixed duration while the robot's own
  controller does the gait, stops, exits stair mode. All of it has a
  `--dry-run` path that never opens a network channel.

## Run order

Only one terminal is needed for this feature - no ROS2 involved.

**Terminal 1 - plain terminal, NOT foxy-sourced** (this talks DDS directly
to the robot; sourcing ROS2 Foxy or setting `CYCLONEDDS_URI` here is exactly
the mistake documented in the hard-constraints section below):

```bash
cd /home/unitree/unitree_sdk2_python/example/go2/front_camera

# 1. dry run first, always - proves the mode switch/verify/restore logic
#    works without touching the robot at all
python3 climb_stairs.py --dry-run --duration 3

# 2. real run, ascent only, human holding the remote
python3 climb_stairs.py eth0 --speed 0.2 --duration 8
```

## Network state required

- `eth0` must carry **only** `192.168.123.18/24` while any of this runs.
  Check with `ip -4 addr show eth0`. If a second address such as
  `192.168.1.102/24` (the RoboSense LiDAR subnet) is also present, Unitree's
  DDS will bind to the wrong address and every SDK call - including
  `MotionSwitcherClient.Init()`/`SelectMode()` - will return error `3102`,
  while `ping` and the handheld remote keep working normally. Remove the
  extra address before running this.
- Confirm no other script already holds the robot (`go2_master.py`,
  `follow_depth_avoid.py`, `go2_agent.py`, etc.) - `error 3102` is also what
  you get when a second process is competing for the same lease/session.
  `pgrep -af "go2_master.py|follow_depth_avoid.py|go2_agent.py|climb_stairs.py"`
  before starting.
- This module never imports `rclpy`, and must not be imported from a
  process that does - `rclpy` and the Unitree DDS layer collide on
  CycloneDDS if they share a process. Nothing here needs ROS2 at all.

## Safety requirements implemented

- **Ascent only by default.** `--allow-descent` exists to acknowledge the
  risk (and prints a loud warning) but does not change any command sent -
  going downstairs is still forward walking, just aimed the other way. That
  choice of where to point the robot is on the operator, not the code.
- **Speed cap.** `MAX_STAIR_SPEED = 0.3` m/s in `climb_stairs.py`, enforced
  by `clamp_forward()` regardless of `--speed`.
- **No reverse, ever.** `clamp_forward()` floors at `0.0`; nothing in this
  deliverable ever passes a negative `vx` to `Move()`.
- **Mode restoration guaranteed.** `exit_stair_mode()` runs in a `finally`
  block in `climb_stairs.py`'s `climb()`, and the `StairMode` context
  manager's `__exit__` always calls it too, so `with StairMode():` cannot
  leave the robot in AI mode on any exit path.
- **Refuse to move on unverified mode.** If `CheckMode()` never reports
  `"ai"` within ~3s of `SelectMode("ai")` returning success,
  `enter_stair_mode()` raises before `climb_stairs.py` sends a single
  `Move()`.
- **Physical e-stop.** L2+B on the handheld remote is the physical e-stop.
  A human must be present with the remote in hand for every stair test -
  `climb_stairs.py` prints this and pauses on `input()` before opening the
  channel in real (non-dry-run) mode, but that prompt is a reminder, not a
  substitute for someone actually holding the remote.

## Hard constraints from this project (respected here)

- `rclpy` and the Unitree DDS layer cannot share one process - this module
  never imports `rclpy`.
- Do not source ROS2 Foxy, and do not set `CYCLONEDDS_URI`, in the terminal
  running `climb_stairs.py` - Foxy's CycloneDDS rejects the newer
  `<NetworkInterface name=...>` element.
- `eth0` must carry only `192.168.123.18/24` while controlling the robot -
  see "Network state required" above.
- Only one process may hold the robot at a time - check before blaming
  error `3102` on this code.
- ASCII only - verified below.
- Python 3.8 syntax only - no `match`, no `list[str]` annotations. Verified
  by `python3 -m py_compile` on both files.

```bash
python3 -m py_compile stair_mode.py climb_stairs.py   # both: clean
python3 -c "open('stair_mode.py','rb').read().decode('ascii')"    # no smart quotes/NBSP
python3 -c "open('climb_stairs.py','rb').read().decode('ascii')"  # no smart quotes/NBSP
```

## How this would be wired into the existing scripts (not done yet)

`go2_master.py`, `follow_depth_avoid.py`, `go2_agent.py` /
`go2_mcp_server.py`, and the pose/waypoint scripts were **not modified**, as
instructed. If stair mode should be reachable from any of them later, the
integration point is small because `stair_mode.py` already exposes a plain
function interface with no required arguments:

- **`go2_agent.py` / `go2_mcp_server.py`** (natural-language / MCP control):
  add one new MCP tool, e.g. `climb_stairs(duration, speed)`, whose handler
  is a thin wrapper around `stair_mode.enter_stair_mode()` /
  `sport.Move(...)` / `stair_mode.exit_stair_mode()` - same shape as
  `climb_stairs.climb()`. This is the most natural fit, since that process
  already owns a live `SportClient` and channel.
- **`go2_master.py`** (gesture console): a new `BINDINGS` entry (e.g. hold
  a specific gesture) that calls `stair_mode.enter_stair_mode()` before
  driving and `stair_mode.exit_stair_mode()` on release/exit, following the
  same non-blocking-deadline pattern already used for the HandStand
  auto-revert. Would need its own explicit confirmation step given the
  ascent/descent risk, not just a gesture toggle.
- **`follow_depth_avoid.py`**: not a good fit as-is. Its whole design is
  `DepthAvoider` gating forward motion off RealSense depth, which has no
  concept of a staircase and would very likely read the first step's rise
  as an obstacle and refuse to advance. It would need `DepthAvoider` bypassed
  entirely while `stair_mode` is active, not layered with it - out of scope
  here, flagged for later if requested.

None of this is wired in. Wiring it into a live-gesture or live-agent
control path multiplies the blast radius of getting the mode switch wrong,
and the instructions were explicit not to do it yet.

## Honest limitations

**What was tested:**
- `stair_mode.py` and `climb_stairs.py` both compile clean under Python 3.8
  (`py_compile`) and contain only ASCII.
- `climb_stairs.py --dry-run` was run end to end: it prints the speed-cap
  warning when `--speed` exceeds the cap, prints the descent warning under
  `--allow-descent`, enters and exits the dry-run stair-mode session
  cleanly, and every printed `Move()` call shows the clamped `0.30` value
  even when `5.0` was requested - the speed cap and the never-negative
  clamp both verified in isolation.
- The exact `MotionSwitcherClient` API (`CheckMode`/`SelectMode`/
  `ReleaseMode`) and its `result["name"]` response shape were confirmed by
  reading Unitree's own shipped example code
  (`go2_stand_example.py`, `go2w_stand_example.py`, `motion_switcher_example.py`),
  not guessed.
- The `rt/sportmodestate` topic name for Go2 (used nowhere in this
  deliverable's actual code, since verification goes through `CheckMode()`,
  not telemetry) was cross-checked against the SDK's own topic list at
  `/unitree/lib/unitree_go2_sdk/unitree_dds_idl/go2/0TopicList.md`, which
  documents it directly for reference if it's ever wanted for a
  gait_type readout.

**What was NOT tested:**
- **`SelectMode("ai")` was never actually sent to the physical robot in
  this session.** Everything above is dry-run/static verification plus
  reading Unitree's own example code. Whether `CheckMode()` really reports
  `"ai"` back within the 3s retry window on this specific unit, on this
  specific firmware, has not been confirmed live.
- **No real staircase test was performed.** Whether the onboard controller
  actually produces a working climb gait once in AI mode - on this
  firmware, this staircase, this battery/load state - is unverified.
- **The `--disable-avoid` / onboard-avoidance-blocks-stairs report is
  unverified on this unit.** It comes from one community forum thread, not
  from testing here. The option exists and is safely restored on exit, but
  its premise has not been confirmed.
- **Whether AI mode plays nicely with an active depth-avoidance loop
  (`depth_avoider.py`) has not been tested**, and per the limitation noted
  above under "wiring in," it's expected not to without changes.

**What the robot genuinely cannot do, as far as this research found:**
- There is no way, on this SDK, to command a specific gait (stairs or
  otherwise) - AI mode either produces the stair gait on its own when it
  perceives stairs, or it doesn't; there is no `ForceStairGait()` to fall
  back to.
- Reliable descent is not something this project can deliver by switching
  modes and driving - it is a hardware/onboard-controller limitation
  repeatedly reported by the community, not something `climb_stairs.py`
  can compensate for. That's exactly why descent stays behind an explicit,
  loudly-warned opt-in rather than being treated as symmetric with ascent.
