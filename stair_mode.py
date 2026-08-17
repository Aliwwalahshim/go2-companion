"""
stair_mode.py - switch the Go2 into AI Sport mode (the mode that unlocks
the robot's own onboard stair gait) and back again, safely.

WHY THIS FILE EXISTS
--------------------
Go2's high-level SportClient has no ClimbStair/StairMode/SwitchGait command
on this SDK. Verified at runtime, not assumed:

    print([m for m in dir(SportClient) if not m.startswith('_')])

lists no such method. gait_type is a read-only field on the
rt/sportmodestate telemetry topic (SportModeState_.gait_type) - it is
status readback, not a command interface, and it cannot be written to.
SwitchGait exists only on unitree_sdk2py.a2.sport.sport_client and
unitree_sdk2py.as2.sport.sport_client (the A2/AS2 robots), not on Go2's
go2.sport.sport_client.

The real capability lives in "ai" mode, the same mode the Unitree phone app
uses. While active, the robot's own controller selects gaits (including the
stair-climbing gait) automatically using its onboard 4D LiDAR terrain
perception. Getting there is one RPC call - MotionSwitcherClient.SelectMode
("ai") - documented and shipped in this exact SDK installation at
unitree_sdk2py/comm/motion_switcher/. This module's job is to make that
switch safely (verify it actually took effect, restore the previous mode no
matter how the caller exits) and nothing more. It does not, and cannot,
implement a gait - see STAIR_MODE_README.md for the full research trail.

USAGE
-----
Caller must already have run ChannelFactoryInitialize(0, interface) once
per process, same as every other script in this directory (Robot.__init__
in go2_master.py does the same thing) - this module does not do it for you,
and must never be imported from a process that also imports rclpy (see
STAIR_MODE_README.md, "hard constraints").

    import stair_mode
    try:
        stair_mode.enter_stair_mode()
    except stair_mode.StairModeError as exc:
        print("refusing to move:", exc)
    else:
        sport.Move(0.2, 0.0, 0.0)
    finally:
        stair_mode.exit_stair_mode()

or, so you cannot forget the finally:

    with stair_mode.StairMode():
        sport.Move(0.2, 0.0, 0.0)
"""

import time

MODE_AI = "ai"

CHECK_RETRIES = 6
CHECK_RETRY_DELAY = 0.5

# module-level client handles - created lazily on first real (non dry-run)
# call, so importing this file never touches the network by itself.
_msc = None
_avoid = None

# state of the currently-open stair-mode session, if any
_active = False
_dry = False
_prev_mode = None
_prev_avoid_state = None


class StairModeError(RuntimeError):
    """Raised when the mode switch could not be verified. Callers must
    treat this as 'do not send any movement command'."""


def _get_msc():
    global _msc
    if _msc is None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        _msc = MotionSwitcherClient()
        _msc.SetTimeout(5.0)
        _msc.Init()
    return _msc


def _get_avoid():
    global _avoid
    if _avoid is None:
        from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
            ObstaclesAvoidClient,
        )
        _avoid = ObstaclesAvoidClient()
        _avoid.Init()
    return _avoid


def _read_mode():
    msc = _get_msc()
    code, result = msc.CheckMode()
    if code != 0:
        raise StairModeError("CheckMode() returned error code %d" % code)
    return result.get("name", "") if result else ""


def is_stair_mode_active(dry_run=False):
    """Read back the current motion mode. True if it is 'ai' right now."""
    if dry_run:
        return _dry and _active
    return _read_mode() == MODE_AI


def enter_stair_mode(disable_onboard_avoid=False, dry_run=False):
    """
    Switch to AI sport mode and verify it actually took effect before
    returning. Raises StairModeError (and sends nothing) if verification
    fails - callers must not send any Move() command in that case.

    disable_onboard_avoid: an unverified community report says the
    onboard LiDAR obstacle-avoidance system can read a stair riser as a
    wall and refuse to approach it (see STAIR_MODE_README.md). If set,
    the onboard avoid switch is turned off here and always restored in
    exit_stair_mode(), regardless of how the caller exits.
    """
    global _active, _dry, _prev_mode, _prev_avoid_state

    if _active:
        raise StairModeError(
            "enter_stair_mode() called while a session is already open - "
            "call exit_stair_mode() first.")

    if dry_run:
        print("[stair_mode] DRY RUN - pretending SelectMode('ai') "
              "succeeded and CheckMode confirmed it. Nothing is sent to "
              "the robot.")
        _prev_mode = ""
        _prev_avoid_state = None
        _dry = True
        _active = True
        return

    previous = _read_mode()   # raises StairModeError if this fails - we
                               # refuse to switch blind into a mode we can
                               # never confirm we came from

    prev_avoid_state = None
    if disable_onboard_avoid:
        avoid = _get_avoid()
        code, state = avoid.SwitchGet()
        if code == 0 and state:
            avoid.SwitchSet(False)
            prev_avoid_state = True
            print("[stair_mode] onboard obstacle-avoidance switch turned "
                  "off (will be restored on exit).")

    msc = _get_msc()
    code, _ = msc.SelectMode(MODE_AI)
    if code != 0:
        if prev_avoid_state:
            avoid.SwitchSet(True)
        raise StairModeError("SelectMode('ai') returned error code %d" % code)

    confirmed = False
    for _ in range(CHECK_RETRIES):
        time.sleep(CHECK_RETRY_DELAY)
        try:
            if _read_mode() == MODE_AI:
                confirmed = True
                break
        except StairModeError:
            continue

    if not confirmed:
        if prev_avoid_state:
            avoid.SwitchSet(True)
        raise StairModeError(
            "SelectMode('ai') returned success but CheckMode never read "
            "back 'ai' mode after %.1fs - refusing to send any movement "
            "command." % (CHECK_RETRIES * CHECK_RETRY_DELAY))

    _prev_mode = previous
    _prev_avoid_state = prev_avoid_state
    _dry = False
    _active = True
    print("[stair_mode] AI mode confirmed active (was %r before)."
          % (previous or "<none>"))


def exit_stair_mode():
    """
    Restore whatever mode was active before enter_stair_mode(). Safe to
    call even if enter_stair_mode() was never called or already failed -
    it is a no-op in that case. Always call this from a finally block, or
    use the StairMode context manager, which does that for you.
    """
    global _active, _dry, _prev_mode, _prev_avoid_state

    if not _active:
        return

    if _dry:
        print("[stair_mode] DRY RUN - pretending to restore previous mode.")
        _active = False
        _dry = False
        _prev_mode = None
        _prev_avoid_state = None
        return

    msc = _get_msc()
    try:
        msc.ReleaseMode()
    except Exception as exc:
        print("[stair_mode] ReleaseMode() raised: %s" % exc)

    if _prev_mode and _prev_mode != MODE_AI:
        try:
            msc.SelectMode(_prev_mode)
        except Exception as exc:
            print("[stair_mode] could not restore previous mode %r: %s"
                  % (_prev_mode, exc))

    if _prev_avoid_state:
        try:
            _get_avoid().SwitchSet(True)
        except Exception as exc:
            print("[stair_mode] could not re-enable onboard obstacle "
                  "avoidance: %s" % exc)

    print("[stair_mode] mode restored.")
    _active = False
    _prev_mode = None
    _prev_avoid_state = None


class StairMode:
    """
    Context manager wrapper so callers cannot forget to restore the mode.

        with StairMode() as active:
            if active:
                sport.Move(0.2, 0.0, 0.0)

    __enter__ raises StairModeError if verification fails, so the `with`
    body never runs and no movement command can be sent. exit_stair_mode()
    always runs on the way out - normal exit, exception, or Ctrl+C.
    """

    def __init__(self, disable_onboard_avoid=False, dry_run=False):
        self.disable_onboard_avoid = disable_onboard_avoid
        self.dry_run = dry_run

    def __enter__(self):
        enter_stair_mode(disable_onboard_avoid=self.disable_onboard_avoid,
                          dry_run=self.dry_run)
        return True

    def __exit__(self, exc_type, exc, tb):
        exit_stair_mode()
        return False
