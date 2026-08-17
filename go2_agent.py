#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
go2_agent.py  --  runs ON THE JETSON.

Holds the Unitree SportClient and listens on TCP for commands.
All safety limits live HERE, not in the AI model.

RUN:
    cd ~/Go2_project
    python3 go2_agent.py eth0

Do NOT source foxy for this terminal.  It uses Unitree DDS, not ROS 2.
Physical e-stop stays the remote:  L2 + B.
"""

import sys
import json
import socket
import time
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 45610

MAX_LIN = 0.6      # m/s forward   (never negative: no rear sensing)
MAX_LAT = 0.4      # m/s sideways
MAX_ROT = 0.8      # rad/s turn
MAX_SECONDS = 5.0  # max duration of one move command

# ----------------------------------------------------------------------
# ACTION TABLE
#   name : (SportClient method, takes_bool, dangerous, description)
# ----------------------------------------------------------------------
ACTIONS = {
    # --- posture ---
    "stand_up":       ("StandUp",       False, False, "Stand up"),
    "stand_down":     ("StandDown",     False, False, "Lie down"),
    "sit":            ("Sit",           False, False, "Sit"),
    "rise_sit":       ("RiseSit",       False, False, "Rise from sitting"),
    "balance_stand":  ("BalanceStand",  False, False, "Balanced standing"),
    "recovery_stand": ("RecoveryStand", False, False, "Recover after a fall"),
    "damp":           ("Damp",          False, False, "Relax the motors"),
    "stop":           ("StopMove",      False, False, "Stop all movement now"),

    # --- gaits (toggle: enable true/false) ---
    "classic_walk":   ("ClassicWalk",   True,  False, "Classic walking gait"),
    "static_walk":    ("StaticWalk",    True,  False, "Slow high-stability walk"),
    "trot_run":       ("TrotRun",       True,  False, "Faster trot gait"),
    "free_walk":      ("FreeWalk",      True,  False, "Free walk gait"),
    "free_bound":     ("FreeBound",     True,  False, "Bounding gait"),
    "free_avoid":     ("FreeAvoid",     True,  False, "Built-in obstacle avoidance"),
    "walk_upright":   ("WalkUpright",   True,  False, "Walk on hind legs"),
    "cross_step":     ("CrossStep",     True,  False, "Cross-step gait"),

    # --- tricks ---
    "stretch":        ("Stretch",       False, False, "Stretch"),
    "hello":          ("Hello",         False, False, "Wave hello"),
    "heart":          ("Heart",         False, False, "Make a heart gesture"),
    "scrape":         ("Scrape",        False, False, "Scrape / bow"),
    "dance1":         ("Dance1",        False, False, "Dance routine 1"),
    "dance2":         ("Dance2",        False, False, "Dance routine 2"),
    "content":        ("Content",       False, False, "Happy gesture"),
    "pose":           ("Pose",          True,  False, "Strike a pose"),

    # --- dangerous: need confirm=true ---
    "front_flip":     ("FrontFlip",     False, True,  "Front flip - CLEAR SPACE"),
    "back_flip":      ("BackFlip",      False, True,  "Back flip - CLEAR SPACE"),
    "left_flip":      ("LeftFlip",      False, True,  "Left flip - CLEAR SPACE"),
    "hand_stand":     ("HandStand",     True,  True,  "Handstand - CLEAR SPACE"),
    "free_jump":      ("FreeJump",      True,  True,  "Free jump - CLEAR SPACE"),
    "front_jump":     ("FrontJump",     False, True,  "Front jump - CLEAR SPACE"),
    "front_pounce":   ("FrontPounce",   False, True,  "Front pounce - CLEAR SPACE"),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Go2Agent(object):

    def __init__(self, iface):
        print("Init Unitree channel on %s ..." % iface)
        ChannelFactoryInitialize(0, iface)

        self.sport = SportClient()
        self.sport.SetTimeout(10.0)
        self.sport.Init()
        print("SportClient ready.")

        # runtime check: keep only what this firmware really has
        self.available = {}
        missing = []
        for name, spec in ACTIONS.items():
            if hasattr(self.sport, spec[0]):
                self.available[name] = spec
            else:
                missing.append(name)
        print("Available actions: %d" % len(self.available))
        if missing:
            print("Not on this firmware: %s" % ", ".join(sorted(missing)))

        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def do_list(self):
        out = []
        for name in sorted(self.available):
            m, takes_bool, danger, desc = self.available[name]
            out.append({
                "action": name,
                "description": desc,
                "toggle": takes_bool,
                "needs_confirm": danger,
            })
        return {"ok": True, "actions": out}

    # ------------------------------------------------------------------
    def do_action(self, name, enable=True, confirm=False):
        if name not in self.available:
            return {"ok": False,
                    "msg": "Unknown action '%s'. Use list_robot_actions." % name}

        method_name, takes_bool, danger, desc = self.available[name]

        if danger and not confirm:
            return {"ok": False,
                    "msg": "'%s' is dangerous. Ask the human out loud, then call "
                           "again with confirm=true." % name}

        fn = getattr(self.sport, method_name)
        try:
            with self.lock:
                if takes_bool:
                    code = fn(bool(enable))
                else:
                    code = fn()
            return {"ok": True,
                    "msg": "%s -> %s (code %s)" % (name, desc, code)}
        except Exception as e:
            return {"ok": False, "msg": "error running %s: %s" % (name, e)}

    # ------------------------------------------------------------------
    def do_move(self, x=0.0, y=0.0, z=0.0, seconds=1.0):
        # HARD LIMITS - applied no matter what was requested
        x = clamp(float(x), 0.0, MAX_LIN)        # never reverse
        y = clamp(float(y), -MAX_LAT, MAX_LAT)
        z = clamp(float(z), -MAX_ROT, MAX_ROT)
        seconds = clamp(float(seconds), 0.0, MAX_SECONDS)

        try:
            end = time.time() + seconds
            with self.lock:
                while time.time() < end:
                    self.sport.Move(x, y, z)
                    time.sleep(0.05)
                self.sport.StopMove()
            return {"ok": True,
                    "msg": "moved x=%.2f y=%.2f z=%.2f for %.1fs" % (x, y, z, seconds)}
        except Exception as e:
            try:
                self.sport.StopMove()
            except Exception:
                pass
            return {"ok": False, "msg": "error during move: %s" % e}

    # ------------------------------------------------------------------
    def handle(self, req):
        cmd = req.get("cmd", "")
        args = req.get("args", {}) or {}

        if cmd == "ping":
            return {"ok": True, "msg": "agent alive"}
        if cmd == "list":
            return self.do_list()
        if cmd == "action":
            return self.do_action(args.get("action", ""),
                                  args.get("enable", True),
                                  args.get("confirm", False))
        if cmd == "move":
            return self.do_move(args.get("x", 0.0), args.get("y", 0.0),
                                args.get("z", 0.0), args.get("seconds", 1.0))
        if cmd == "stop":
            try:
                with self.lock:
                    self.sport.StopMove()
                return {"ok": True, "msg": "stopped"}
            except Exception as e:
                return {"ok": False, "msg": "stop failed: %s" % e}

        return {"ok": False, "msg": "unknown cmd '%s'" % cmd}


# ----------------------------------------------------------------------
def serve(agent):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    print("")
    print("=" * 56)
    print(" GO2 AGENT listening on %s:%d" % (HOST, PORT))
    print(" Waiting for the MCP server to connect.")
    print(" Keep the remote in your hand.  E-stop = L2 + B.")
    print("=" * 56)
    print("")

    while True:
        try:
            conn, addr = srv.accept()
        except KeyboardInterrupt:
            break
        try:
            f = conn.makefile("rwb")
            line = f.readline()
            if not line:
                conn.close()
                continue
            try:
                req = json.loads(line.decode("utf-8").strip())
            except Exception as e:
                reply = {"ok": False, "msg": "bad json: %s" % e}
            else:
                print("[%s] %s" % (addr[0], json.dumps(req)))
                reply = agent.handle(req)
                print("    -> %s" % reply.get("msg", "ok"))
            f.write((json.dumps(reply) + "\n").encode("utf-8"))
            f.flush()
        except Exception as e:
            print("connection error: %s" % e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    print("\nShutting down. Stopping robot.")
    try:
        agent.sport.StopMove()
    except Exception:
        pass


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    agent = Go2Agent(iface)
    serve(agent)


if __name__ == "__main__":
    main()
