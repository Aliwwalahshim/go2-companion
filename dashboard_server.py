"""
dashboard_server.py - the Go2 navigation dashboard web server.

Serves one page + a JSON API. Reads the robot pose from the existing
pose_server.py over UDP (via pose_client), reads the map PNG/JSON written by
map_bridge.py, reads rooms.json, and drives navigation by launching the
existing go_to_room.py / record_room.py as subprocesses.

CRITICAL: this process must NEVER import rclpy (it would collide with the
Unitree SDK's CycloneDDS). It only uses standard library + tornado + the
pure-UDP pose_client. The subprocesses it launches (go_to_room.py etc.) ARE
the SDK/control side, so run THIS server in a NON-foxy terminal so they
inherit the right environment.

    python3 dashboard_server.py --interface eth0 --port 8080

Then open http://<jetson-wifi-ip>:8080 from a laptop/tablet.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import tornado.ioloop
import tornado.web

HERE = os.path.dirname(os.path.abspath(__file__))
# this file now lives IN front_camera alongside pose_client / go_to_room /
# record_room / rooms.json, so the project dir is just its own directory.
FRONT = HERE
ROOMS_JSON = os.path.join(FRONT, "rooms.json")
MAP_DIR = "/tmp/go2_dashboard"
MAP_PNG = os.path.join(MAP_DIR, "map.png")
MAP_JSON = os.path.join(MAP_DIR, "map.json")

sys.path.insert(0, FRONT)
from pose_client import PoseClient      # pure UDP, no rclpy


class Nav:
    """Owns the single navigation subprocess (go_to_room.py)."""

    def __init__(self, interface):
        self.interface = interface
        self.proc = None
        self.room = None
        self.last_result = "idle"
        self._lock = threading.Lock()

    def running(self):
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def _reap(self):
        # update last_result if the process just finished
        if self.proc is not None and self.proc.poll() is not None:
            rc = self.proc.returncode
            if rc == 0:
                self.last_result = "arrived at %s" % self.room
            elif rc in (-15, 143):
                self.last_result = "stopped (%s)" % self.room
            else:
                self.last_result = "nav to %s ended (code %s)" % (self.room, rc)
            self.proc = None

    def go(self, room):
        with self._lock:
            self._reap()
            if self.proc is not None and self.proc.poll() is None:
                return False, "busy: already driving to %s" % self.room
            self.room = room
            self.last_result = "driving to %s ..." % room
            try:
                self.proc = subprocess.Popen(
                    [sys.executable, "-u", os.path.join(FRONT, "go_to_room.py"),
                     room, self.interface],
                    cwd=FRONT,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                self.last_result = "failed to launch: %s" % exc
                return False, self.last_result
            return True, "driving to %s" % room

    def stop(self):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                # go_to_room.py issues StopMove in its SIGTERM handler
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=4.0)
                except Exception:
                    self.proc.kill()
                self.last_result = "STOPPED (%s)" % self.room
                self.proc = None
                return True, "stopped"
            self.last_result = "stop: nothing was running"
            return True, "nothing to stop"


def load_rooms():
    try:
        with open(ROOMS_JSON, "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_map_meta():
    try:
        with open(MAP_JSON, "r") as fh:
            meta = json.load(fh)
        meta["available"] = os.path.isfile(MAP_PNG)
        return meta
    except (OSError, ValueError):
        return {"available": False}


# --- request handlers ---------------------------------------------------
class Base(tornado.web.RequestHandler):
    def initialize(self, ctx):
        self.ctx = ctx

    def set_default_headers(self):
        self.set_header("Cache-Control", "no-store")


class IndexHandler(Base):
    def get(self):
        self.render(os.path.join(HERE, "static", "index.html"))


class MapImageHandler(Base):
    def get(self):
        if not os.path.isfile(MAP_PNG):
            self.set_status(404)
            self.finish(b"")
            return
        self.set_header("Content-Type", "image/png")
        with open(MAP_PNG, "rb") as fh:
            self.finish(fh.read())


class StateHandler(Base):
    def get(self):
        pc = self.ctx["pose"]
        nav = self.ctx["nav"]
        nav._reap()
        p = pc.get()
        pose = None
        if p is not None:
            pose = {"x": p.x, "y": p.y, "yaw": p.yaw, "age": p.age}
        meta = read_map_meta()
        state = {
            "ok": True,
            "pose": pose,
            "rooms": load_rooms(),
            "map": meta,
            "nav": {"running": nav.running(), "room": nav.room},
            "status": {"pose_ok": pose is not None,
                       "map_ok": bool(meta.get("available"))},
            "last_result": nav.last_result,
        }
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(state))


class GotoHandler(Base):
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except ValueError:
            body = {}
        room = str(body.get("room", "")).strip()
        rooms = load_rooms()
        if room not in rooms:
            self.set_status(400)
            self.finish(json.dumps({"ok": False, "error": "unknown room"}))
            return
        ok, msg = self.ctx["nav"].go(room)
        self.finish(json.dumps({"ok": ok, "message": msg}))


class StopHandler(Base):
    def post(self):
        ok, msg = self.ctx["nav"].stop()
        self.finish(json.dumps({"ok": ok, "message": msg}))


class RecordHandler(Base):
    async def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except ValueError:
            body = {}
        name = str(body.get("name", "")).strip()
        if not name or "/" in name or " " in name:
            self.set_status(400)
            self.finish(json.dumps({"ok": False,
                                    "error": "bad room name"}))
            return

        def run_record():
            try:
                r = subprocess.run(
                    [sys.executable, os.path.join(FRONT, "record_room.py"),
                     name],
                    cwd=FRONT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=8.0)
                return r.returncode, (r.stdout or b"").decode("utf-8", "replace")
            except Exception as exc:
                return 1, "record failed: %s" % exc

        loop = tornado.ioloop.IOLoop.current()
        rc, out = await loop.run_in_executor(None, run_record)
        ok = rc == 0
        self.ctx["nav"].last_result = ("recorded '%s'" % name if ok
                                       else "record '%s' failed" % name)
        self.finish(json.dumps({"ok": ok, "message": out.strip()[:300]}))


def make_app(ctx):
    return tornado.web.Application([
        (r"/", IndexHandler, {"ctx": ctx}),
        (r"/map.png", MapImageHandler, {"ctx": ctx}),
        (r"/api/state", StateHandler, {"ctx": ctx}),
        (r"/api/goto", GotoHandler, {"ctx": ctx}),
        (r"/api/stop", StopHandler, {"ctx": ctx}),
        (r"/api/record", RecordHandler, {"ctx": ctx}),
    ],
        static_path=os.path.join(HERE, "static"),
        template_path=os.path.join(HERE, "static"))


def main():
    ap = argparse.ArgumentParser(description="Go2 navigation dashboard")
    ap.add_argument("--interface", default="eth0",
                    help="robot control interface passed to go_to_room.py")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ctx = {"pose": PoseClient(), "nav": Nav(args.interface)}
    app = make_app(ctx)
    app.listen(args.port)
    print("[dashboard] serving on http://0.0.0.0:%d  (interface=%s)"
          % (args.port, args.interface))
    print("[dashboard] open http://<jetson-wifi-ip>:%d from your laptop"
          % args.port)
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        ctx["nav"].stop()
        print("\n[dashboard] stopped.")


if __name__ == "__main__":
    main()
