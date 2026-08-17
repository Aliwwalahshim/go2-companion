"use strict";

// ---- state ----
var state = null;
var mapImg = new Image();
var mapReady = false;
var mapStamp = null;
var activeRoom = null;

var canvas = document.getElementById("map");
var ctx = canvas.getContext("2d");

// ---- helpers ----
function el(id) { return document.getElementById(id); }

function fitCanvas() {
  var wrap = canvas.parentElement;
  var dpr = window.devicePixelRatio || 1;
  var w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: w, h: h };
}

// world (x,y) -> image pixel (col,row), matching map_bridge.py
function worldToImg(wx, wy, m) {
  var col = (wx - m.origin_x) / m.resolution;
  var row = (m.height - 1) - (wy - m.origin_y) / m.resolution;
  return { col: col, row: row };
}

// compute the fit rectangle of the map image inside the canvas
function mapDrawRect(view, m) {
  var scale = Math.min(view.w / m.width, view.h / m.height);
  var dw = m.width * scale, dh = m.height * scale;
  var dx = (view.w - dw) / 2, dy = (view.h - dh) / 2;
  return { dx: dx, dy: dy, scale: scale };
}

function draw() {
  var view = fitCanvas();
  ctx.clearRect(0, 0, view.w, view.h);

  var m = state && state.map && state.map.available ? state.map : null;
  if (!m || !mapReady) { return; }

  var r = mapDrawRect(view, m);
  // map image
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(mapImg, r.dx, r.dy, m.width * r.scale, m.height * r.scale);

  // room markers
  var rooms = (state && state.rooms) || {};
  Object.keys(rooms).forEach(function (name) {
    var rm = rooms[name];
    var p = worldToImg(rm.x, rm.y, m);
    var cx = r.dx + p.col * r.scale, cy = r.dy + p.row * r.scale;
    var on = name === activeRoom;
    ctx.beginPath();
    ctx.arc(cx, cy, on ? 8 : 6, 0, Math.PI * 2);
    ctx.fillStyle = on ? "#ffd257" : "#f0b429";
    ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.stroke();
    ctx.font = "600 12px -apple-system, Segoe UI, sans-serif";
    ctx.fillStyle = "#e6edf3";
    ctx.fillText(name, cx + 10, cy + 4);
  });

  // robot pose
  var pose = state && state.pose;
  if (pose) {
    var pp = worldToImg(pose.x, pose.y, m);
    var rx = r.dx + pp.col * r.scale, ry = r.dy + pp.row * r.scale;
    // heading: world dir (cos yaw, sin yaw) -> canvas (cos yaw, -sin yaw)
    var hx = Math.cos(pose.yaw), hy = -Math.sin(pose.yaw);
    var L = 16;
    // glow
    ctx.beginPath();
    ctx.arc(rx, ry, 11, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(53,208,127,0.18)"; ctx.fill();
    // heading triangle
    var ax = rx + hx * L, ay = ry + hy * L;
    var px = -hy, py = hx; // perpendicular
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(rx + px * 7, ry + py * 7);
    ctx.lineTo(rx - px * 7, ry - py * 7);
    ctx.closePath();
    ctx.fillStyle = "#35d07f"; ctx.fill();
    // body dot
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#eafff3"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#35d07f"; ctx.stroke();
  }
}

// ---- UI rendering ----
function renderRooms() {
  var box = el("rooms");
  var rooms = (state && state.rooms) || {};
  var names = Object.keys(rooms).sort();
  if (names.length === 0) {
    box.innerHTML = '<div class="muted">No rooms recorded yet.</div>';
    return;
  }
  box.innerHTML = "";
  names.forEach(function (name) {
    var rm = rooms[name];
    var b = document.createElement("button");
    b.className = "room-btn" + (name === activeRoom ? " active" : "");
    b.innerHTML = '<span>' + name + '</span><span class="coord">x ' +
      rm.x.toFixed(1) + '  y ' + rm.y.toFixed(1) + '</span>';
    b.onclick = function () { goToRoom(name); };
    box.appendChild(b);
  });
}

function setV(id, text, cls) {
  var e = el(id); e.textContent = text;
  e.className = "v" + (cls ? " " + cls : "");
}

function renderStatus() {
  var st = state.status || {};
  setV("stPose", st.pose_ok ? "tracking" : "no signal",
    st.pose_ok ? "good" : "bad");
  setV("stMap", st.map_ok ? "live" : "waiting",
    st.map_ok ? "good" : "warn");
  var nav = state.nav || {};
  setV("stNav", nav.running ? ("-> " + nav.room) : "idle",
    nav.running ? "warn" : "");
  el("lastResult").textContent = state.last_result || "";
  activeRoom = nav.running ? nav.room : null;

  var connOk = state.ok;
  el("statusDot").className = "status-dot " + (connOk ? "ok" : "bad");
  el("statusText").textContent = connOk ? "connected" : "disconnected";

  el("mapEmpty").style.display = st.map_ok ? "none" : "flex";

  var pose = state.pose;
  el("poseReadout").textContent = pose
    ? ("pose  x " + pose.x.toFixed(2) + "  y " + pose.y.toFixed(2) +
       "  yaw " + (pose.yaw * 180 / Math.PI).toFixed(0) + " deg")
    : "pose --";
}

// ---- networking ----
function poll() {
  fetch("/api/state", { cache: "no-store" })
    .then(function (r) { return r.json(); })
    .then(function (s) {
      state = s;
      // (re)load the map image only when it changes
      if (s.map && s.map.available && s.map.stamp !== mapStamp) {
        mapStamp = s.map.stamp;
        var img = new Image();
        img.onload = function () { mapImg = img; mapReady = true; };
        img.src = "/map.png?t=" + encodeURIComponent(mapStamp);
      }
      renderStatus();
      renderRooms();
      draw();
    })
    .catch(function () {
      if (state) { state.ok = false; }
      el("statusDot").className = "status-dot bad";
      el("statusText").textContent = "disconnected";
    });
}

function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  }).then(function (r) { return r.json(); });
}

function goToRoom(name) {
  activeRoom = name; renderRooms();
  post("/api/goto", { room: name }).then(function (res) {
    el("lastResult").textContent = res.message || res.error || "";
  });
}

function stopAll() {
  post("/api/stop", {}).then(function (res) {
    el("lastResult").textContent = res.message || "stopped";
  });
}

function recordRoom() {
  var name = el("roomName").value.trim();
  if (!name) { el("roomName").focus(); return; }
  el("lastResult").textContent = "recording '" + name + "' ...";
  post("/api/record", { name: name }).then(function (res) {
    el("lastResult").textContent = res.ok
      ? ("recorded '" + name + "'") : (res.error || res.message || "record failed");
    el("roomName").value = "";
    poll();
  });
}

// ---- wire up ----
el("stopBtn").onclick = stopAll;
el("recordBtn").onclick = recordRoom;
el("roomName").addEventListener("keydown", function (e) {
  if (e.key === "Enter") { recordRoom(); }
});
window.addEventListener("resize", draw);

poll();
setInterval(poll, 200);
