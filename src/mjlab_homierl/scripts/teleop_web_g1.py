"""Browser-based remote teleop for HOMIE/HOMIE+ ONNX policies.

Runs the classic-MuJoCo sim2sim harness (same runtime.py pipeline as the
real robot / BiGym plugin) headless on the workstation and serves:

  /        control page: live video + keyboard capture + policy buttons
  /stream  MJPEG video (EGL offscreen render, ~25 fps)
  /key     POST keyboard events
  /status  JSON HUD state

Multiple policies are registered on the command line and hot-swappable from
the page (buttons or number keys) — same robot model, PD table re-applied
from each ONNX's metadata on switch.

Usage (on the workstation, e.g. inside tmux):
  MUJOCO_GL=egl uv run python -m mjlab_homierl.scripts.teleop_web_g1 \
    --policy v7=path/to/v7.onnx --policy v3=path/to/v3.onnx [--port 8642]

Remote viewing (from your laptop):
  ssh -L 8642:localhost:8642 <user>@<workstation>
  open http://localhost:8642

Keys: W/S vx, A/D vy, Q/E yaw, Up/Down height, R/F torso pitch (HOMIE+),
Space stop, 0 reset commands, Backspace reset robot, 1..9 switch policy.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

from mjlab_homierl import runtime as rt
from mjlab_homierl.scripts.teleop_sim_g1 import (
  DECIMATION,
  PHYSICS_DT,
  TeleopState,
  build_model,
  reset_robot,
)

# Browser key -> teleop_sim_g1 keycode (GLFW-ish ints TeleopState expects).
KEYMAP = {
  "w": ord("W"), "s": ord("S"), "a": ord("A"), "d": ord("D"),
  "q": ord("Q"), "e": ord("E"), "r": ord("R"), "f": ord("F"),
  "ArrowUp": 265, "ArrowDown": 264, " ": 32, "0": ord("0"),
  "Backspace": 259,
}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>G1 teleop</title>
<style>
 body{margin:0;background:#14181b;color:#e3e8e4;font:14px/1.5 ui-sans-serif,system-ui;display:flex;flex-direction:column;align-items:center;gap:10px;padding:14px}
 img{border:1px solid #2c3438;border-radius:4px;max-width:96vw}
 #hud{font-family:ui-monospace,monospace;white-space:pre;color:#9fe8da}
 .row{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
 button{background:#1c2226;color:#e3e8e4;border:1px solid #2c3438;border-radius:4px;padding:6px 14px;cursor:pointer;font:inherit}
 button.active{border-color:#54ada1;color:#7cc7bc}
 #keys{color:#94a09a;max-width:70ch;text-align:center}
</style></head><body>
<div class="row" id="policies"></div>
<img id="view" src="/stream">
<div id="hud">…</div>
<div id="keys">W/S 前后 · A/D 横移 · Q/E 转向 · ↑/↓ 蹲起 · R/F 前倾/回正 ·
空格 急停 · 0 指令复位 · 退格 机器人复位 · 数字键切 policy(点页面任意处取得键盘焦点)</div>
<script>
const send=k=>fetch('/key',{method:'POST',body:JSON.stringify({key:k})});
document.addEventListener('keydown',e=>{
  if(e.repeat&&!['w','s','a','d','q','e','r','f','ArrowUp','ArrowDown'].includes(e.key))return;
  if(e.key==='Backspace'||e.key.startsWith('Arrow')||e.key===' ')e.preventDefault();
  send(e.key);
});
async function refresh(){
  try{
    const s=await (await fetch('/status')).json();
    document.getElementById('hud').textContent=s.hud;
    const box=document.getElementById('policies');
    if(box.children.length!==s.policies.length){
      box.innerHTML='';
      s.policies.forEach((n,i)=>{const b=document.createElement('button');
        b.textContent=(i+1)+' · '+n;b.onclick=()=>send(String(i+1));box.appendChild(b);});
    }
    [...box.children].forEach((b,i)=>b.classList.toggle('active',i===s.active));
  }catch(e){}
  setTimeout(refresh,500);
}
refresh();
</script></body></html>"""


class Shared:
  def __init__(self):
    self.lock = threading.Lock()
    self.jpeg: bytes = b""
    self.keys: list[str] = []
    self.status: dict = {}

  def push_key(self, k: str) -> None:
    with self.lock:
      self.keys.append(k)

  def pop_keys(self) -> list[str]:
    with self.lock:
      out, self.keys = self.keys, []
      return out


def make_handler(shared: Shared):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
      pass

    def do_GET(self):
      if self.path == "/":
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
      elif self.path == "/status":
        body = json.dumps(shared.status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
      elif self.path == "/stream":
        self.send_response(200)
        self.send_header(
          "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
          while True:
            with shared.lock:
              frame = shared.jpeg
            if frame:
              self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
              self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
              self.wfile.write(frame)
              self.wfile.write(b"\r\n")
            time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
          return
      else:
        self.send_error(404)

    def do_POST(self):
      if self.path == "/key":
        n = int(self.headers.get("Content-Length", 0))
        try:
          key = json.loads(self.rfile.read(n))["key"]
          shared.push_key(str(key))
        except Exception:
          pass
        self.send_response(204)
        self.end_headers()
      else:
        self.send_error(404)

  return Handler


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy",
    action="append",
    required=True,
    metavar="NAME=ONNX",
    help="repeatable; first one is active at start",
  )
  parser.add_argument("--port", type=int, default=8642)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--smoke", action="store_true", help="headless self-test")
  args = parser.parse_args()

  control_dt = PHYSICS_DT * DECIMATION
  registry: list[tuple[str, rt.HomieOnnxPolicy]] = []
  for spec_arg in args.policy:
    name, _, path = spec_arg.partition("=")
    registry.append((name, rt.HomieOnnxPolicy(path, control_dt=control_dt)))

  active = 0
  policy = registry[0][1]
  model = build_model(policy)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=480, width=640)
  state = TeleopState(policy)
  reset_robot(model, data, policy)
  policy.reset()

  qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in policy.joint_names])
  dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in policy.joint_names])

  shared = Shared()
  server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
  threading.Thread(target=server.serve_forever, daemon=True).start()
  print(f"serving on http://{args.host}:{args.port}  "
        f"policies: {[n for n, _ in registry]}")

  import PIL.Image

  def switch(idx: int) -> None:
    nonlocal active, policy, state
    if not (0 <= idx < len(registry)) or idx == active:
      return
    active = idx
    policy = registry[idx][1]
    # Same robot; re-apply this export's PD table (identical across our
    # versions, but keep the contract honest).
    name_to_i = {n: i for i, n in enumerate(policy.joint_names)}
    for aid in range(model.nu):
      jn = model.joint(model.actuator_trnid[aid, 0]).name
      k = name_to_i.get(jn)
      if k is not None:
        kp, kd = float(policy.kps[k]), float(policy.kds[k])
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kd
    policy.reset()
    state = TeleopState(policy)
    print(f"\nswitched to {registry[idx][0]}")

  frame_tick = 0
  t_end = time.time() + 6.0 if args.smoke else None
  if args.smoke:
    shared.push_key("ArrowDown"); shared.push_key("2" if len(registry) > 1 else "w")

  while t_end is None or time.time() < t_end:
    t0 = time.perf_counter()
    for k in shared.pop_keys():
      if k.isdigit() and k != "0":
        switch(int(k) - 1)
      elif k.lower() in KEYMAP or k in KEYMAP:
        state.key(KEYMAP.get(k, KEYMAP.get(k.lower())))
    if state.reset_requested:
      state.reset_requested = False
      state.reset_commands()
      reset_robot(model, data, policy)
      policy.reset()

    state.slew(control_dt)
    one = policy.one_step_obs(
      state.command(),
      data.qvel[3:6].astype(np.float32),
      data.qpos[3:7].astype(np.float32),
      data.qpos[qadr].astype(np.float32),
      data.qvel[dadr].astype(np.float32),
    )
    targets = policy.act(one)
    data.ctrl[:] = targets  # actuators created in policy.joint_names order
    for _ in range(DECIMATION):
      mujoco.mj_step(model, data)

    frame_tick += 1
    if frame_tick % 2 == 0:  # ~25 fps
      cam = mujoco.MjvCamera()
      cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
      cam.azimuth, cam.elevation, cam.distance = 135, -18, 2.2
      renderer.update_scene(data, camera=cam)
      buf = io.BytesIO()
      PIL.Image.fromarray(renderer.render()).save(buf, "JPEG", quality=75)
      with shared.lock:
        shared.jpeg = buf.getvalue()
    g = rt.gravity_orientation(data.qpos[3:7].astype(np.float32))
    tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
    shared.status = {
      "policies": [n for n, _ in registry],
      "active": active,
      "hud": (
        f"policy {registry[active][0]}   {state.status()}\n"
        f"base z {data.qpos[2]:.2f} m   tilt {tilt:4.1f}°   "
        f"{'⚠ 已倒,按退格复位' if tilt > 60 else ''}"
      ),
    }

    leftover = control_dt - (time.perf_counter() - t0)
    if leftover > 0:
      time.sleep(leftover)

  if args.smoke:
    import urllib.request

    page = urllib.request.urlopen(
      f"http://{args.host}:{args.port}/", timeout=5
    ).read()
    status = json.loads(
      urllib.request.urlopen(
        f"http://{args.host}:{args.port}/status", timeout=5
      ).read()
    )
    with shared.lock:
      frame_len = len(shared.jpeg)
    print(
      f"smoke: page {len(page)}B, status active={status['active']} "
      f"policies={status['policies']}, latest frame {frame_len}B, "
      f"height_target {state.height_target:.2f} (ArrowDown applied)"
    )


if __name__ == "__main__":
  main()
