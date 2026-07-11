"""Browser-based keyboard teleop for HOMIE/HOMIE+ ONNX policies (sim2sim).

Same deployment stack as teleop_sim_g1 (reuses its TeleopState / build_model /
reset_robot and the runtime.py observation pipeline), but renders offscreen
via EGL and serves an MJPEG stream plus a keyboard-capture page over HTTP —
for SSH-only servers where a native GLFW window is not available.

On the server:
  MUJOCO_GL=egl uv run --extra deploy python -m \
    mjlab_homierl.scripts.teleop_web_g1 --onnx homie-v7.onnx --port 8735

On your laptop:
  ssh -L 8735:localhost:8735 <server>    # then open http://localhost:8735

Keys are identical to teleop_sim_g1 (W/S/A/D/Q/E twist, Up/Down height,
R/F torso pitch, Space zero twist, 0 reset commands, Backspace reset robot),
plus camera-only extras handled in the browser page: Left/Right arrows orbit,
[ / ] zoom. Key taps are discrete (browser auto-repeat is dropped) to match
the native viewer's tap-to-adjust semantics.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

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

RENDER_EVERY = 2  # render at 25 fps against the 50 Hz control loop
JPEG_QUALITY = 80

# Browser KeyboardEvent.key -> the GLFW-style keycodes TeleopState.key expects.
_KEY_TO_CODE = {
  "ArrowUp": 265,
  "ArrowDown": 264,
  "Backspace": 259,
  " ": 32,
}


class FrameBuffer:
  """Latest-JPEG mailbox shared between the sim loop and stream handlers."""

  def __init__(self) -> None:
    self._cond = threading.Condition()
    self._jpeg: bytes | None = None
    self._seq = 0

  def put(self, jpeg: bytes) -> None:
    with self._cond:
      self._jpeg = jpeg
      self._seq += 1
      self._cond.notify_all()

  def get(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
    with self._cond:
      self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
      return self._jpeg, self._seq


class Camera:
  """Free camera orbiting the robot base (browser-adjustable)."""

  def __init__(self) -> None:
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.distance = 3.0
    self.cam.azimuth = 135.0
    self.cam.elevation = -15.0

  def key(self, k: str) -> bool:
    if k == "ArrowLeft":
      self.cam.azimuth = (self.cam.azimuth - 15.0) % 360.0
    elif k == "ArrowRight":
      self.cam.azimuth = (self.cam.azimuth + 15.0) % 360.0
    elif k == "[":
      self.cam.distance = float(np.clip(self.cam.distance + 0.4, 1.2, 8.0))
    elif k == "]":
      self.cam.distance = float(np.clip(self.cam.distance - 0.4, 1.2, 8.0))
    else:
      return False
    return True

  def track(self, base_pos: np.ndarray) -> None:
    self.cam.lookat[:] = [base_pos[0], base_pos[1], 0.7]


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>HOMIE web teleop</title><style>
  body { background:#14161a; color:#cdd3dc; font:14px/1.5 system-ui, sans-serif;
         display:flex; flex-direction:column; align-items:center; margin:0 }
  img { margin-top:12px; border:1px solid #2a2e36; max-width:96vw }
  #status { font:13px ui-monospace, monospace; margin:10px; color:#8fd48f }
  table { border-collapse:collapse; font-size:12px; color:#9aa3af }
  td { padding:1px 10px }
</style></head><body>
<img id="view" src="/stream">
<div id="status">connecting...</div>
<table><tr>
  <td>W/S vx</td><td>A/D vy</td><td>Q/E wz</td><td>&uarr;/&darr; height</td>
  <td>R/F pitch</td><td>Space zero</td><td>0 reset cmd</td>
  <td>Backspace reset robot</td><td>&larr;/&rarr; orbit</td><td>[ ] zoom</td>
</tr></table>
<script>
  document.addEventListener("keydown", (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    const teleop = "wsadqerfWSADQERF0 ".includes(e.key) ||
      ["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","Backspace","[","]"].includes(e.key);
    if (!teleop) return;
    e.preventDefault();
    fetch("/key?k=" + encodeURIComponent(e.key));
  });
  setInterval(async () => {
    try {
      const r = await (await fetch("/status")).json();
      document.getElementById("status").textContent =
        r.status + "   |   base z " + r.base_z.toFixed(2) + " m, tilt " +
        r.tilt_deg.toFixed(0) + "\\u00b0" + (r.fallen ? "   [FELL - Backspace to reset]" : "");
    } catch (err) { /* server gone; leave last status */ }
  }, 250);
</script></body></html>"""


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", required=True, help="HOMIE/HOMIE+ ONNX policy")
  parser.add_argument("--port", type=int, default=8735)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=540)
  parser.add_argument(
    "--smoke", action="store_true", help="headless render/encode self-test, no server"
  )
  args = parser.parse_args()

  from PIL import Image

  control_dt = PHYSICS_DT * DECIMATION
  policy = rt.HomieOnnxPolicy(args.onnx, control_dt=control_dt)
  model = build_model(policy)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)

  qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in policy.joint_names])
  dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in policy.joint_names])
  act_jname = [
    model.joint(model.actuator_trnid[aid, 0]).name for aid in range(model.nu)
  ]
  act_targets_idx = {n: i for i, n in enumerate(policy.joint_names)}

  state = TeleopState(policy)
  camera = Camera()
  frames = FrameBuffer()
  reset_robot(model, data, policy)
  policy.reset()

  def control_step() -> None:
    state.slew(control_dt)
    quat = data.qpos[3:7].astype(np.float32)
    gyro = data.qvel[3:6].astype(np.float32)
    q = data.qpos[qadr].astype(np.float32)
    dq = data.qvel[dadr].astype(np.float32)
    one_step = policy.one_step_obs(state.command(), gyro, quat, q, dq)
    targets = policy.act(one_step)
    for aid in range(model.nu):
      k = act_targets_idx.get(act_jname[aid])
      if k is not None:
        data.ctrl[aid] = targets[k]
    for _ in range(DECIMATION):
      mujoco.mj_step(model, data)

  def render_frame() -> bytes:
    camera.track(data.qpos[0:3])
    renderer.update_scene(data, camera=camera.cam)
    img = Image.fromarray(renderer.render())
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()

  def tilt_deg() -> float:
    g = rt.gravity_orientation(data.qpos[3:7].astype(np.float32))
    return float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))

  if args.smoke:
    t0 = time.perf_counter()
    for i in range(75):
      control_step()
      if i % RENDER_EVERY == 0:
        jpeg = render_frame()
    dt = time.perf_counter() - t0
    print(
      f"smoke: 75 control steps + {75 // RENDER_EVERY} frames in {dt:.2f}s "
      f"(budget {75 * control_dt:.2f}s), last jpeg {len(jpeg)} bytes "
      f"-> {'OK' if dt < 75 * control_dt else 'TOO SLOW'}"
    )
    return

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:  # silence per-request spam
      pass

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
      url = urlparse(self.path)
      if url.path == "/":
        body = _PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
      elif url.path == "/key":
        k = parse_qs(url.query).get("k", [""])[0]
        if not camera.key(k):
          code = _KEY_TO_CODE.get(k, ord(k.upper()) if len(k) == 1 else None)
          if code is not None:
            state.key(code)
        self.send_response(204)
        self.end_headers()
      elif url.path == "/status":
        body = json.dumps(
          {
            "status": state.status(),
            "base_z": float(data.qpos[2]),
            "tilt_deg": tilt_deg(),
            "fallen": tilt_deg() > 60.0,
          }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
      elif url.path == "/stream":
        self.send_response(200)
        self.send_header(
          "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        seq = 0
        try:
          while True:
            jpeg, seq = frames.get(seq)
            if jpeg is None:
              continue
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
          return
      else:
        self.send_response(404)
        self.end_headers()

  server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
  threading.Thread(target=server.serve_forever, daemon=True).start()

  print(
    f"Loaded {args.onnx}\n  {policy.num_commands}-dim command"
    f"{' (torso pitch: R/F)' if policy.has_pitch else ''}, "
    f"height {policy.height_range}, one-step obs {policy.num_one_step_obs}"
  )
  print(
    f"Serving on http://127.0.0.1:{args.port} — from your machine:\n"
    f"  ssh -L {args.port}:localhost:{args.port} <this-server>  "
    f"then open http://localhost:{args.port}"
  )

  step = 0
  while True:
    t0 = time.perf_counter()
    if state.reset_requested:
      state.reset_requested = False
      state.reset_commands()
      reset_robot(model, data, policy)
      policy.reset()
    control_step()
    if step % RENDER_EVERY == 0:
      frames.put(render_frame())
    step += 1
    leftover = control_dt - (time.perf_counter() - t0)
    if leftover > 0:
      time.sleep(leftover)


if __name__ == "__main__":
  main()
