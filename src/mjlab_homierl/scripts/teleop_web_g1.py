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


class CameraState:
  """Preset + free-adjust tracking camera. `chase` locks behind the robot's
  heading (driving view); the others are world-fixed angles."""

  PRESETS = ("chase", "iso", "side", "front", "top")
  _ANGLES = {  # name -> (azimuth or None for heading-locked, elevation)
    "chase": (None, -15),
    "iso": (135, -18),
    "side": (90, -10),
    "front": (180, -10),
    "top": (135, -70),
  }

  def __init__(self):
    self.mode = "chase"
    self.dist = 2.2
    self.az_offset = 0.0  # free rotation on top of the preset

  def handle(self, key: str) -> bool:
    if key.startswith("cam:") and key[4:] in self.PRESETS:
      self.mode = key[4:]
      self.az_offset = 0.0
      return True
    if key == "c":
      i = self.PRESETS.index(self.mode)
      self.mode = self.PRESETS[(i + 1) % len(self.PRESETS)]
      self.az_offset = 0.0
      return True
    if key == "[":
      self.az_offset -= 15.0
      return True
    if key == "]":
      self.az_offset += 15.0
      return True
    if key in ("-", "_"):
      self.dist = min(5.0, self.dist * 1.2)
      return True
    if key in ("=", "+"):
      self.dist = max(0.8, self.dist / 1.2)
      return True
    return False

  def apply(self, cam: mujoco.MjvCamera, data: mujoco.MjData) -> None:
    az, elev = self._ANGLES[self.mode]
    if az is None:  # chase: behind the robot's heading
      w, x, y, z = data.qpos[3:7]
      yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
      az = yaw + 180.0
    lookat_z = 0.5 if self.mode != "top" else 0.1
    cam.lookat[:] = [data.qpos[0], data.qpos[1], lookat_z]
    cam.azimuth = az + self.az_offset
    cam.elevation = elev
    cam.distance = self.dist

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
<div class="row" id="cams">
  <button onclick="send('cam:chase')">跟随</button>
  <button onclick="send('cam:iso')">斜角</button>
  <button onclick="send('cam:side')">侧面</button>
  <button onclick="send('cam:front')">正面</button>
  <button onclick="send('cam:top')">俯视</button>
</div>
<div id="hud">…</div>
<div id="keys">W/S 前后 · A/D 横移 · Q/E 转向 · ↑/↓ 蹲起 · R/F 前倾/回正 ·
空格 急停 · 0 指令复位 · 退格 机器人复位 · 数字键切 policy ·
C 切镜头 · [ ] 旋转 · +/- 缩放(点页面任意处取得键盘焦点)</div>
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


class SonicBackend:
  """GEAR-SONIC (GR00T-WBC) motion-tracking playback channel.

  SONIC is a motion-tracking foundation policy — its input is a reference
  motion stream, not a joystick, so WASD does not apply; Backspace restarts
  the clip and playback loops. One backend instance shares the ONNX
  sessions + MuJoCo model across all registered motions.
  """

  def __init__(self, repo_root: str):
    import sys
    from pathlib import Path

    root = Path(repo_root)
    src = root / "mujoco_inference" / "src"
    if str(src) not in sys.path:
      sys.path.insert(0, str(src))
    from sonic_mujoco import runner as R  # noqa: N814

    self.R = R
    enc, dec = R.ensure_policy_files(
      root / "gear_sonic_deploy" / "policy" / "release", download=False
    )
    self.encoder = R.SonicMujocoRunner._session(enc)
    self.decoder = R.SonicMujocoRunner._session(dec)
    self.enc_in = self.encoder.get_inputs()[0].name
    self.dec_in = self.decoder.get_inputs()[0].name
    self.motion_dir = root / "gear_sonic_deploy" / "reference" / "example"
    self.model = mujoco.MjModel.from_xml_path(
      str(root / "gear_sonic_deploy" / "g1" / "scene_29dof.xml")
    )
    self.model.opt.timestep = R.SonicMujocoRunner.sim_dt
    self.data = mujoco.MjData(self.model)
    self.motion_name = ""
    self.motion = None

  def set_motion(self, name: str) -> None:
    self.motion_name = name
    self.motion = self.R.Motion.load(self.motion_dir / name)
    self.reset()

  def reset(self) -> None:
    import collections

    R, model, data = self.R, self.model, self.data
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, 0.80]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:36] = R.DEFAULT_ANGLES
    mujoco.mj_forward(model, data)
    self.init_quat = data.qpos[3:7].copy()
    self.anchor = data.qpos[:3].copy()
    self.last_action = np.zeros(29, dtype=np.float64)
    self.history = collections.deque(maxlen=10)
    warm = round(1.0 / R.SonicMujocoRunner.sim_dt)
    for step in range(warm):
      if step % R.SonicMujocoRunner.control_decimation == 0:
        self.history.append(R.SonicMujocoRunner._body_state(data, self.last_action))
      R.SonicMujocoRunner._apply_pd(data, R.DEFAULT_ANGLES)
      R.SonicMujocoRunner._apply_tether(model, data, self.anchor)
      mujoco.mj_step(model, data)
    self.frame = 0
    self.elapsed = 0.0
    self.target = R.DEFAULT_ANGLES.copy()

  def control_step(self) -> None:
    R, model, data = self.R, self.model, self.data
    state = R.SonicMujocoRunner._body_state(data, self.last_action)
    self.history.append(state)
    enc_obs = self._enc_obs(state)
    token = self.encoder.run(None, {self.enc_in: enc_obs})[0]
    obs = R.SonicMujocoRunner._history_observation(token, self.history)
    action = self.decoder.run(None, {self.dec_in: obs})[0][0]
    if np.all(np.isfinite(action)):
      self.last_action = action.astype(np.float64)
      self.target = (
        R.DEFAULT_ANGLES + R.ACTION_SCALE * self.last_action[R.ISAACLAB_TO_MUJOCO]
      )
    self.frame += 1
    if self.frame >= len(self.motion.joint_pos):
      self.reset()  # loop the clip
      return
    for _ in range(R.SonicMujocoRunner.control_decimation):
      data.xfrc_applied[:] = 0.0
      if self.elapsed < 0.5:
        R.SonicMujocoRunner._apply_tether(model, data, self.anchor)
      R.SonicMujocoRunner._apply_pd(data, self.target)
      mujoco.mj_step(model, data)
      self.elapsed += R.SonicMujocoRunner.sim_dt

  def _enc_obs(self, state):
    # _encoder_observation is an instance method upstream but never touches
    # self; call it unbound (verified against runner.py).
    return self.R.SonicMujocoRunner._encoder_observation(
      None, self.motion, self.frame, state.base_quat, self.init_quat
    )

  def hud(self) -> str:
    total = len(self.motion.joint_pos) if self.motion is not None else 0
    return (
      f"SONIC 参考动作 {self.motion_name}   frame {self.frame}/{total}\n"
      f"base z {self.data.qpos[2]:.2f} m   (动作跟踪型:WASD 无效,退格重播,播完自动循环)"
    )


class _GearRanges:
  """TeleopState-compatible command ranges for the gear-wbc policy."""

  vx_range = (-1.0, 1.0)
  vy_range = (-1.0, 1.0)
  wz_range = (-1.0, 1.0)
  height_range = (0.40, 0.80)
  standing_height = 0.74
  has_pitch = True
  pitch_range = (-0.5, 0.5)


class GearWbcBackend:
  """GR00T-WBC / SONIC command-mode policy (vx, vy, wz + height + torso rpy).

  Drives OUR classic-MuJoCo robot (same model as the HOMIE channels) through
  the official frozen G1GearWbcPolicy (Balance+Walk ONNX pair). Controls the
  15 leg+waist joints; arms stay held at the HOMIE default. Needs its own PD
  table + armature (knee 200, waist 250, armature 0.01), applied on switch.
  """

  PD = {  # joint suffix pattern -> (kp, kd), from the official deploy config
    "hip": (150.0, 2.0), "knee": (200.0, 4.0), "ankle": (40.0, 2.0),
    "waist": (250.0, 5.0),
  }

  def __init__(self, repo_root: str):
    import sys
    from pathlib import Path

    root = Path(repo_root)
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))
    from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    class _Groups:
      @staticmethod
      def get_joint_group_indices(group):
        return {"body": np.arange(29), "lower_body": np.arange(15)}[group]

    cfg_path = (
      root / "decoupled_wbc" / "sim2mujoco" / "resources" / "robots" / "g1"
      / "g1_gear_wbc.yaml"
    )
    self.policy = G1GearWbcPolicy(
      robot_model=_Groups(),
      config=str(cfg_path),
      model_path=(
        "policy/GR00T-WholeBodyControl-Balance.onnx,"
        "policy/GR00T-WholeBodyControl-Walk.onnx"
      ),
    )
    self.policy.use_policy_action = True
    self.policy.set_use_teleop_policy_cmd(True)
    self.default_15 = np.asarray(
      self.policy.config["default_angles"], dtype=np.float32
    ).reshape(15)
    self.ranges = _GearRanges()

  def reset_policy(self) -> None:
    self.policy.reset()
    self.policy.use_policy_action = True
    self.policy.set_use_teleop_policy_cmd(True)

  def apply_physics(self, model, joint_names) -> None:
    """Their PD + flat 0.01 armature on the 15 controlled joints."""
    for i, jn in enumerate(joint_names[:15]):
      kp = kd = None
      for pat, (p, d) in self.PD.items():
        if pat in jn:
          kp, kd = p, d
      model.actuator_gainprm[i, 0] = kp
      model.actuator_biasprm[i, 1] = -kp
      model.actuator_biasprm[i, 2] = -kd
      jid = model.joint(jn).id
      model.dof_armature[model.jnt_dofadr[jid]] = 0.01

  def targets(self, model, data, qadr, dadr, cmd, height, pitch) -> np.ndarray:
    self.policy.set_observation(
      {
        "q": data.qpos[qadr].astype(np.float32, copy=True),
        "dq": data.qvel[dadr].astype(np.float32, copy=True),
        "floating_base_pose": data.qpos[:7].astype(np.float32, copy=True),
        "floating_base_vel": data.qvel[:6].astype(np.float32, copy=True),
      }
    )
    result = self.policy.get_action(
      arms_target_pose=None,
      base_height_command=np.asarray([height], dtype=np.float32),
      torso_orientation_rpy=np.asarray([0.0, pitch, 0.0], dtype=np.float32),
      interpolated_navigate_cmd=np.asarray(cmd, dtype=np.float32),
    )
    t = np.asarray(result["body_action"][0], dtype=np.float32).reshape(15)
    return t if np.all(np.isfinite(t)) else self.default_15.copy()


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
  parser.add_argument(
    "--sonic",
    action="append",
    default=[],
    metavar="MOTION",
    help="repeatable GEAR-SONIC reference-motion channels (dir name under "
    "gear_sonic_deploy/reference/example)",
  )
  parser.add_argument(
    "--sonic-root", default="/home/jz5725/Projects/GR00T-WholeBodyControl"
  )
  parser.add_argument(
    "--gear-wbc",
    action="store_true",
    help="add the GR00T-WBC/SONIC command-mode channel (vx/vy/wz + height + "
    "torso pitch on our robot model)",
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

  gear = GearWbcBackend(args.sonic_root) if args.gear_wbc else None
  sonic = SonicBackend(args.sonic_root) if args.sonic else None
  sonic_names = [f"SONIC·{m.split('__')[0]}" for m in args.sonic]
  all_names = (
    [n for n, _ in registry]
    + (["SONIC-cmd·gear-wbc"] if gear else [])
    + sonic_names
  )
  n_homie = len(registry)
  gear_idx = n_homie if gear else -1
  sonic_base = n_homie + (1 if gear else 0)

  active = 0
  policy = registry[0][1]
  model = build_model(policy)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=480, width=640)
  state = TeleopState(policy)
  reset_robot(model, data, policy)
  policy.reset()

  policy_joint_names = list(policy.joint_names)
  qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in policy.joint_names])
  dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in policy.joint_names])

  shared = Shared()
  server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
  threading.Thread(target=server.serve_forever, daemon=True).start()
  print(f"serving on http://{args.host}:{args.port}  "
        f"policies: {[n for n, _ in registry]}")

  import PIL.Image

  homie_armature = model.dof_armature.copy()

  def kind_of(idx: int) -> str:
    if idx < n_homie:
      return "homie"
    if idx == gear_idx:
      return "gear"
    return "sonic"

  def gear_reset() -> None:
    reset_robot(model, data, registry[0][1])  # arms/base from HOMIE defaults
    for i in range(15):
      adr = model.jnt_qposadr[model.joint(policy_joint_names[i]).id]
      data.qpos[adr] = float(gear.default_15[i])
    data.qpos[2] = 0.76
    for aid in range(model.nu):
      data.ctrl[aid] = data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]]
    mujoco.mj_forward(model, data)
    gear.reset_policy()

  def switch(idx: int) -> None:
    nonlocal active, policy, state
    if not (0 <= idx < len(all_names)) or idx == active:
      return
    kind = kind_of(idx)
    if kind == "sonic":
      active = idx
      sonic.set_motion(args.sonic[idx - sonic_base])
      print(f"\nswitched to {all_names[idx]}")
      return
    if kind == "gear":
      active = idx
      gear.apply_physics(model, policy_joint_names)
      state = TeleopState(gear.ranges)
      gear_reset()
      print(f"\nswitched to {all_names[idx]}")
      return
    active = idx
    policy = registry[idx][1]
    # Same robot; re-apply this export's PD table + our armature (the gear
    # channel overwrites both).
    model.dof_armature[:] = homie_armature
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
    shared.push_key("ArrowDown")
    if len(all_names) > 1:
      shared.push_key(str(len(all_names)))  # switch to the LAST channel

  camera = CameraState()
  sonic_renderer = None

  while t_end is None or time.time() < t_end:
    t0 = time.perf_counter()
    kind = kind_of(active)
    for k in shared.pop_keys():
      if camera.handle(k if k.startswith("cam:") else k.lower()):
        continue
      if k.isdigit() and k != "0":
        switch(int(k) - 1)
      elif kind == "sonic":
        if k == "Backspace":
          sonic.reset()
      elif k.lower() in KEYMAP or k in KEYMAP:
        state.key(KEYMAP.get(k, KEYMAP.get(k.lower())))
    kind = kind_of(active)

    if kind == "sonic":
      sonic.control_step()
      cur_data = sonic.data
      hud_top = sonic.hud()
    else:
      if state.reset_requested:
        state.reset_requested = False
        state.reset_commands()
        if kind == "gear":
          gear_reset()
        else:
          reset_robot(model, data, policy)
          policy.reset()
      state.slew(control_dt)
      if kind == "gear":
        t15 = gear.targets(
          model, data, qadr, dadr,
          [state.vx, state.vy, state.wz], state.height, state.pitch,
        )
        data.ctrl[:15] = t15  # legs+waist; arms hold their reset targets
      else:
        one = policy.one_step_obs(
          state.command(),
          data.qvel[3:6].astype(np.float32),
          data.qpos[3:7].astype(np.float32),
          data.qpos[qadr].astype(np.float32),
          data.qvel[dadr].astype(np.float32),
        )
        data.ctrl[:] = policy.act(one)  # actuator order == joint_names order
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
      cur_data = data
      g = rt.gravity_orientation(data.qpos[3:7].astype(np.float32))
      tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
      hud_top = (
        f"policy {all_names[active]}   {state.status()}\n"
        f"base z {data.qpos[2]:.2f} m   tilt {tilt:4.1f}°   "
        f"{'⚠ 已倒,按退格复位' if tilt > 60 else ''}"
      )

    frame_tick += 1
    if frame_tick % 2 == 0:  # ~25 fps
      if kind == "sonic":
        if sonic_renderer is None:
          sonic_renderer = mujoco.Renderer(sonic.model, height=480, width=640)
        rnd = sonic_renderer
      else:
        rnd = renderer
      cam = mujoco.MjvCamera()
      camera.apply(cam, cur_data)
      rnd.update_scene(cur_data, camera=cam)
      buf = io.BytesIO()
      PIL.Image.fromarray(rnd.render()).save(buf, "JPEG", quality=75)
      with shared.lock:
        shared.jpeg = buf.getvalue()
    shared.status = {
      "policies": all_names,
      "active": active,
      "hud": hud_top + f"   镜头 {camera.mode}",
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
      f"hud={status['hud']!r}"
    )


if __name__ == "__main__":
  main()
