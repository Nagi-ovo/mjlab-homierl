"""Deploy a G1 HOMIE/HOMIE+ lower-body ONNX policy to the real robot.

All policy-specific conventions come from the ONNX metadata contract via
``mjlab_homierl.runtime`` (a standalone, mjlab-free module — the same file the
BiGym plugin vendors). The DDS side follows unitree_rl_gym's deploy_real.py
state machine: zero-torque -> move-to-default -> wait for START -> run at
50 Hz -> damping on exit. This process owns the full 29-motor LowCmd: legs
get policy targets, waist/arms are held at the default pose with the
training-time gains (upper-body teleop integration can replace that block).

Remote mapping: left stick = vx/vy, right stick x = yaw rate, dpad up/down =
height command +-, X/B = torso pitch -/+ (HOMIE+ models only), SELECT = exit
to damping mode.

Usage:
  Real robot (env from scripts/setup_deploy_env.sh; robot in debug/low-level
  mode, hung from a harness for first trials):
    .venv-deploy/bin/python src/mjlab_homierl/scripts/deploy_g1_homie.py \\
      --onnx <file> --net eno1

  Local validation (training venv; cross-checks the runtime obs builder
  bit-for-bit against the mjlab plant):
    uv run --extra deploy python -m mjlab_homierl.scripts.deploy_g1_homie \\
      --onnx <file> --sim
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _load_runtime():
  """Import mjlab_homierl.runtime without requiring the mjlab stack.

  Package import pulls in mjlab (task registration), which a robot-side
  machine does not have — fall back to loading runtime.py by file path.
  """
  try:
    from mjlab_homierl import runtime  # noqa: PLC0415

    return runtime
  except ImportError:
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "runtime.py"
    spec = importlib.util.spec_from_file_location("homie_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load_runtime()


##
# Real-robot path (unitree_sdk2py; structure follows unitree_rl_gym
# deploy_real.py).
##


def run_real(
  policy,
  net_iface: str,
  control_dt: float,
  speed_scale: float = 0.5,
  enable_pitch: bool = False,
) -> None:
  from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
  )
  from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
  )
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
  from unitree_sdk2py.utils.crc import CRC

  # Import the remote/cmd helpers without touching the mjlab_homierl package.
  try:
    from mjlab_homierl.scripts._deploy_common import (
      KeyMap,
      RemoteController,
      create_damping_cmd,
      create_zero_cmd,
      init_cmd_hg,
    )
  except ImportError:
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
      "_deploy_common", pathlib.Path(__file__).resolve().parent / "_deploy_common.py"
    )
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    KeyMap = common.KeyMap
    RemoteController = common.RemoteController
    create_damping_cmd = common.create_damping_cmd
    create_zero_cmd = common.create_zero_cmd
    init_cmd_hg = common.init_cmd_hg

  motor_of = {n: i for i, n in enumerate(rt.G1_MOTOR_ORDER)}
  jm = [motor_of[n] for n in policy.joint_names]  # metadata order -> motor idx

  crc = CRC()
  remote = RemoteController()
  ChannelFactoryInitialize(0, net_iface)
  low_cmd = unitree_hg_msg_dds__LowCmd_()
  low_state = unitree_hg_msg_dds__LowState_()
  state = {"mode_machine": 0, "have_state": False}

  def on_lowstate(msg: LowStateHG):
    nonlocal low_state
    low_state = msg
    state["mode_machine"] = msg.mode_machine
    state["have_state"] = True
    remote.set(msg.wireless_remote)

  pub = ChannelPublisher("rt/lowcmd", LowCmdHG)
  pub.Init()
  sub = ChannelSubscriber("rt/lowstate", LowStateHG)
  sub.Init(on_lowstate, 10)

  def send(cmd):
    cmd.mode_machine = state["mode_machine"]
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)

  def write_all_joints(targets_29: np.ndarray) -> None:
    for k in range(len(policy.joint_names)):
      mc = low_cmd.motor_cmd[jm[k]]
      mc.q = float(targets_29[k])
      mc.qd = 0.0
      mc.kp = float(policy.kps[k])
      mc.kd = float(policy.kds[k])
      mc.tau = 0.0

  print("Waiting for LowState...")
  while not state["have_state"]:
    time.sleep(0.05)
  init_cmd_hg(low_cmd, state["mode_machine"])

  print("Zero-torque. Press START to move to the default pose.")
  while remote.button[KeyMap.start] != 1:
    create_zero_cmd(low_cmd)
    send(low_cmd)
    time.sleep(control_dt)

  # 2 s interpolation from the current pose to the training default pose.
  q0 = np.array([low_state.motor_state[jm[k]].q for k in range(29)])
  steps = int(2.0 / control_dt)
  for i in range(steps):
    alpha = (i + 1) / steps
    write_all_joints(q0 * (1 - alpha) + policy.default_pos * alpha)
    send(low_cmd)
    time.sleep(control_dt)

  print("Holding default pose. Press A to start the policy; SELECT to exit.")
  while remote.button[KeyMap.A] != 1:
    if remote.button[KeyMap.select] == 1:
      create_damping_cmd(low_cmd)
      send(low_cmd)
      return
    write_all_joints(policy.default_pos)
    send(low_cmd)
    time.sleep(control_dt)

  policy.reset()
  cmd_state = rt.CommandState(policy, speed_scale=speed_scale)
  # Height slews at 0.3 m/s while the button is held — inside the v4 trained
  # rate envelope (0.25-0.75 m/s); anything slower degrades toward the
  # heavily-trained steady hold, so the low side is safe too.
  height_step = 0.3 * control_dt
  print("Policy running.")
  try:
    while remote.button[KeyMap.select] != 1:
      t0 = time.perf_counter()
      q = np.array(
        [low_state.motor_state[jm[k]].q for k in range(29)], dtype=np.float32
      )
      dq = np.array(
        [low_state.motor_state[jm[k]].dq for k in range(29)], dtype=np.float32
      )
      quat = np.array(low_state.imu_state.quaternion, dtype=np.float32)  # wxyz
      gyro = np.array(low_state.imu_state.gyroscope, dtype=np.float32)

      dh = height_step * (
        (remote.button[KeyMap.up] == 1) - (remote.button[KeyMap.down] == 1)
      )
      # Pitch is hard-gated OFF by default: this drives waist_pitch_joint at
      # kp 300, and the lab G1 is a G1_29 lock_waist unit (waist roll/pitch
      # mechanically fastened — xr_teleoperate/SETUP_GUIDE.md, verified with
      # check_dof.py). Commanding pitch against the fastener stalls the motor
      # at full torque. Pass --enable-pitch ONLY on a waist-unlocked robot
      # (re-verify with check_dof.py first).
      dp = (
        height_step
        * ((remote.button[KeyMap.B] == 1) - (remote.button[KeyMap.X] == 1))
        if enable_pitch
        else 0.0
      )
      command = cmd_state.vector(remote.lx, remote.ly, remote.rx, dh, dp)

      one_step = policy.one_step_obs(command, gyro, quat, q, dq)
      targets = policy.act(one_step)
      write_all_joints(targets)  # legs = policy, waist/arms = default (held)
      send(low_cmd)
      time.sleep(max(0.0, control_dt - (time.perf_counter() - t0)))
  finally:
    create_damping_cmd(low_cmd)
    send(low_cmd)
    print("Exited to damping mode.")


##
# Sim validation path: the runtime obs builder against the mjlab plant.
##


def run_sim(policy, task: str) -> int:
  """Validate the runtime observation builder against the mjlab plant.

  Runs the exact training environment (1 env, play cfg, startup DR stripped)
  but drives it with the runtime's observation assembly + the ONNX session,
  cross-checking every step against mjlab's own actor observation. A max
  elementwise deviation ~0 certifies the deploy pipeline; tracking stats then
  certify the closed loop.
  """
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  for term in cfg.commands.values():
    term.resampling_time_range = (1e9, 1e9)
  for ev in ("link_mass", "payload_mass", "hand_payload", "encoder_bias",
             "base_com", "foot_friction", "pd_gains"):
    cfg.events.pop(ev, None)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs_dict, _ = env.reset(seed=0)
  robot = env.scene["robot"]
  twist = env.command_manager.get_term("twist")
  height = env.command_manager.get_term("height")
  pitch_term = None
  if policy.has_pitch:
    pitch_term = env.command_manager.get_term("torso_pitch")

  # Map metadata joint order onto the env's joint order.
  env_index = [list(robot.joint_names).index(n) for n in policy.joint_names]

  def pin(vx, wz, h, pitch):
    twist.vel_command_b[:] = 0.0
    twist.vel_command_b[:, 0] = vx
    twist.vel_command_b[:, 2] = wz
    height.height_command[:, 0] = h
    # height_command is a setpoint slewing toward _height_target (v4+).
    # Pinning only height_command leaves the term walking one slew-tick per
    # step toward a stale randomly-sampled target (squat-mode reset draws)
    # — a constant ~1e-2 obs offset that fails the bit-for-bit check. Pin
    # the target too so the command holds exactly at h.
    height._height_target[:] = h
    if pitch_term is not None:
      pitch_term.pitch_command[:, 0] = pitch

  policy.reset()

  # (name, vx, wz, height, pitch, seconds)
  phases = [("stand", 0.0, 0.0, policy.standing_height, 0.0, 3.0),
            ("walk_0.6", 0.6, 0.0, policy.standing_height, 0.0, 4.0),
            ("squat_0.5", 0.0, 0.0, 0.5, 0.0, 4.0)]
  if policy.has_pitch:
    phases.append(("squat_0.5+lean", 0.0, 0.0, 0.5, 0.3, 4.0))

  failures = 0
  for name, vx, wz, h, pitch, dur in phases:
    pin(vx, wz, h, pitch)
    vx_log, h_log, obs_diff = [], [], 0.0
    n_steps = int(dur / env.step_dt)
    for i in range(n_steps):
      cmd = [vx, 0.0, wz, h] + ([pitch] if policy.has_pitch else [])
      one_step = policy.one_step_obs(
        np.array(cmd, dtype=np.float32),
        robot.data.root_link_ang_vel_b[0].cpu().numpy(),
        robot.data.root_link_quat_w[0].cpu().numpy(),
        robot.data.joint_pos[0, env_index].cpu().numpy(),
        robot.data.joint_vel[0, env_index].cpu().numpy(),
      )
      policy.act(one_step)
      # Cross-check the full flattened history against mjlab's actor obs
      # (skip the warmup steps where the two history fills differ).
      if i >= policy.history_length:
        mine = np.concatenate(list(policy.history))
        theirs = obs_dict["actor"][0].cpu().numpy()
        obs_diff = max(obs_diff, float(np.abs(mine - theirs).max()))
      action = torch.from_numpy(policy.last_action).unsqueeze(0)
      obs_dict, _, _, _, _ = env.step(action)
      pin(vx, wz, h, pitch)
      vx_log.append(float(robot.data.root_link_lin_vel_b[0, 0]))
      h_log.append(float(height._compute_relative_height()[0]))

    tail = len(vx_log) // 2
    vx_err = abs(np.mean(vx_log[tail:]) - vx)
    h_err = abs(np.mean(h_log[tail:]) - h)
    fell = min(h_log) < 0.25
    ok = (not fell) and vx_err < 0.25 and obs_diff < 1e-4
    failures += 0 if ok else 1
    print(
      f"{name:>15s}: obs_max_diff {obs_diff:.2e}  vx_err {vx_err:.3f}  "
      f"h_err {h_err:.3f}  {'OK' if ok else 'FAIL'}"
    )
  env.close()
  return failures


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", required=True, help="Policy ONNX with metadata.")
  parser.add_argument("--net", help="Network interface for the real robot.")
  parser.add_argument("--sim", action="store_true", help="Local MuJoCo check.")
  parser.add_argument("--task", default="Mjlab-Homie-Unitree-G1")
  parser.add_argument("--control-dt", type=float, default=0.02)
  parser.add_argument(
    "--enable-pitch",
    action="store_true",
    help="Enable X/B torso-pitch commands. OFF by default: the lab G1 is a "
    "G1_29 lock_waist unit and pitch drives waist_pitch_joint at kp 300 "
    "against the mechanical fastener. Only pass this on a waist-unlocked "
    "robot, after re-verifying with xr_teleoperate/check_dof.py.",
  )
  parser.add_argument(
    "--speed-scale",
    type=float,
    default=0.5,
    help="Fraction of the training twist range at full stick deflection "
    "(default 0.5: fwd 0.6 m/s, back 0.4, yaw 0.4). Raise toward 1.0 once "
    "comfortable.",
  )
  args = parser.parse_args()

  policy = rt.HomieOnnxPolicy(args.onnx)
  print(
    f"Loaded {args.onnx}: {len(policy.joint_names)} joints, "
    f"{len(policy.action_joint_names)} actions, {policy.num_commands}-dim command"
    f"{' (with torso pitch)' if policy.has_pitch else ''}, "
    f"history {policy.history_length} x {policy.num_one_step_obs}."
  )
  if args.sim:
    raise SystemExit(run_sim(policy, args.task))
  if not args.net:
    raise SystemExit("Provide --net <iface> for real deployment or --sim.")
  run_real(
    policy,
    args.net,
    args.control_dt,
    speed_scale=args.speed_scale,
    enable_pitch=args.enable_pitch,
  )


if __name__ == "__main__":
  main()
