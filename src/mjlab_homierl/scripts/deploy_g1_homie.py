"""Deploy a G1 HOMIE/HOMIE+ lower-body ONNX policy to the real robot.

Everything policy-specific is read from the ONNX metadata contract (joint
order, PD gains, default pose, obs layout/scales/history, command ranges,
optional torso-pitch channel) — no hardcoded conventions. The DDS side follows
unitree_rl_gym's deploy_real.py state machine: zero-torque -> move-to-default
-> wait for START -> run at 50 Hz -> damping on exit. This process owns the
full 29-motor LowCmd: legs get policy targets, waist/arms are held at the
default pose with the training-time gains (upper-body teleop integration can
replace that block later).

Remote mapping: left stick = vx/vy, right stick x = yaw rate, dpad up/down =
height command +-, X/B = torso pitch -/+ (HOMIE+ models only), SELECT = exit
to damping mode.

Usage:
  Real robot (requires unitree_sdk2py, robot in debug/low-level mode, hung
  from a harness for first trials):
    uv run python -m mjlab_homierl.scripts.deploy_g1_homie --onnx <file> --net eth0

  Local validation (no SDK needed; runs the same obs builder against the
  mjlab-compiled MuJoCo model and scripted commands):
    uv run --extra deploy python -m mjlab_homierl.scripts.deploy_g1_homie --onnx <file> --sim
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import numpy as np

# Canonical Unitree G1 29-dof low-level motor order (hg LowCmd/LowState).
G1_MOTOR_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
  """Project the world -z gravity direction into the base frame."""
  w, x, y, z = quat_wxyz
  return np.array(
    [
      -2.0 * (x * z - w * y),
      -2.0 * (y * z + w * x),
      -(1.0 - 2.0 * (x * x + y * y)),
    ],
    dtype=np.float32,
  )


class HomieOnnxPolicy:
  """ONNX session + the metadata-driven observation builder."""

  def __init__(self, onnx_path: str):
    import onnxruntime as ort

    self.session = ort.InferenceSession(
      onnx_path, providers=["CPUExecutionProvider"]
    )
    self.meta = dict(self.session.get_modelmeta().custom_metadata_map)
    self.input_name = self.session.get_inputs()[0].name

    m = self.meta
    self.joint_names = m["joint_names"].split(",")
    self.action_joint_names = m["action_joint_names"].split(",")
    self.default_pos = np.array(
      [float(v) for v in m["default_joint_pos"].split(",")], dtype=np.float32
    )
    self.kps = np.array(
      [float(v) for v in m["joint_stiffness"].split(",")], dtype=np.float32
    )
    self.kds = np.array(
      [float(v) for v in m["joint_damping"].split(",")], dtype=np.float32
    )
    self.action_scale = float(m["action_scale"])
    self.history_length = int(m["obs_history_length"])
    self.num_one_step_obs = int(m["num_one_step_obs"])
    layout = json.loads(m["one_step_obs_layout"])
    self.num_commands = int(layout["command"])
    self.has_pitch = self.num_commands >= 5
    self.scale_lin_vel = float(m["obs_scale_lin_vel"])
    self.scale_ang_vel = float(m["obs_scale_ang_vel"])
    self.scale_dof_pos = float(m["obs_scale_dof_pos"])
    self.scale_dof_vel = float(m["obs_scale_dof_vel"])
    twist = json.loads(m["twist_command_ranges"])
    self.vx_range = tuple(twist["lin_vel_x"])
    self.vy_range = tuple(twist["lin_vel_y"])
    self.wz_range = tuple(twist["ang_vel_z"])
    self.height_range = tuple(
      float(v) for v in m["height_command_range"].split(",")
    )
    self.standing_height = float(m["standing_height"])
    self.pitch_range = (0.0, 0.0)
    if self.has_pitch:
      pr = json.loads(m["pitch_command_ranges"])
      lows, highs = zip(pr["walk"], pr["squat"])
      self.pitch_range = (min(lows), max(highs))

    expected = self.num_commands + 6 + 2 * len(self.joint_names) + len(
      self.action_joint_names
    )
    if expected != self.num_one_step_obs:
      raise ValueError(
        f"Metadata inconsistent: layout sums to {expected}, "
        f"num_one_step_obs is {self.num_one_step_obs}."
      )

    self.action_ids = [self.joint_names.index(n) for n in self.action_joint_names]
    self.last_action = np.zeros(len(self.action_joint_names), dtype=np.float32)
    self.history: deque[np.ndarray] = deque(maxlen=self.history_length)

  def reset(self) -> None:
    self.last_action[:] = 0.0
    self.history.clear()

  def one_step_obs(
    self,
    command: np.ndarray,
    ang_vel: np.ndarray,
    quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
  ) -> np.ndarray:
    cmd = command.astype(np.float32).copy()
    cmd[0:2] *= self.scale_lin_vel
    cmd[2] *= self.scale_ang_vel  # height (cmd[3]) and pitch (cmd[4]) unscaled
    return np.concatenate(
      [
        cmd,
        ang_vel.astype(np.float32) * self.scale_ang_vel,
        gravity_orientation(quat_wxyz),
        (joint_pos - self.default_pos) * self.scale_dof_pos,
        joint_vel.astype(np.float32) * self.scale_dof_vel,
        self.last_action,
      ]
    )

  def act(self, one_step: np.ndarray) -> np.ndarray:
    """Push one observation step, return leg position targets (29-dim mask)."""
    if not self.history:
      self.history.extend([one_step] * self.history_length)
    else:
      self.history.append(one_step)
    obs = np.concatenate(list(self.history))[None].astype(np.float32)
    action = self.session.run(None, {self.input_name: obs})[0][0]
    self.last_action = action.astype(np.float32)
    targets = self.default_pos.copy()
    targets[self.action_ids] += self.action_scale * action
    return targets


class CommandState:
  """Joystick-driven (vx, vy, wz, height[, pitch]) command with rate limits."""

  def __init__(self, policy: HomieOnnxPolicy):
    self.p = policy
    self.height = policy.standing_height
    self.pitch = 0.0

  def vector(self, lx, ly, rx, height_step=0.0, pitch_step=0.0) -> np.ndarray:
    p = self.p
    vx = ly * (p.vx_range[1] if ly >= 0 else -p.vx_range[0])
    vy = -lx * p.vy_range[1]
    wz = -rx * p.wz_range[1]
    self.height = float(
      np.clip(self.height + height_step, p.height_range[0], p.height_range[1])
    )
    cmd = [vx, vy, wz, self.height]
    if p.has_pitch:
      self.pitch = float(
        np.clip(self.pitch + pitch_step, p.pitch_range[0], p.pitch_range[1])
      )
      cmd.append(self.pitch)
    return np.array(cmd, dtype=np.float32)


##
# Real-robot path (unitree_sdk2py; structure follows unitree_rl_gym
# deploy_real.py).
##


def run_real(policy: HomieOnnxPolicy, net_iface: str, control_dt: float) -> None:
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

  # Import the remote/cmd helpers without touching the mjlab_homierl package
  # (a robot-side machine only needs numpy + onnxruntime + unitree_sdk2py).
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

  motor_of = {n: i for i, n in enumerate(G1_MOTOR_ORDER)}
  jm = [motor_of[n] for n in policy.joint_names]  # metadata order -> motor idx
  leg_motors = [motor_of[n] for n in policy.action_joint_names]

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
    for k, name in enumerate(policy.joint_names):
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
  cmd_state = CommandState(policy)
  height_step = 0.05 * control_dt / 0.5  # full step in 0.5 s of holding
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
      dp = height_step * (
        (remote.button[KeyMap.B] == 1) - (remote.button[KeyMap.X] == 1)
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
  del leg_motors  # (legs are the only entries of targets that move)


##
# Sim validation path: same obs builder against the mjlab-compiled model.
##


def run_sim(policy: HomieOnnxPolicy, task: str) -> int:
  """Validate the deploy-side obs builder against the mjlab plant.

  Runs the exact training environment (1 env, play cfg, startup DR stripped)
  but drives it with THIS script's observation assembly + the ONNX session,
  and cross-checks every step against mjlab's own actor observation. A max
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
      targets = policy.act(one_step)
      del targets  # env applies default + scale internally from raw action
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
  args = parser.parse_args()

  policy = HomieOnnxPolicy(args.onnx)
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
  run_real(policy, args.net, args.control_dt)


if __name__ == "__main__":
  main()
