"""Interactive MuJoCo teleop for HOMIE ONNX policies (sim2sim).

Drives the exported policy through the SAME runtime.py observation pipeline
used by the real-robot deploy script and the BiGym plugin, inside plain
CPU MuJoCo (classic, not warp) — so what you see here is the deployment
stack end to end, physics independent of the training engine.

Keys (tap to adjust, values print to the terminal):
  W / S        vx +/- 0.1 m/s          Space  zero twist
  A / D        vy +/- 0.1 (A = left)   0      reset all commands
  Q / E        wz +/- 0.1 (Q = CCW)    Backspace  reset robot pose
  Up / Down    height target +/- 0.05 m
Mouse: double-click a body, then Ctrl+drag to shove the robot (native
MuJoCo perturbation) — handy for robustness poking.

Height commands are slewed at the deployment rate (0.3 m/s) before entering
the observation, matching the BiGym backend's button ramp.

Usage:
  uv run python -m mjlab_homierl.scripts.teleop_sim_g1 --onnx <policy.onnx>
  ... --smoke  # 5 s headless self-test, no window
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from mjlab_homierl import runtime as rt

HEIGHT_RATE = 0.3  # m/s, matches the BiGym backend / deploy button rate
PHYSICS_DT = 0.005
DECIMATION = 4  # 50 Hz policy


def build_model(policy: rt.HomieOnnxPolicy) -> mujoco.MjModel:
  """G1 spec + ground plane, PD/armature overridden from training constants.

  The raw asset-zoo spec compiles with armature = 0 — mjlab injects motor
  reflected inertias at Entity build time, so a classic-MuJoCo harness must
  re-inject them or the PD loop is badly under-damped and the robot falls.
  """
  import re

  from mjlab_homierl.robots.unitree_g1_deploy import get_g1_deploy_robot_cfg

  cfg = get_g1_deploy_robot_cfg()
  spec = cfg.spec_fn()
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20, 20, 0.1],
    rgba=[0.82, 0.84, 0.82, 1.0],
    name="teleop_floor",
  )
  spec.worldbody.add_light(pos=[1.5, -1.5, 2.5], dir=[-0.5, 0.5, -1])
  spec.worldbody.add_light(pos=[-1.5, 1.5, 2.5], dir=[0.5, -0.5, -1])

  # The raw asset-zoo spec ships with NO actuators (mjlab adds them at
  # Entity build time; nu would be 0 and ctrl writes silently no-ops).
  # Add a position actuator per policy joint with the metadata PD table —
  # exactly what the real robot and the BiGym plugin run — and the
  # articulation cfg's effort limits.
  effort: dict[str, float] = {}
  for act_cfg in cfg.articulation.actuators:
    if act_cfg.effort_limit is None:
      continue
    for pattern in act_cfg.target_names_expr:
      rx = re.compile(pattern)
      for jn in policy.joint_names:
        if rx.fullmatch(jn):
          effort[jn] = float(act_cfg.effort_limit)
  for i, jn in enumerate(policy.joint_names):
    a = spec.add_actuator()
    a.name = jn
    a.target = jn
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    kp, kd = float(policy.kps[i]), float(policy.kds[i])
    a.gainprm[0] = kp
    a.biasprm[0] = 0.0
    a.biasprm[1] = -kp
    a.biasprm[2] = -kd
    if jn in effort:
      a.forcerange[0] = -effort[jn]
      a.forcerange[1] = effort[jn]

  model = spec.compile()
  model.opt.timestep = PHYSICS_DT
  # Training-parity solver options (mjlab sim cfg). The critical one is the
  # implicitfast integrator: with knee kp = 300 at a 5 ms step, explicit
  # Euler is marginally unstable and the robot shakes itself over.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  model.opt.tolerance = 1e-8
  model.opt.ls_tolerance = 0.01

  # Armature per joint from the articulation cfg patterns.
  for act_cfg in cfg.articulation.actuators:
    if act_cfg.armature is None:
      continue
    for pattern in act_cfg.target_names_expr:
      rx = re.compile(pattern)
      for jid in range(model.njnt):
        if rx.fullmatch(model.joint(jid).name or ""):
          model.dof_armature[model.jnt_dofadr[jid]] = float(act_cfg.armature)

  assert model.nu == len(policy.joint_names), (
    f"actuator count {model.nu} != {len(policy.joint_names)} policy joints"
  )
  return model


class TeleopState:
  """Raw command targets (keyboard) + slewed setpoints (observation)."""

  def __init__(self, policy: rt.HomieOnnxPolicy):
    self.p = policy
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.height_target = policy.standing_height
    self.height = self.height_target
    self.reset_requested = False

  def zero_twist(self) -> None:
    self.vx = self.vy = self.wz = 0.0

  def reset_commands(self) -> None:
    self.zero_twist()
    self.height_target = self.p.standing_height

  def slew(self, dt: float) -> None:
    dh = np.clip(self.height_target - self.height, -HEIGHT_RATE * dt, HEIGHT_RATE * dt)
    self.height = float(self.height + dh)

  def command(self) -> np.ndarray:
    return np.array([self.vx, self.vy, self.wz, self.height], dtype=np.float32)

  def status(self) -> str:
    return (
      f"vx {self.vx:+.2f}  vy {self.vy:+.2f}  wz {self.wz:+.2f}  "
      f"height {self.height_target:.2f}"
    )

  def key(self, keycode: int) -> None:
    p = self.p
    if keycode in (ord("W"), ord("w")):
      self.vx = float(np.clip(self.vx + 0.1, p.vx_range[0], p.vx_range[1]))
    elif keycode in (ord("S"), ord("s")):
      self.vx = float(np.clip(self.vx - 0.1, p.vx_range[0], p.vx_range[1]))
    elif keycode in (ord("A"), ord("a")):
      self.vy = float(np.clip(self.vy + 0.1, p.vy_range[0], p.vy_range[1]))
    elif keycode in (ord("D"), ord("d")):
      self.vy = float(np.clip(self.vy - 0.1, p.vy_range[0], p.vy_range[1]))
    elif keycode in (ord("Q"), ord("q")):
      self.wz = float(np.clip(self.wz + 0.1, p.wz_range[0], p.wz_range[1]))
    elif keycode in (ord("E"), ord("e")):
      self.wz = float(np.clip(self.wz - 0.1, p.wz_range[0], p.wz_range[1]))
    elif keycode == 265:  # Up arrow
      self.height_target = float(
        np.clip(self.height_target + 0.05, p.height_range[0], p.height_range[1])
      )
    elif keycode == 264:  # Down arrow
      self.height_target = float(
        np.clip(self.height_target - 0.05, p.height_range[0], p.height_range[1])
      )
    elif keycode == 32:  # Space
      self.zero_twist()
    elif keycode == ord("0"):
      self.reset_commands()
    elif keycode == 259:  # Backspace
      self.reset_requested = True
    else:
      return
    print(f"\r  {self.status()}   ", end="", flush=True)


def reset_robot(
  model: mujoco.MjModel, data: mujoco.MjData, policy: rt.HomieOnnxPolicy
) -> None:
  """Reset to the policy's training default pose (raw spec has no keyframe —
  mjlab injects the HOME keyframe at Entity build time, so it must come from
  the ONNX metadata here)."""
  mujoco.mj_resetData(model, data)
  data.qpos[2] = float(policy.meta.get("init_base_height", 0.783675))
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for i, n in enumerate(policy.joint_names):
    data.qpos[model.jnt_qposadr[model.joint(n).id]] = float(policy.default_pos[i])
  # PD targets to the current pose so the robot doesn't twitch at t=0.
  for aid in range(model.nu):
    jid = model.actuator_trnid[aid, 0]
    data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
  mujoco.mj_forward(model, data)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", required=True, help="HOMIE ONNX policy")
  parser.add_argument(
    "--smoke", action="store_true", help="5 s headless self-test, no viewer"
  )
  args = parser.parse_args()

  control_dt = PHYSICS_DT * DECIMATION
  policy = rt.HomieOnnxPolicy(args.onnx)
  model = build_model(policy)
  data = mujoco.MjData(model)

  qadr = np.array(
    [model.jnt_qposadr[model.joint(n).id] for n in policy.joint_names]
  )
  dadr = np.array(
    [model.jnt_dofadr[model.joint(n).id] for n in policy.joint_names]
  )
  act_jname = [
    model.joint(model.actuator_trnid[aid, 0]).name for aid in range(model.nu)
  ]
  act_targets_idx = {n: i for i, n in enumerate(policy.joint_names)}

  state = TeleopState(policy)
  reset_robot(model, data, policy)
  policy.reset()

  print(
    f"Loaded {args.onnx}\n  {policy.num_commands}-dim command, "
    f"height {policy.height_range}, one-step obs {policy.num_one_step_obs}"
  )
  print(f"  {state.status()}")

  def control_step() -> None:
    state.slew(control_dt)
    quat = data.qpos[3:7].astype(np.float32)  # wxyz
    gyro = data.qvel[3:6].astype(np.float32)  # base-local (freejoint)
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

  if args.smoke:
    for i in range(int(5.0 / control_dt)):
      if i == 50:
        state.height_target = 0.35
      control_step()
    g = rt.gravity_orientation(data.qpos[3:7].astype(np.float32))
    tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
    print(
      f"smoke: 5 s simulated, base z {data.qpos[2]:.3f} (cmd 0.35), "
      f"pelvis tilt {tilt:.0f} deg -> {'OK' if tilt < 45 else 'FELL'}"
    )
    return

  from mujoco import viewer as mj_viewer

  def key_cb(keycode: int) -> None:
    state.key(keycode)

  with mj_viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
    while viewer.is_running():
      t0 = time.perf_counter()
      if state.reset_requested:
        state.reset_requested = False
        state.reset_commands()
        reset_robot(model, data, policy)
        policy.reset()
      control_step()
      viewer.sync()
      # Real-time pacing.
      leftover = control_dt - (time.perf_counter() - t0)
      if leftover > 0:
        time.sleep(leftover)


if __name__ == "__main__":
  main()
