#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import torch


def _quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
  w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
  q_inv = np.array([w, -x, -y, -z], dtype=np.float32)

  def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
      [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
      ],
      dtype=np.float32,
    )

  qv = np.array([0.0, float(v_xyz[0]), float(v_xyz[1]), float(v_xyz[2])], dtype=np.float32)
  return qmul(qmul(q_inv, qv), q_wxyz.astype(np.float32))[1:4]


@dataclass(frozen=True)
class PolicyIO:
  obs_dim: int
  act_dim: int
  joint_obs_dim: int
  joint_names_in_obs_order: tuple[str, ...]
  action_joint_names: tuple[str, ...]


class HomieLowerBodyPolicy:
  """Loads HOMIE H1-with_hands lower-body policy from homie_rl.pt (RSL-RL ActorCritic)."""

  def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    sd = ckpt["model_state_dict"]

    self.obs_dim = int(sd["actor_obs_normalizer._mean"].shape[-1])
    self.act_dim = int(sd["actor.6.weight"].shape[0])
    if self.obs_dim != 97 or self.act_dim != 10:
      raise ValueError(
        f"Unexpected policy dims: obs_dim={self.obs_dim}, act_dim={self.act_dim} "
        "(expected 97->10 for H1-with_hands homie_rl.pt)."
      )

    self.actor = torch.nn.Sequential(
      torch.nn.Linear(self.obs_dim, 512),
      torch.nn.ELU(),
      torch.nn.Linear(512, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, self.act_dim),
    )

    actor_sd = {k.removeprefix("actor."): v for k, v in sd.items() if k.startswith("actor.")}
    self.actor.load_state_dict(actor_sd, strict=True)
    self.actor.eval()

    self._mean = sd["actor_obs_normalizer._mean"].to(dtype=torch.float32)
    self._std = sd["actor_obs_normalizer._std"].to(dtype=torch.float32)
    self._eps = 1e-2  # rsl_rl.networks.EmpiricalNormalization default

    self.device = torch.device(device)
    self.actor.to(self.device)
    self._mean = self._mean.to(self.device)
    self._std = self._std.to(self.device)

  @torch.no_grad()
  def act(self, obs: np.ndarray) -> np.ndarray:
    if obs.shape != (self.obs_dim,):
      raise ValueError(f"Expected obs shape ({self.obs_dim},), got {obs.shape}")
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    obs_t = (obs_t - self._mean) / (self._std + self._eps)
    act_t = self.actor(obs_t)
    return act_t.squeeze(0).detach().cpu().numpy()


def _build_model_with_floor(xml_path: Path) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # Add a plane so the robot can actually walk.
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(20.0, 20.0, 0.1),
    rgba=(0.92, 0.92, 0.92, 1.0),
    contype=1,
    conaffinity=1,
    friction=(1.0, 0.005, 0.0001),
  )
  return spec.compile()


def _get_policy_io(model: mujoco.MjModel) -> PolicyIO:
  nonfree_jids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  nonfree_jids.sort(key=lambda j: int(model.jnt_qposadr[j]))
  joint_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in nonfree_jids
  )
  if len(joint_names) != 37:
    raise ValueError(f"Expected 37 non-free joints for H1-with_hands, got {len(joint_names)}")

  action_joint_names = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
  )
  return PolicyIO(
    obs_dim=97,
    act_dim=10,
    joint_obs_dim=len(joint_names),
    joint_names_in_obs_order=joint_names,
    action_joint_names=action_joint_names,
  )


def _sim_loop(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  policy: HomieLowerBodyPolicy,
  *,
  cmd_vx: float,
  cmd_vy: float,
  cmd_wz: float,
  cmd_height: float,
  decimation: int,
  realtime: bool,
  max_steps: int | None,
  fixed_height: float | None = None,
  step_callback: Callable[[], None] | None = None,
  should_continue: Callable[[], bool] | None = None,
) -> None:
  io = _get_policy_io(model)

  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id < 0:
    raise ValueError("Missing keyframe 'init_state' in the MJCF.")
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  # Set fixed height if specified (z coordinate is at index 2 for free joint)
  if fixed_height is not None:
    data.qpos[2] = fixed_height
    # Also set z velocity to 0 to prevent falling
    data.qvel[2] = 0.0
  mujoco.mj_forward(model, data)

  # Sensors used by policy obs.
  imu_ang_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
  imu_lin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_vel")
  if imu_ang_id < 0 or imu_lin_id < 0:
    raise ValueError("Missing IMU sensors (expected 'imu_ang_vel' and 'imu_lin_vel').")

  ang_adr, ang_dim = int(model.sensor_adr[imu_ang_id]), int(model.sensor_dim[imu_ang_id])
  lin_adr, lin_dim = int(model.sensor_adr[imu_lin_id]), int(model.sensor_dim[imu_lin_id])
  if ang_dim != 3 or lin_dim != 3:
    raise ValueError(f"Unexpected sensor dims: imu_ang_vel={ang_dim}, imu_lin_vel={lin_dim}")

  # Joint qpos/qvel addresses (policy uses all non-free joints).
  nonfree_jids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  nonfree_jids.sort(key=lambda j: int(model.jnt_qposadr[j]))
  j_qadr = np.asarray([int(model.jnt_qposadr[j]) for j in nonfree_jids], dtype=np.int32)
  j_vadr = np.asarray([int(model.jnt_dofadr[j]) for j in nonfree_jids], dtype=np.int32)

  # Default joint state from init_state keyframe.
  default_joint_pos = data.qpos[j_qadr].copy()
  default_joint_vel = data.qvel[j_vadr].copy()

  # Map policy actions to actuators (named after joints).
  act_ids = []
  act_qposadr = []
  default_action_pos = []
  action_scale = []
  for jname in io.action_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
      raise ValueError(f"Missing joint '{jname}' in model.")
    qadr = int(model.jnt_qposadr[jid])
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    if act_id < 0:
      raise ValueError(f"Missing position actuator '{jname}' (expected actuator name == joint name).")

    stiffness = float(model.actuator_gainprm[act_id, 0])
    effort = float(abs(model.actuator_forcerange[act_id, 1]))
    if stiffness <= 0.0 or effort <= 0.0:
      raise ValueError(f"Invalid actuator params for '{jname}': stiffness={stiffness}, effort={effort}")

    act_ids.append(act_id)
    act_qposadr.append(qadr)
    default_action_pos.append(float(data.qpos[qadr]))
    action_scale.append(0.25 * effort / stiffness)  # matches mjlab.asset_zoo.robots.H1_ACTION_SCALE

  act_ids_np = np.asarray(act_ids, dtype=np.int32)
  default_action_pos_np = np.asarray(default_action_pos, dtype=np.float32)
  action_scale_np = np.asarray(action_scale, dtype=np.float32)

  last_action = np.zeros((io.act_dim,), dtype=np.float32)
  cmd = np.asarray([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)
  height_cmd = np.asarray([cmd_height], dtype=np.float32)

  dt_policy = float(model.opt.timestep) * float(decimation)
  steps_done = 0

  while (max_steps is None or steps_done < max_steps) and (
    should_continue is None or bool(should_continue())
  ):
    t0 = time.perf_counter()

    mujoco.mj_forward(model, data)

    # Base velocities from IMU sensors (torso-site frame, matches HOMIE config).
    base_ang = data.sensordata[ang_adr : ang_adr + ang_dim].copy()
    base_lin = data.sensordata[lin_adr : lin_adr + lin_dim].copy()

    # Projected gravity in pelvis body frame.
    proj_g = _quat_rotate_inverse_wxyz(data.qpos[3:7], np.array([0.0, 0.0, -1.0], dtype=np.float32))

    joint_pos_rel = data.qpos[j_qadr] - default_joint_pos
    joint_vel_rel = data.qvel[j_vadr] - default_joint_vel

    obs = np.concatenate(
      [
        base_lin,
        base_ang,
        proj_g,
        joint_pos_rel.astype(np.float32, copy=False),
        joint_vel_rel.astype(np.float32, copy=False),
        last_action,
        cmd,
        height_cmd,
      ]
    ).astype(np.float32, copy=False)

    action = policy.act(obs).astype(np.float32, copy=False)
    last_action = action

    # Apply position targets to lower-body actuators.
    target_pos = default_action_pos_np + action_scale_np * action
    data.ctrl[act_ids_np] = target_pos

    for _ in range(decimation):
      mujoco.mj_step(model, data)
      # Maintain fixed height if specified
      if fixed_height is not None:
        data.qpos[2] = fixed_height
        data.qvel[2] = 0.0

    steps_done += 1

    if step_callback is not None:
      step_callback()

    if realtime:
      elapsed = time.perf_counter() - t0
      if dt_policy > elapsed:
        time.sleep(dt_policy - elapsed)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run HOMIE lower-body policy (homie_rl.pt) on Unitree H1-with_hands in pure MuJoCo (no mjlab)."
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("homie_rl.pt"),
    help="Path to homie_rl.pt checkpoint (expects keys: model_state_dict/actor_obs_normalizer).",
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=Path("src/mjlab/asset_zoo/robots/unitree_h1/xmls/h1-with_hands.xml"),
    help="Path to H1-with_hands MJCF (generated in this repo).",
  )
  parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu/cuda:0.")
  parser.add_argument("--cmd-vx", type=float, default=0.8, help="Desired forward velocity (body frame), m/s.")
  parser.add_argument("--cmd-vy", type=float, default=0.0, help="Desired lateral velocity (body frame), m/s.")
  parser.add_argument("--cmd-wz", type=float, default=0.0, help="Desired yaw rate (body frame), rad/s.")
  parser.add_argument(
    "--cmd-height",
    type=float,
    default=0.98,
    help="Desired relative height command (pelvis z - min(foot z)), meters.",
  )
  parser.add_argument(
    "--fixed-height",
    type=float,
    default=None,
    help="Fixed height for robot root body (z coordinate), meters. Robot will be constrained to this height throughout simulation.",
  )
  parser.add_argument("--timestep", type=float, default=0.005, help="MuJoCo timestep (s).")
  parser.add_argument("--decimation", type=int, default=4, help="Physics steps per policy step.")
  parser.add_argument("--iterations", type=int, default=10, help="MuJoCo solver iterations.")
  parser.add_argument("--ls-iterations", type=int, default=20, help="MuJoCo solver LS iterations.")
  parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible.")
  parser.add_argument("--steps", type=int, default=2000, help="Number of policy steps to run (headless).")
  parser.add_argument(
    "--viewer",
    type=str,
    default="passive",
    choices=("passive", "none"),
    help="Use MuJoCo viewer (passive) or run headless (none).",
  )
  args = parser.parse_args()

  if not args.xml.exists():
    raise FileNotFoundError(f"Missing MJCF: {args.xml}")
  if not args.checkpoint.exists():
    raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
  if args.decimation <= 0:
    raise ValueError("--decimation must be > 0")

  policy = HomieLowerBodyPolicy(args.checkpoint, device=args.device)
  model = _build_model_with_floor(args.xml)
  data = mujoco.MjData(model)

  model.opt.timestep = float(args.timestep)
  model.opt.iterations = int(args.iterations)
  model.opt.ls_iterations = int(args.ls_iterations)

  if args.viewer == "passive":
    import mujoco.viewer as mj_viewer  # noqa: PLC0415

    with mj_viewer.launch_passive(model, data) as viewer:
      _sim_loop(
        model,
        data,
        policy,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_wz=args.cmd_wz,
        cmd_height=args.cmd_height,
        decimation=args.decimation,
        realtime=not args.no_realtime,
        max_steps=None,
        fixed_height=args.fixed_height,
        step_callback=viewer.sync,
        should_continue=viewer.is_running,
      )
  else:
    _sim_loop(
      model,
      data,
      policy,
      cmd_vx=args.cmd_vx,
      cmd_vy=args.cmd_vy,
      cmd_wz=args.cmd_wz,
      cmd_height=args.cmd_height,
      decimation=args.decimation,
      realtime=not args.no_realtime,
      max_steps=int(args.steps),
      fixed_height=args.fixed_height,
    )


if __name__ == "__main__":
  main()
