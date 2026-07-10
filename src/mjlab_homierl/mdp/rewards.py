"""HOMIE reward terms.

Formulas and default parameters follow the OpenHomie reference implementation
(HomieRL/legged_gym, G1 29-dof config). Robot-specific values (weights, gates,
actuator limits) are wired in the task configs.

Conventions:
- Penalty terms return positive costs and are combined with negative weights.
- Height-gated terms use the relative-height command: the gate is open when the
  commanded height is at (or near) the standing height, mirroring OpenHomie's
  ``commands[:, 4] >= 0.735`` gating on G1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _height_gate(
  env: ManagerBasedRlEnv, height_command_name: str, min_height: float
) -> torch.Tensor:
  """Float mask that is 1 when the commanded height is >= min_height."""
  height_cmd = env.command_manager.get_command(height_command_name)
  assert height_cmd is not None, f"Command '{height_command_name}' not found."
  return (height_cmd[:, 0] >= float(min_height)).float()


def _points_pos_w(
  asset: Entity, point_ids: torch.Tensor, point_type: str
) -> torch.Tensor:
  """World positions of a set of sites or geoms. Shape (num_envs, N, 3)."""
  if point_type == "site":
    return asset.data.site_pos_w[:, point_ids]
  if point_type == "geom":
    return asset.data.geom_pos_w[:, point_ids]
  raise ValueError(f"Unknown point_type '{point_type}'. Use 'site' or 'geom'.")


def _find_points(asset: Entity, names: tuple[str, ...], point_type: str) -> list[int]:
  if point_type == "site":
    ids, _ = asset.find_sites(names, preserve_order=True)
  elif point_type == "geom":
    ids, _ = asset.find_geoms(names, preserve_order=True)
  else:
    raise ValueError(f"Unknown point_type '{point_type}'. Use 'site' or 'geom'.")
  if len(ids) == 0:
    raise ValueError(f"No {point_type}s matched {names}.")
  return ids


##
# Command tracking.
##


def track_lin_vel_x_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential tracking of the commanded forward velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])
  return torch.exp(-error / std**2)


def track_lin_vel_y_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential tracking of the commanded lateral velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 1] - asset.data.root_link_lin_vel_b[:, 1])
  return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential tracking of the commanded yaw rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def track_relative_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  scale: float = 4.0,
) -> torch.Tensor:
  """Track the commanded base height relative to the support (lowest) foot.

  OpenHomie form: ``exp(-4 * |h_rel - cmd|)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  support_z = torch.min(foot_z, dim=1).values
  actual_height = asset.data.root_link_pos_w[:, 2] - support_z

  error = torch.abs(actual_height - command[:, 0])
  env.extras["log"]["Metrics/relative_height_error_mean"] = torch.mean(error)
  env.extras["log"]["Metrics/relative_height_mean"] = torch.mean(actual_height)
  return torch.exp(-error * float(scale))


##
# Base motion regularization.
##


def lin_vel_z_penalty(
  env: ManagerBasedRlEnv,
  height_command_name: str,
  min_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical base velocity, disabled while squatting."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.square(asset.data.root_link_lin_vel_b[:, 2])
  return cost * _height_gate(env, height_command_name, min_height)


def ang_vel_xy_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize base roll/pitch rates."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_command_name: str | None = None,
  min_height: float = 0.0,
  low_scale: float = 1.0,
) -> torch.Tensor:
  """Penalize base tilt (squared projected-gravity xy).

  With ``height_command_name`` set, the penalty is scaled down to
  ``low_scale`` while the commanded height is below ``min_height``: a deep
  human squat needs anterior pelvis tilt (the chest-to-thigh fold lives at
  the hip, not the +-30 deg waist), and the full ungated penalty pins the
  pelvis upright, capping the whole-torso fold at the waist's range.
  Defaults keep OpenHomie parity (ungated, full weight).
  """
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  if height_command_name is not None:
    gate = _height_gate(env, height_command_name, min_height)
    cost = cost * (float(low_scale) + (1.0 - float(low_scale)) * gate)
  return cost


##
# Joint-space regularization.
##


def joint_deviation_gated(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  height_command_name: str,
  min_height: float,
  pitch_command_name: str | None = None,
  pitch_epsilon: float = 0.05,
) -> torch.Tensor:
  """Squared deviation from the default pose, active only near standing height.

  Used for OpenHomie's ``deviation_hip_joint`` / ``deviation_ankle_joint``.
  With ``pitch_command_name`` set (HOMIE+), the gate additionally requires the
  commanded torso pitch to be near zero — leaning demands hip deviation to
  shift the CoM, which must not be penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(
      asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    ),
    dim=1,
  )
  gate = _height_gate(env, height_command_name, min_height)
  if pitch_command_name is not None:
    pitch_cmd = env.command_manager.get_command(pitch_command_name)
    assert pitch_cmd is not None
    gate = gate * (pitch_cmd[:, 0].abs() < float(pitch_epsilon)).float()
  return error * gate


def track_torso_pitch(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """HOMIE+: exponential tracking of the commanded waist_pitch joint angle."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd_term = env.command_manager.get_term(command_name)
  joint_id = cmd_term.joint_id  # TorsoPitchCommand
  cmd = cmd_term.command
  error = torch.abs(asset.data.joint_pos[:, joint_id] - cmd[:, 0])
  return torch.exp(-error * float(scale))


def knee_deviation(
  env: ManagerBasedRlEnv,
  command_name: str,
  knee_asset_cfg: SceneEntityCfg,
  foot_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Encourage height changes to be driven by the knees.

  For each knee, ``u`` is the joint position normalized by its hard limits; the
  penalty is ``sum_i |height_error * (u_i - 0.5)|``. Bending the knee toward
  mid-range while a height error persists reduces the penalty.
  """
  asset: Entity = env.scene[knee_asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  foot_z = asset.data.site_pos_w[:, foot_asset_cfg.site_ids, 2]
  support_z = torch.min(foot_z, dim=1).values
  height_err = asset.data.root_link_pos_w[:, 2] - support_z - command[:, 0]

  knee_ids = knee_asset_cfg.joint_ids
  knee_pos = asset.data.joint_pos[:, knee_ids]
  limits = asset.data.joint_pos_limits[:, knee_ids]
  denom = torch.clamp(limits[..., 1] - limits[..., 0], min=1e-6)
  u = (knee_pos - limits[..., 0]) / denom
  return torch.sum(torch.abs(height_err.unsqueeze(1) * (u - 0.5)), dim=1)


def action_smoothness_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Second-order action difference penalty (OpenHomie ``smoothness``)."""
  mgr = env.action_manager
  diff = mgr.action - 2.0 * mgr.prev_action + mgr.prev_prev_action
  return torch.sum(torch.square(diff), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_power(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize mechanical power, normalized by the commanded speed."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  power = torch.sum(
    torch.abs(
      asset.data.joint_vel[:, asset_cfg.joint_ids]
      * asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )
  cmd_norm = torch.sum(torch.square(command[:, 0:2]), dim=-1) + 0.2 * torch.square(
    command[:, 2]
  )
  return power / torch.clamp(cmd_norm, min=0.1)


def joint_tracking_error(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize PD position-target tracking error."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  error = asset.data.joint_pos_target[:, joint_ids] - asset.data.joint_pos[:, joint_ids]
  return torch.sum(torch.square(error), dim=1)


class joint_torques_l2_normalized:
  """Penalize squared joint torques normalized by their PD stiffness.

  Matches OpenHomie's ``torques`` term: ``sum((tau / kp)^2)``. Stiffness is
  resolved once from a {joint-name-pattern: kp} mapping.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    _, _, values = resolve_matching_names_values(
      data=cfg.params["stiffness"], list_of_strings=joint_names
    )
    self._kp = torch.tensor(values, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    stiffness: dict[str, float],
  ) -> torch.Tensor:
    del stiffness  # Resolved at init.
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau / self._kp), dim=1)


class joint_vel_limits_cost:
  """Penalize joint velocities beyond a soft fraction of their limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    _, _, values = resolve_matching_names_values(
      data=cfg.params["velocity_limits"], list_of_strings=joint_names
    )
    self._limits = torch.tensor(values, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    velocity_limits: dict[str, float],
    soft_factor: float = 0.8,
  ) -> torch.Tensor:
    del velocity_limits  # Resolved at init.
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    excess = torch.abs(vel) - self._limits * float(soft_factor)
    return torch.sum(torch.clamp(excess, min=0.0), dim=1)


class joint_torque_limits_cost:
  """Penalize joint torques beyond a soft fraction of their limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    _, _, values = resolve_matching_names_values(
      data=cfg.params["effort_limits"], list_of_strings=joint_names
    )
    self._limits = torch.tensor(values, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    effort_limits: dict[str, float],
    soft_factor: float = 0.95,
  ) -> torch.Tensor:
    del effort_limits  # Resolved at init.
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    excess = torch.abs(tau) - self._limits * float(soft_factor)
    return torch.sum(torch.clamp(excess, min=0.0), dim=1)


class action_out_of_bounds:
  """Penalize raw policy actions outside the joint-limit-derived action range.

  OpenHomie's ``action_vanish``: with position targets ``q = default + scale *
  action``, actions beyond ``(q_limit - default) / scale`` saturate at the joint
  limit and produce no motion. The excess is penalized to keep actions in the
  useful range.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._bounds: tuple[torch.Tensor, torch.Tensor] | None = None

  def _resolve_bounds(
    self, env: ManagerBasedRlEnv, action_name: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._bounds is not None:
      return self._bounds
    term = env.action_manager.get_term(action_name)
    entity: Entity = term._entity
    joint_ids = term._target_ids
    limits = entity.data.joint_pos_limits[:, joint_ids]
    default = entity.data.default_joint_pos[:, joint_ids]
    scale = term._scale
    lower = (limits[..., 0] - default) / scale
    upper = (limits[..., 1] - default) / scale
    self._bounds = (lower, upper)
    return self._bounds

  def __call__(self, env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
    lower, upper = self._resolve_bounds(env, action_name)
    raw = env.action_manager.get_term(action_name).raw_action
    excess = torch.clamp(raw - upper, min=0.0) + torch.clamp(lower - raw, min=0.0)
    return torch.sum(excess, dim=-1)


##
# Feet and contacts.
##


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward long steps, evaluated at the landing instant."""
  sensor: ContactSensor = env.scene[sensor_name]
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  valid = first_contact & (last_air_time > float(threshold_min))
  reward = torch.sum((last_air_time - float(threshold_max)) * valid.float(), dim=1)

  num_landings = torch.sum(valid.float())
  env.extras["log"]["Metrics/air_time_mean"] = torch.sum(
    last_air_time * valid.float()
  ) / torch.clamp(num_landings, min=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    active = (torch.norm(command[:, :3], dim=1) > float(command_threshold)).float()
    reward = reward * active
  return reward


def no_fly(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward single support; full reward for near-zero commands."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  # The net-force z sign depends on the contact normal orientation (foot-ground
  # contacts report negative z here); contact presence is a magnitude test.
  contacts = torch.abs(sensor.data.force[:, :, 2]) > float(force_threshold)
  reward = (torch.sum(contacts.float(), dim=1) == 1.0).float()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    is_zero_cmd = (torch.norm(command[:, :3], dim=1) < float(command_threshold)).float()
    reward = torch.max(reward, is_zero_cmd)
  return reward


def feet_load_asymmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = "twist",
  vx_threshold: float = 0.1,
  twist_threshold: float = 0.3,
  force_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward single-sided foot loading for IN-PLACE locomotion commands.

  Bridges the weight-shift exploration valley that kills in-place turning:
  from symmetric standing, lifting a foot without first moving the CoM over
  the stance foot tips the robot, and the weight shift itself changes no
  contact state, so neither ``no_fly`` (+ single support) nor twist tracking
  provides a gradient along it — the v3/v4 policies stand at 100% double
  support through every pure-turn command despite forfeiting ~2.6/step.
  Load asymmetry |Fl - Fr| / (Fl + Fr) climbs SMOOTHLY from 0 (symmetric
  stand) to 1 (single support), from where in-support rotation and stepping
  become reachable by ordinary exploration. Walking bootstraps through
  lean-and-catch and needs no such bridge, so the term is gated to the
  in-place command corner (|vx_cmd| < ``vx_threshold``, sampled |vy|/|wz|
  dominant axis >= ``twist_threshold``) and leaves the rest of the recipe
  untouched.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  fz = torch.abs(sensor.data.force[:, :, 2])
  total = fz.sum(dim=1)
  asym = (fz[:, 0] - fz[:, 1]).abs() / (total + 1.0)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  in_place = (command[:, 0].abs() < float(vx_threshold)) & (
    torch.norm(command[:, 1:3], dim=1) >= float(twist_threshold)
  )
  grounded = total > float(force_threshold)
  return asym * in_place.float() * grounded.float()


def feet_flat(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  twist_command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize foot tilt from horizontal while stationary (kills toe-crouch).

  Measures gravity in each foot body frame — sensitive to BOTH pitch
  (heels-up toe-crouch) and roll (inner-edge contact), unlike the
  collision-point z-variance form, which is nearly blind to either on the
  G1's narrow sphere layout. Gated to twist ~ 0 so swing-leg tilt during
  walking is untouched; ``asset_cfg.body_names`` should match the foot
  (ankle_roll) links.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  n, f = quats.shape[0], quats.shape[1]
  g = torch.tensor((0.0, 0.0, -1.0), device=quats.device).expand(n * f, 3)
  g_foot = quat_apply_inverse(quats.reshape(-1, 4), g).reshape(n, f, 3)
  cost = torch.sum(torch.square(g_foot[..., :2]), dim=(1, 2))

  command = env.command_manager.get_command(twist_command_name)
  assert command is not None
  stationary = torch.norm(command[:, :3], dim=1) < float(command_threshold)
  return cost * stationary.float()


def squat_pose_prior(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  height_command_name: str = "height",
  twist_command_name: str = "twist",
  scale: float = 2.0,
  fade_range: tuple[float, float] = (0.28, 0.62),
  ankle_ref_min: float = -0.78,
) -> torch.Tensor:
  """One-frame reference: reward proximity to the flat-foot deep-squat pose.

  Pure-reward G1 policies fall into toe-crouch / kneeling attractors for
  deep height commands; the works that show human-form deep squats (AMO)
  guide RL with trajectory-optimization reference libraries. This term is
  the minimal version of that idea: a kinematically derived, balanced
  (|com_dx| < 5 mm), flat-foot, knee-clear (0.25 m) squat family, linear in
  the commanded height (scanned 2026-07-10, waist_pitch 0.25):

    hip_pitch(h)   = -1.79 + 2.67 (h - 0.28)
    knee(h)        =  2.64 - 3.36 (h - 0.28)
    ankle_pitch(h) = -0.85 + 0.73 (h - 0.28), clamped >= ``ankle_ref_min``
                     (the raw fit hugs the dorsiflexion soft limit).

  Soft guidance only (exp form, moderate scale): the policy reconciles the
  prior with the torso-pitch command and DR. Fades out linearly above the
  deep range and gates to stationary envs, so walking and standing are
  untouched. ``asset_cfg.joint_names`` must be, in order: left/right
  hip_pitch, left/right knee, left/right ankle_pitch (preserve_order).
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_cmd = env.command_manager.get_command(height_command_name)
  assert height_cmd is not None
  h = height_cmd[:, 0]
  d = h - 0.28
  hip_ref = -1.79 + 2.67 * d
  knee_ref = 2.64 - 3.36 * d
  ankle_ref = torch.clamp(-0.85 + 0.73 * d, min=float(ankle_ref_min))
  refs = torch.stack(
    (hip_ref, hip_ref, knee_ref, knee_ref, ankle_ref, ankle_ref), dim=1
  )
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  err = torch.sum(torch.square(q - refs), dim=1)
  reward = torch.exp(-float(scale) * err)

  lo, hi = fade_range
  fade = torch.clamp((hi - h) / (hi - lo), min=0.0, max=1.0)
  command = env.command_manager.get_command(twist_command_name)
  assert command is not None
  stationary = torch.norm(command[:, :3], dim=1) < 0.1
  return reward * fade * stationary.float()


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_command_name: str,
  min_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize swing-foot height deviating from the target, scaled by foot speed.

  Active only near standing-height commands (as in OpenHomie), so deep squats
  are not forced to lift their feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  height_error = torch.square(foot_z - float(target_height))

  foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]  # (B, N, 3)
  root_vel = asset.data.root_link_lin_vel_w.unsqueeze(1)
  root_quat = asset.data.root_link_quat_w
  rel_vel = foot_vel - root_vel
  n = rel_vel.shape[1]
  quat = root_quat.unsqueeze(1).expand(-1, n, -1).reshape(-1, 4)
  vel_b = quat_apply_inverse(quat, rel_vel.reshape(-1, 3)).view_as(rel_vel)
  lateral_speed = torch.norm(vel_b[:, :, :2], dim=-1)

  cost = torch.sum(height_error * lateral_speed, dim=1)
  return cost * _height_gate(env, height_command_name, min_height)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize horizontal foot velocity while in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  in_contact = (torch.abs(sensor.data.force[:, :, 2]) > float(force_threshold)).float()
  vel_xy = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  return torch.sum(vel_xy * in_contact, dim=1)


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize feet hitting vertical surfaces (large tangential contact force)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force = sensor.data.force
  tangential = torch.norm(force[:, :, :2], dim=-1)
  normal = torch.abs(force[:, :, 2])
  return torch.any(tangential > 3.0 * normal, dim=1).float()


def feet_contact_forces(
  env: ManagerBasedRlEnv, sensor_name: str, max_force: float = 400.0
) -> torch.Tensor:
  """Penalize contact forces above a maximum magnitude."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  magnitude = torch.norm(sensor.data.force, dim=-1)
  return torch.sum(torch.clamp(magnitude - float(max_force), min=0.0), dim=1)


def contact_momentum(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  force_threshold: float = 50.0,
) -> torch.Tensor:
  """Encourage soft contacts: downward foot velocity times excess normal force.

  The product is negative, so this term is used with a *positive* weight
  (OpenHomie ``contact_momentum = 2.5e-4``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  foot_vel_z = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
  down_vel = torch.clamp(foot_vel_z, max=0.0)
  # Normal-force magnitude: the net-force z sign depends on the contact
  # normal orientation (foot-ground contacts report negative z here).
  excess_force = torch.clamp(
    torch.abs(sensor.data.force[:, :, 2]) - float(force_threshold), min=0.0
  )
  return torch.sum(down_vel * excess_force, dim=1)


class feet_ground_parallel:
  """Penalize landing on a tilted sole.

  At the landing instant (after a sufficiently long swing), penalizes the
  variance of the foot's contact-point heights.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    point_type = cfg.params.get("point_type", "site")
    left = _find_points(asset, tuple(cfg.params["left_foot_points"]), point_type)
    right = _find_points(asset, tuple(cfg.params["right_foot_points"]), point_type)
    self._left = torch.tensor(left, device=env.device, dtype=torch.long)
    self._right = torch.tensor(right, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    left_foot_points: tuple[str, ...],
    right_foot_points: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    point_type: str = "site",
    min_air_time_steps: int = 3,
  ) -> torch.Tensor:
    del left_foot_points, right_foot_points  # Resolved at init.
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    last_air_time = sensor.data.last_air_time
    assert last_air_time is not None

    left_var = torch.var(_points_pos_w(asset, self._left, point_type)[..., 2], dim=1)
    right_var = torch.var(_points_pos_w(asset, self._right, point_type)[..., 2], dim=1)

    landing = sensor.compute_first_contact(dt=env.step_dt) & (
      last_air_time >= float(min_air_time_steps) * env.step_dt
    )
    landing = landing.float()
    return left_var * landing[:, 0] + right_var * landing[:, 1]


class feet_parallel:
  """Penalize the two soles not being parallel to each other.

  Uses the variance of distances between corresponding left/right foot points.
  Active only near standing-height commands.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    point_type = cfg.params.get("point_type", "site")
    left = _find_points(asset, tuple(cfg.params["left_foot_points"]), point_type)
    right = _find_points(asset, tuple(cfg.params["right_foot_points"]), point_type)
    if len(left) != len(right):
      raise ValueError(
        "feet_parallel requires the same number of left/right foot points, got "
        f"{len(left)} vs {len(right)}."
      )
    self._left = torch.tensor(left, device=env.device, dtype=torch.long)
    self._right = torch.tensor(right, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_foot_points: tuple[str, ...],
    right_foot_points: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    height_command_name: str,
    min_height: float,
    point_type: str = "site",
  ) -> torch.Tensor:
    del left_foot_points, right_foot_points  # Resolved at init.
    asset: Entity = env.scene[asset_cfg.name]
    left_pos = _points_pos_w(asset, self._left, point_type)
    right_pos = _points_pos_w(asset, self._right, point_type)
    distances = torch.norm(left_pos - right_pos, dim=2)
    variance = torch.var(distances, dim=1)
    return variance * _height_gate(env, height_command_name, min_height)


def feet_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  least_distance: float = 0.2,
  most_distance: float = 0.35,
  height_command_name: str | None = None,
  min_height: float | None = None,
) -> torch.Tensor:
  """Keep the feet laterally separated within a window.

  Returns ``clamp(d - least, max=0) + clamp(most - d, max=0) * gate`` — always
  <= 0, so this term is used with a *positive* weight (OpenHomie ``0.5``). The
  "too far apart" component is gated to standing height so squats may widen the
  stance.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  rel = foot_pos - asset.data.root_link_pos_w.unsqueeze(1)
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
  pos_b = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).view_as(rel)
  distance = torch.abs(pos_b[:, 0, 1] - pos_b[:, 1, 1])

  gate = torch.ones(env.num_envs, device=env.device)
  if height_command_name is not None and min_height is not None:
    gate = _height_gate(env, height_command_name, min_height)

  too_close = torch.clamp(distance - float(least_distance), max=0.0)
  too_far = torch.clamp(float(most_distance) - distance, max=0.0)
  return too_close + too_far * gate


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  least_distance: float = 0.2,
  most_distance: float = 0.35,
  height_command_name: str | None = None,
  min_height: float | None = None,
) -> torch.Tensor:
  """Keep the knees laterally separated within a window.

  ``asset_cfg.body_names`` must resolve to two left/right body *pairs* in the
  order (left_a, left_b, right_a, right_b); the summed pairwise lateral distance
  is compared against ``2 * least`` / ``2 * most``. Always <= 0; use a positive
  weight (OpenHomie ``1.0``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (B, 4, 3)
  if body_pos.shape[1] != 4:
    raise ValueError(
      f"knee_distance_lateral expects 4 bodies (2 left/right pairs), got "
      f"{body_pos.shape[1]}."
    )
  rel = body_pos - asset.data.root_link_pos_w.unsqueeze(1)
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
  pos_b = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).view_as(rel)
  distance = torch.abs(pos_b[:, 0, 1] - pos_b[:, 2, 1]) + torch.abs(
    pos_b[:, 1, 1] - pos_b[:, 3, 1]
  )

  gate = torch.ones(env.num_envs, device=env.device)
  if height_command_name is not None and min_height is not None:
    gate = _height_gate(env, height_command_name, min_height)

  too_close = torch.clamp(distance - 2.0 * float(least_distance), max=0.0)
  too_far = torch.clamp(2.0 * float(most_distance) - distance, max=0.0)
  return too_close + too_far * gate


def stand_still(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.1,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  height_command_name: str | None = None,
  min_height: float | None = None,
) -> torch.Tensor:
  """Penalize lifted feet when commanded to stand still at standing height."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  contact_forces_z = torch.abs(sensor.data.force[:, :, 2])
  num_feet_no_contact = torch.sum(
    (contact_forces_z < float(force_threshold)).float(), dim=1
  )

  gate = torch.ones(env.num_envs, device=env.device)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    gate = gate * (torch.norm(command[:, :3], dim=1) < float(command_threshold)).float()
  if height_command_name is not None and min_height is not None:
    gate = gate * _height_gate(env, height_command_name, min_height)
  return num_feet_no_contact * gate


##
# Contact penalties (replacements for IsaacGym contact-based terminations).
##


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions detected by the given contact sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1).float()


def penalize_body_contacts(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize unwanted body contacts (e.g. hips/knees touching the ground)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.sum(dim=-1).float()
