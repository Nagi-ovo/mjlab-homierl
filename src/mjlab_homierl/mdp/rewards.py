from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _env_group_scale(
  env: ManagerBasedRlEnv, env_group: str | list[str] | None
) -> torch.Tensor:
  """Return a float mask (num_envs,) scaling a term by env group membership."""
  if env_group is None:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

  if isinstance(env_group, str):
    env_group = [env_group]

  mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for group in env_group:
    try:
      mask |= env.get_env_group_mask(group)
    except Exception:
      # If group not found, we assume the mask is all zeros for that group
      pass
  return mask.float()


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  env_group: str | None = None,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  reward = torch.exp(-lin_vel_error / std**2)
  return reward * _env_group_scale(env, env_group)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  env_group: str | None = None,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  reward = torch.exp(-ang_vel_error / std**2)
  return reward * _env_group_scale(env, env_group)


def track_relative_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  env_group: str | None = None,
) -> torch.Tensor:
  """Reward for tracking commanded base height relative to the support foot.

  The tracked quantity is:
    h_rel = base_z - min_i(foot_site_z[i])
  where the support foot is approximated as the lowest foot site.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  if foot_z.shape[1] == 0:
    raise ValueError(
      "track_relative_height requires at least one foot site in asset_cfg.site_names."
    )
  support_z = torch.min(foot_z, dim=1).values

  base_z = asset.data.root_link_pos_w[:, 2]
  actual_height = base_z - support_z
  target_height = command[:, 0]

  error = actual_height - target_height
  env.extras["log"]["Metrics/relative_height_error_mean"] = torch.mean(torch.abs(error))
  env.extras["log"]["Metrics/relative_height_mean"] = torch.mean(actual_height)
  reward = torch.exp(-torch.square(error) / std**2)
  return reward * _env_group_scale(env, env_group)


def knee_deviation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  knee_asset_cfg: SceneEntityCfg,
  foot_asset_cfg: SceneEntityCfg,
  env_group: str | None = None,
) -> torch.Tensor:
  """Encourage using knees to control height by penalizing knee deviation when height error is present.

  Implements:
    height_err = h_r - h_t
    u = (q_knee - q_min) / (q_max - q_min)
    dev = u - 0.5
    penalty = sum_i |height_err * dev_i|
  Returns the unweighted penalty (positive). Use a negative weight in RewardTermCfg.
  """
  asset: Entity = env.scene[knee_asset_cfg.name]

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  foot_z = asset.data.site_pos_w[:, foot_asset_cfg.site_ids, 2]
  if foot_z.shape[1] == 0:
    raise ValueError(
      "knee_deviation_reward requires at least one foot site in foot_asset_cfg.site_names."
    )
  support_z = torch.min(foot_z, dim=1).values

  base_z = asset.data.root_link_pos_w[:, 2]
  actual_height = base_z - support_z
  target_height = command[:, 0]
  height_err = actual_height - target_height  # signed

  knee_ids = knee_asset_cfg.joint_ids
  # Resolve to a sized tensor to validate we actually selected something.
  knee_pos = asset.data.joint_pos[:, knee_ids]
  if knee_pos.shape[1] == 0:
    raise ValueError(
      "knee_deviation_reward requires at least one knee joint in knee_asset_cfg.joint_names."
    )

  limits = asset.data.soft_joint_pos_limits[:, knee_ids, :]  # [B, K, 2]
  qmin = limits[..., 0]
  qmax = limits[..., 1]
  denom = torch.clamp(qmax - qmin, min=1e-6)
  u = (knee_pos - qmin) / denom
  u = torch.clamp(u, 0.0, 1.0)
  dev = u - 0.5

  penalty = torch.sum(torch.abs(height_err.unsqueeze(1) * dev), dim=1)
  env.extras["log"]["Metrics/knee_height_penalty_mean"] = torch.mean(penalty)
  return penalty * _env_group_scale(env, env_group)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation (LeggedGym style).

  Returns the sum of squared projected gravity XY components.
  Use with a negative weight to penalize tilting.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Always use root link projected gravity for simplicity.
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def penalize_body_contacts(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize unwanted body contacts (e.g., hip/knee touching ground).

  Returns the total number of contacts detected by the specified contact sensor.
  Each contact adds to the penalty. Use with a negative weight.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Sum over all monitored bodies to get total contact count
  return sensor.data.found.sum(dim=-1).float()

def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time on landing (LeggedGym-style).

  This matches the reference legged_gym implementation:
    - Reward is provided only at the instant a foot establishes contact
      (first_contact), not continuously while the foot is in the air.
    - The reward is proportional to (air_time - threshold_max), encouraging
      longer steps (air_time > threshold_max) and discouraging very short hops.

  Args:
    threshold_min: Minimum air time required to count as a valid step/landing.
      Helps ignore contact jitter and initial contacts after reset.
    threshold_max: Baseline air time subtracted from the last air time at
      landing. In the reference implementation this baseline is 0.5 seconds.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  last_air_time = sensor_data.last_air_time
  assert last_air_time is not None

  first_contact = sensor.compute_first_contact(dt=env.step_dt)  # [B, N_feet]
  valid = first_contact & (last_air_time > float(threshold_min))

  reward = torch.sum((last_air_time - float(threshold_max)) * valid.float(), dim=1)

  num_landings = torch.sum(valid.float())
  mean_air_time = torch.sum(last_air_time * valid.float()) / torch.clamp(
    num_landings, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
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
  """Reward having at least one foot in contact (biped "no-flight" heuristic).

  This is aligned with the reference legged_gym term named "no_fly":
    - Returns 1 when exactly one foot is in contact (single support).
    - Returns 1 for near-zero commanded motion to avoid penalizing standing/squat
      modes where both feet should remain in contact.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  contacts = sensor.data.force[:, :, 2] > float(force_threshold)  # [B, N_feet]
  single_contact = (torch.sum(contacts.float(), dim=1) == 1.0).float()
  reward = single_contact

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      is_zero_cmd = (torch.norm(command[:, :3], dim=1) < float(command_threshold)).float()
      reward = torch.max(reward, is_zero_cmd)

  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None

  # Contact detection based on force threshold
  contact_forces_z = torch.abs(contact_sensor.data.force[:, :, 2])  # [B, N_feet]
  in_contact = (contact_forces_z > force_threshold).float()  # [B, N]

  # Foot xy velocity
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]

  return torch.sum(vel_xy_norm * in_contact, dim=1)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_ground_parallel(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  left_foot_sites: tuple[str, ...],
  right_foot_sites: tuple[str, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize tilted foot soles (deviation from flat contact).

  For each foot, computes the variance of heights across multiple sites.
  Higher variance indicates the foot is tilted; zero variance means flat.
  Only penalizes feet that are in continuous contact with the ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  # Get site IDs for left and right foot
  left_site_ids, _ = asset.find_sites(left_foot_sites, preserve_order=True)
  right_site_ids, _ = asset.find_sites(right_foot_sites, preserve_order=True)

  if len(left_site_ids) == 0 or len(right_site_ids) == 0:
    # If no sites found, return zero penalty
    return torch.zeros(env.num_envs, device=env.device)

  left_site_ids_tensor = torch.tensor(
    left_site_ids, device=env.device, dtype=torch.long
  )
  right_site_ids_tensor = torch.tensor(
    right_site_ids, device=env.device, dtype=torch.long
  )

  # Get heights of all sites (z-coordinate)
  left_heights = asset.data.site_pos_w[:, left_site_ids_tensor, 2]  # [B, N_left]
  right_heights = asset.data.site_pos_w[:, right_site_ids_tensor, 2]  # [B, N_right]

  # Compute variance for each foot
  left_var = (
    torch.var(left_heights, dim=1)
    if left_heights.shape[1] > 1
    else torch.zeros(env.num_envs, device=env.device)
  )
  right_var = (
    torch.var(right_heights, dim=1)
    if right_heights.shape[1] > 1
    else torch.zeros(env.num_envs, device=env.device)
  )

  # Get continuous contact mask (feet that have been in contact for at least 3*dt)
  assert contact_sensor.data.current_contact_time is not None
  assert contact_sensor.data.found is not None
  contact_time = contact_sensor.data.current_contact_time  # [B, 2] (left, right)
  in_contact = (contact_sensor.data.found > 0).float()  # [B, 2]
  continuous_contact = (contact_time >= 3 * env.step_dt).float() * in_contact  # [B, 2]

  # Apply mask: only penalize variance when foot is in continuous contact
  cost = left_var * continuous_contact[:, 0] + right_var * continuous_contact[:, 1]

  return cost


def feet_parallel(
  env: ManagerBasedRlEnv,
  left_foot_sites: tuple[str, ...],
  right_foot_sites: tuple[str, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_command_name: str | None = None,
  min_height: float | None = None,
  env_group: str | list[str] | None = None,
) -> torch.Tensor:
  """Penalize inconsistent left-right foot positions (gait asymmetry).

  Computes pairwise distances between corresponding left/right foot sites,
  then penalizes variance in these distances. High variance indicates one
  foot is ahead/behind the other (asymmetric gait).

  In the reference HOMIE implementation (legged_gym), this term is typically
  enabled only for near-standing height commands. Use (height_command_name,
  min_height) to reproduce this gating without relying on env_group masks.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get site IDs for left and right foot
  left_site_ids, _ = asset.find_sites(left_foot_sites, preserve_order=True)
  right_site_ids, _ = asset.find_sites(right_foot_sites, preserve_order=True)

  if len(left_site_ids) == 0 or len(right_site_ids) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  # Ensure both feet have same number of sites
  num_sites = min(len(left_site_ids), len(right_site_ids))
  left_site_ids_tensor = torch.tensor(
    left_site_ids[:num_sites], device=env.device, dtype=torch.long
  )
  right_site_ids_tensor = torch.tensor(
    right_site_ids[:num_sites], device=env.device, dtype=torch.long
  )

  # Get 3D positions of all sites
  left_pos = asset.data.site_pos_w[:, left_site_ids_tensor, :]  # [B, N, 3]
  right_pos = asset.data.site_pos_w[:, right_site_ids_tensor, :]  # [B, N, 3]

  # Compute pairwise distances between corresponding left/right sites
  feet_distances = torch.norm(left_pos - right_pos, dim=2)  # [B, N]

  # Compute variance of these distances
  feet_distances_var = (
    torch.var(feet_distances, dim=1)
    if num_sites > 1
    else torch.zeros(env.num_envs, device=env.device)
  )

  if height_command_name is not None and min_height is not None:
    height_cmd = env.command_manager.get_command(height_command_name)
    if height_cmd is None:
      raise ValueError(f"Height command '{height_command_name}' not found.")
    height_mask = (height_cmd[:, 0] >= float(min_height)).float()
    feet_distances_var = feet_distances_var * height_mask

  return feet_distances_var * _env_group_scale(env, env_group)


def deviation_ankle_joint(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  height_command_name: str | None = None,
  min_height: float | None = None,
  env_group: str | list[str] | None = None,
) -> torch.Tensor:
  """Penalize ankle joint deviation from default pose.

  Encourages stable ankle positioning when standing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  current_pos = asset.data.joint_pos[:, joint_ids]
  default_pos = asset.data.default_joint_pos[:, joint_ids]
  error = torch.sum(torch.square(current_pos - default_pos), dim=1)

  if height_command_name is not None and min_height is not None:
    height_cmd = env.command_manager.get_command(height_command_name)
    if height_cmd is None:
      raise ValueError(f"Height command '{height_command_name}' not found.")
    error = error * (height_cmd[:, 0] >= float(min_height)).float()

  return error * _env_group_scale(env, env_group)


def deviation_hip_joint(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  height_command_name: str | None = None,
  min_height: float | None = None,
  env_group: str | list[str] | None = None,
) -> torch.Tensor:
  """Penalize hip joint deviation from default pose.

  Encourages stable hip positioning when standing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  current_pos = asset.data.joint_pos[:, joint_ids]
  default_pos = asset.data.default_joint_pos[:, joint_ids]
  error = torch.sum(torch.square(current_pos - default_pos), dim=1)

  if height_command_name is not None and min_height is not None:
    height_cmd = env.command_manager.get_command(height_command_name)
    if height_cmd is None:
      raise ValueError(f"Height command '{height_command_name}' not found.")
    error = error * (height_cmd[:, 0] >= float(min_height)).float()

  return error * _env_group_scale(env, env_group)


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float = 0.1,
  max_distance: float = 0.4,
  height_command_name: str | None = None,
  min_height: float | None = None,
  env_group: str | list[str] | None = None,
) -> torch.Tensor:
  """Reward for maintaining proper lateral knee distance.

  Penalizes knees being too close (collision risk) or too far apart (unstable).
  Returns positive reward when knees are within acceptable range.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get knee body positions in world frame
  knee_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]

  # Transform to body frame
  root_quat = asset.data.root_link_quat_w  # [B, 4]
  root_pos = asset.data.root_link_pos_w  # [B, 3]

  # Translate knee positions relative to root
  knee_pos_translated = knee_pos_w - root_pos.unsqueeze(1)  # [B, 2, 3]

  # Rotate to body frame
  knee_pos_b = torch.zeros_like(knee_pos_translated)
  for i in range(knee_pos_translated.shape[1]):
    knee_pos_b[:, i, :] = quat_apply_inverse(root_quat, knee_pos_translated[:, i, :])

  # Compute lateral distance (y-axis in body frame)
  # Assuming body_ids are [left_knee, right_knee]
  knee_lateral_distance = torch.abs(knee_pos_b[:, 0, 1] - knee_pos_b[:, 1, 1])

  # Penalize if too close or too far
  too_close_penalty = torch.clamp(min_distance - knee_lateral_distance, min=0)
  too_far_penalty = torch.clamp(knee_lateral_distance - max_distance, min=0)

  # In the reference implementation, "too far" is mostly a concern at near-standing
  # heights. Gate this component by the height command to avoid constraining deep
  # squat behaviors.
  if height_command_name is not None and min_height is not None:
    height_cmd = env.command_manager.get_command(height_command_name)
    if height_cmd is None:
      raise ValueError(f"Height command '{height_command_name}' not found.")
    too_far_penalty = too_far_penalty * (height_cmd[:, 0] >= float(min_height)).float()

  # Positive reward when in range, negative when out of range
  reward = 1.0 - too_close_penalty - too_far_penalty

  return reward * _env_group_scale(env, env_group)


def stand_still(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.1,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  height_command_name: str | None = None,
  min_height: float | None = None,
  env_group: str | list[str] | None = None,
) -> torch.Tensor:
  """Penalize feet not being in contact when the robot should be stationary.

  This reward encourages the robot to keep both feet firmly planted when the
  commanded motion is near zero. Optionally gate it by a height command to
  focus on near-standing behaviors (as in the reference legged_gym HOMIE setup).

  Returns the number of feet without proper ground contact (0-2), which should
  be used with a negative weight to penalize lifting feet.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None, (
    f"Contact sensor '{sensor_name}' must have 'force' field enabled."
  )

  # Get contact forces (z-component)
  contact_forces_z = torch.abs(contact_sensor.data.force[:, :, 2])  # [B, N_feet]

  # Count feet without proper contact (force_z < threshold)
  no_contact = (contact_forces_z < force_threshold).float()  # [B, N_feet]
  num_feet_no_contact = torch.sum(no_contact, dim=1)  # [B]

  gate = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is None:
      raise ValueError(f"Command '{command_name}' not found.")
    gate = gate * (torch.norm(command[:, :3], dim=1) < float(command_threshold)).float()

  if height_command_name is not None and min_height is not None:
    height_cmd = env.command_manager.get_command(height_command_name)
    if height_cmd is None:
      raise ValueError(f"Height command '{height_command_name}' not found.")
    gate = gate * (height_cmd[:, 0] >= float(min_height)).float()

  return num_feet_no_contact * gate * _env_group_scale(env, env_group)


class variable_posture:
  """Penalize deviation from default pose, with tighter constraints when standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    env_group: str | None = None,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    reward = torch.exp(-torch.mean(error_squared / (std**2), dim=1))
    return reward * _env_group_scale(env, env_group)
