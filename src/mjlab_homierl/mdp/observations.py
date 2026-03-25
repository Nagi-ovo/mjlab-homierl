from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.observations import (
  base_lin_vel as _base_lin_vel,
  builtin_sensor as _builtin_sensor,
  generated_commands as _generated_commands,
  joint_pos_rel as _joint_pos_rel,
  joint_vel_rel as _joint_vel_rel,
  last_action as _last_action,
  projected_gravity as _projected_gravity,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def him_actor_one_step_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
  height_command_name: str,
  imu_ang_vel_sensor_name: str,
  joint_asset_cfg: SceneEntityCfg,
  obs_scales: dict[str, float],
) -> torch.Tensor:
  """One-step actor observation for HOMIE HIMPPO.

  Layout (matches legged_gym):
    [cmd_x, cmd_y, cmd_yaw, cmd_height,
     imu_ang_vel(3), projected_gravity(3),
     dof_pos(N), dof_vel(N),
     actions(A)]
  """
  twist = _generated_commands(env, command_name=command_name)
  height = _generated_commands(env, command_name=height_command_name)

  twist_scaled = twist.detach().clone()
  twist_scaled[:, 0:2] *= float(obs_scales["lin_vel"])
  twist_scaled[:, 2] *= float(obs_scales["ang_vel"])
  commands = torch.cat((twist_scaled[:, :3], height[:, 0:1]), dim=-1)

  imu_ang_vel = _builtin_sensor(env, sensor_name=imu_ang_vel_sensor_name) * float(
    obs_scales["ang_vel"]
  )
  gravity = _projected_gravity(env)
  dof_pos = _joint_pos_rel(env, asset_cfg=joint_asset_cfg) * float(obs_scales["dof_pos"])
  dof_vel = _joint_vel_rel(env, asset_cfg=joint_asset_cfg) * float(obs_scales["dof_vel"])
  actions = _last_action(env)
  return torch.cat((commands, imu_ang_vel, gravity, dof_pos, dof_vel, actions), dim=-1)


def him_critic_one_step_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
  height_command_name: str,
  imu_ang_vel_sensor_name: str,
  joint_asset_cfg: SceneEntityCfg,
  obs_scales: dict[str, float],
) -> torch.Tensor:
  """One-step critic observation for HOMIE HIMPPO (actor obs + base_lin_vel)."""
  actor_obs = him_actor_one_step_obs(
    env,
    command_name=command_name,
    height_command_name=height_command_name,
    imu_ang_vel_sensor_name=imu_ang_vel_sensor_name,
    joint_asset_cfg=joint_asset_cfg,
    obs_scales=obs_scales,
  )
  base_lin_vel = _base_lin_vel(env, asset_cfg=joint_asset_cfg) * float(obs_scales["lin_vel"])
  return torch.cat((actor_obs, base_lin_vel), dim=-1)
