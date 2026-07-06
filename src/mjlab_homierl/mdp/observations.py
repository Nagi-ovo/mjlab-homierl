"""HOMIE HIM observation terms.

The one-step observation layout matches OpenHomie (legged_gym):
  [cmd_x, cmd_y, cmd_yaw, cmd_height,
   base_ang_vel(3), projected_gravity(3),
   joint_pos_rel(N), joint_vel(N),
   prev_actions(A)]

Angular velocity and projected gravity are both expressed in the root (pelvis)
frame, matching OpenHomie's pelvis-mounted IMU and keeping the two terms
frame-consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.observations import last_action as _last_action
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def him_actor_one_step_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
  height_command_name: str,
  joint_asset_cfg: SceneEntityCfg,
  obs_scales: dict[str, float],
  pitch_command_name: str | None = None,
) -> torch.Tensor:
  """One-step actor observation for HOMIE HIM-PPO.

  With ``pitch_command_name`` set (HOMIE+), the command segment grows from
  (vx, vy, wz, h) to (vx, vy, wz, h, pitch) — 4 to 5 dims, unscaled pitch.
  """
  asset: Entity = env.scene[joint_asset_cfg.name]

  twist = env.command_manager.get_command(command_name)
  height = env.command_manager.get_command(height_command_name)
  assert twist is not None and height is not None

  twist_scaled = twist.detach().clone()
  twist_scaled[:, 0:2] *= float(obs_scales["lin_vel"])
  twist_scaled[:, 2] *= float(obs_scales["ang_vel"])
  parts = [twist_scaled[:, :3], height[:, 0:1]]
  if pitch_command_name is not None:
    pitch = env.command_manager.get_command(pitch_command_name)
    assert pitch is not None
    parts.append(pitch[:, 0:1])
  commands = torch.cat(parts, dim=-1)

  ang_vel = asset.data.root_link_ang_vel_b * float(obs_scales["ang_vel"])
  gravity = asset.data.projected_gravity_b

  joint_ids = joint_asset_cfg.joint_ids
  joint_pos = (
    asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
  ) * float(obs_scales["dof_pos"])
  joint_vel = asset.data.joint_vel[:, joint_ids] * float(obs_scales["dof_vel"])

  actions = _last_action(env)
  return torch.cat((commands, ang_vel, gravity, joint_pos, joint_vel, actions), dim=-1)


def him_critic_one_step_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
  height_command_name: str,
  joint_asset_cfg: SceneEntityCfg,
  obs_scales: dict[str, float],
  pitch_command_name: str | None = None,
) -> torch.Tensor:
  """One-step critic observation: actor observation + base linear velocity."""
  actor_obs = him_actor_one_step_obs(
    env,
    command_name=command_name,
    height_command_name=height_command_name,
    joint_asset_cfg=joint_asset_cfg,
    obs_scales=obs_scales,
    pitch_command_name=pitch_command_name,
  )
  asset: Entity = env.scene[joint_asset_cfg.name]
  base_lin_vel = asset.data.root_link_lin_vel_b * float(obs_scales["lin_vel"])
  return torch.cat((actor_obs, base_lin_vel), dim=-1)
