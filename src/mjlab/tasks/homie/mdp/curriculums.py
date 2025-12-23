from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def assign_env_group_by_fraction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  group_name: str,
  fraction: float,
  complement_group_name: str | None = None,
  method: str = "random",
  seed: int = 0,
  overwrite: bool = False,
) -> dict[str, torch.Tensor]:
  """Assign a fixed subset of environments to a named group.

  This is useful for mixing tasks/curricula across a vectorized environment without
  hard-coding indices. The assignment is persistent across resets unless
  `overwrite=True`.

  Args:
    env: The environment.
    env_ids: Unused (assignment is over all envs). Kept to match curriculum signature.
    group_name: Name of the group to create/update.
    fraction: Fraction of envs to include in the group (0..1).
    complement_group_name: If provided, also sets this group to the logical NOT
      of `group_name`.
    method: Selection method: "random" (seeded) or "first".
    seed: RNG seed used when method="random".
    overwrite: If False and the group already exists, do nothing.
  """
  del env_ids  # Unused; assignment spans all envs for stable partitions.

  if (
    (not overwrite)
    and hasattr(env, "env_group_masks")
    and group_name in env.env_group_masks
  ):
    mask = env.get_env_group_mask(group_name)
    return {
      f"{group_name}_fraction": mask.float().mean(),
      f"{group_name}_count": mask.sum(),
    }

  num_envs = env.num_envs
  frac = float(max(0.0, min(1.0, fraction)))
  count = int(round(num_envs * frac))
  count = max(0, min(num_envs, count))

  if method not in ("random", "first"):
    raise ValueError(f"Unknown method '{method}'. Use 'random' or 'first'.")

  if method == "first":
    selected = torch.arange(count, device=env.device, dtype=torch.long)
  else:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(num_envs, generator=gen)
    selected = perm[:count].to(env.device, dtype=torch.long)

  mask = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
  if count > 0:
    mask[selected] = True

  env.set_env_group_mask(group_name, mask)
  if complement_group_name is not None:
    env.set_env_group_mask(complement_group_name, ~mask)

  return {
    f"{group_name}_fraction": mask.float().mean(),
    f"{group_name}_count": mask.sum(),
  }


def assign_homie_env_groups(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  seed: int = 0,
) -> dict[str, torch.Tensor]:
  """Assign environments to 'squat', 'velocity', and 'standing' groups.

  Partition:
    - [0, 1/3): 'squat' group (~33.3%)
    - [1/3, 1/2]: 'standing' group (~16.7%)
    - (1/2, 1]: 'velocity' group (50.0%)
  """
  del env_ids  # Unused.
  num_envs = env.num_envs
  gen = torch.Generator(device="cpu")
  gen.manual_seed(int(seed))
  set_x = torch.rand(num_envs, 1, generator=gen).to(env.device)

  is_height = set_x < 1 / 3
  is_vel = set_x > 1 / 2
  is_standing = (~is_height) & (~is_vel)

  env.set_env_group_mask("squat", is_height.squeeze(1))
  env.set_env_group_mask("velocity", is_vel.squeeze(1))
  env.set_env_group_mask("standing", is_standing.squeeze(1))

  return {
    "squat_count": is_height.sum(),
    "velocity_count": is_vel.sum(),
    "standing_count": is_standing.sum(),
  }


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class CurriculumScalableAction(Protocol):
  curriculum_ratio: torch.Tensor

  def set_curriculum_ratio(self, ratio: torch.Tensor | float) -> None: ...


def upper_body_action_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  action_name: str,
  reward_name: str = "track_linear_velocity",
  success_threshold: float = 0.8,
  increment: float = 0.05,
  max_ratio: float = 1.0,
  start_step: int = 0,
) -> torch.Tensor:
  """Scale upper-body motion range based on velocity tracking success.

  - Keeps ratio at 0 until ``start_step``.
  - At episode end (called on resets), if the average raw reward across
    ``env_ids`` for ``reward_name`` exceeds ``success_threshold``, increase
    ratio by ``increment`` up to ``max_ratio``. Ratio is global (not per-env).
  - Returns the ratio for logging (scalar tensor).
  """
  action_term = cast(CurriculumScalableAction, env.action_manager.get_term(action_name))
  if not hasattr(action_term, "set_curriculum_ratio"):
    raise ValueError(
      f"Action term '{action_name}' does not support curriculum scaling."
    )

  # Before the start step, hold ratio at zero.
  if env.common_step_counter < start_step:
    action_term.set_curriculum_ratio(0.0)
    return action_term.curriculum_ratio.unsqueeze(0)

  # Episode-average raw reward = accumulated / (episode_len_s * weight)
  reward_manager = env.reward_manager
  if reward_name not in reward_manager._episode_sums:
    return action_term.curriculum_ratio.unsqueeze(0)

  reward_cfg = reward_manager.get_term_cfg(reward_name)
  weight = reward_cfg.weight if reward_cfg.weight != 0 else 1.0
  episode_sum = reward_manager._episode_sums[reward_name][env_ids].mean()
  avg_raw_reward = episode_sum / (env.max_episode_length_s * weight)

  if avg_raw_reward >= success_threshold:
    updated = min(
      (action_term.curriculum_ratio + increment).item(),
      max_ratio,
    )
    action_term.set_curriculum_ratio(updated)

  return action_term.curriculum_ratio.unsqueeze(0)
