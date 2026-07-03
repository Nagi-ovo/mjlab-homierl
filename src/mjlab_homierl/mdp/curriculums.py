"""HOMIE curriculum terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab_homierl.mdp.actions import UpperBodyPoseAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def upper_body_action_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  action_name: str,
  reward_name: str,
  success_threshold: float = 0.8,
  increment: float = 0.05,
  max_ratio: float = 1.0,
  start_step: int = 0,
) -> dict[str, torch.Tensor]:
  """Grow the upper-body motion range as velocity tracking improves.

  Mirrors OpenHomie's ``update_action_curriculum``: the check runs only when
  ``common_step_counter`` is a multiple of the max episode length (~once per
  episode duration), using the environments being reset at that moment. The
  episode-average unweighted tracking reward is normalized by the *max* episode
  length, so early terminations count against advancement.
  """
  action_term = cast(UpperBodyPoseAction, env.action_manager.get_term(action_name))

  if env.common_step_counter < start_step:
    action_term.set_curriculum_ratio(0.0)
    return {"ratio": action_term.curriculum_ratio.unsqueeze(0)}

  max_episode_steps = int(round(env.max_episode_length))
  if env.common_step_counter % max_episode_steps != 0:
    return {"ratio": action_term.curriculum_ratio.unsqueeze(0)}

  reward_manager = env.reward_manager
  if reward_name not in reward_manager._episode_sums:
    return {"ratio": action_term.curriculum_ratio.unsqueeze(0)}

  weight = reward_manager.get_term_cfg(reward_name).weight
  if weight == 0:
    return {"ratio": action_term.curriculum_ratio.unsqueeze(0)}

  episode_sums = reward_manager._episode_sums[reward_name][env_ids]
  denom = max_episode_steps * float(env.step_dt) * weight
  avg_raw_reward = torch.mean(episode_sums) / denom

  if avg_raw_reward >= success_threshold:
    updated = min(float(action_term.curriculum_ratio) + increment, max_ratio)
    action_term.set_curriculum_ratio(updated)

  return {
    "ratio": action_term.curriculum_ratio.unsqueeze(0),
    "avg_raw_reward": avg_raw_reward,
  }
