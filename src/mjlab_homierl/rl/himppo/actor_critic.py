from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from mjlab_homierl.rl.himppo.estimator import HIMEstimator, get_activation


@dataclass(frozen=True)
class HimObsLayout:
  num_one_step_obs: int
  num_one_step_privileged_obs: int
  actor_history_length: int
  critic_history_length: int


class HIMActorCritic(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    obs: Any,
    obs_groups: dict[str, list[str]],
    num_actions: int,
    *,
    layout: HimObsLayout,
    actor_hidden_dims: tuple[int, ...] = (512, 256, 256),
    critic_hidden_dims: tuple[int, ...] = (512, 256, 256),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    dynamic_latent_dim: int = 32,
    terrain_latent_dim: int = 32,
    **_: Any,
  ) -> None:
    super().__init__()

    act = get_activation(activation)
    self.obs_groups = obs_groups

    self.num_actor_obs = int(sum(obs[k].shape[-1] for k in obs_groups["actor"]))
    self.has_critic = "critic" in obs_groups
    self.num_critic_obs = (
      int(sum(obs[k].shape[-1] for k in obs_groups["critic"])) if self.has_critic else 0
    )

    self.num_one_step_obs = int(layout.num_one_step_obs)
    self.num_one_step_critic_obs = (
      int(layout.num_one_step_privileged_obs) if self.has_critic else 0
    )
    self.actor_history_length = int(layout.actor_history_length)
    self.critic_history_length = (
      int(layout.critic_history_length) if self.has_critic else 0
    )
    self.actor_proprioceptive_obs_length = (
      self.actor_history_length * self.num_one_step_obs
    )
    self.critic_proprioceptive_obs_length = (
      self.critic_history_length * self.num_one_step_critic_obs
    )

    self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
    self.actor_use_height = self.num_height_points > 0

    self.num_actions = int(num_actions)
    self.dynamic_latent_dim = int(dynamic_latent_dim)
    self.terrain_latent_dim = int(terrain_latent_dim)

    # Estimator (self-supervised env feature extractor).
    self.estimator = HIMEstimator(
      temporal_steps=self.actor_history_length,
      num_one_step_obs=self.num_one_step_obs,
      num_height_points=0,
      latent_dim=self.dynamic_latent_dim,
    )

    # Optional terrain encoder if height points are present in actor obs.
    if self.actor_use_height:
      self.terrain_encoder = nn.Sequential(
        nn.Linear(self.num_one_step_obs + self.num_height_points, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, self.terrain_latent_dim),
      )
      mlp_input_dim_a = (
        self.num_one_step_obs + 3 + self.dynamic_latent_dim + self.terrain_latent_dim
      )
    else:
      self.terrain_encoder = None
      mlp_input_dim_a = self.num_one_step_obs + 3 + self.dynamic_latent_dim

    # Policy network.
    actor_layers: list[nn.Module] = [
      nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]),
      act,
    ]
    for i, hidden_dim in enumerate(actor_hidden_dims):
      if i == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(hidden_dim, self.num_actions))
      else:
        actor_layers.extend([nn.Linear(hidden_dim, actor_hidden_dims[i + 1]), act])
    self.actor = nn.Sequential(*actor_layers)

    # Value network.
    if self.has_critic:
      critic_layers: list[nn.Module] = [
        nn.Linear(self.num_critic_obs, critic_hidden_dims[0]),
        act,
      ]
      for i, hidden_dim in enumerate(critic_hidden_dims):
        if i == len(critic_hidden_dims) - 1:
          critic_layers.append(nn.Linear(hidden_dim, 1))
        else:
          critic_layers.extend([nn.Linear(hidden_dim, critic_hidden_dims[i + 1]), act])
      self.critic = nn.Sequential(*critic_layers)
    else:
      self.critic = None

    # Action noise.
    self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.num_actions))
    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def action_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def action_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)

  @property
  def output_std(self) -> torch.Tensor:
    if self.distribution is None:
      return self.std
    return self.distribution.stddev

  def forward(self, obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
      return self.act_inference_actor_obs(obs)
    return self.act_inference(obs)

  def get_actor_obs(self, obs: Any) -> torch.Tensor:
    return torch.cat([obs[k] for k in self.obs_groups["actor"]], dim=-1)

  def get_critic_obs(self, obs: Any) -> torch.Tensor:
    if not self.has_critic:
      raise RuntimeError("HIMActorCritic was constructed without critic observations.")
    return torch.cat([obs[k] for k in self.obs_groups["critic"]], dim=-1)

  def _actor_input_from_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      vel, dynamic_latent = self.estimator(
        obs_history[:, : self.actor_proprioceptive_obs_length]
      )
    if self.actor_use_height:
      assert self.terrain_encoder is not None
      terrain_in = obs_history[:, -(self.num_height_points + self.num_one_step_obs) :]
      terrain_latent = self.terrain_encoder(terrain_in)
      last_step = obs_history[
        :, -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points
      ]
      return torch.cat((last_step, vel, dynamic_latent, terrain_latent), dim=-1)
    last_step = obs_history[:, -self.num_one_step_obs :]
    return torch.cat((last_step, vel, dynamic_latent), dim=-1)

  def update_distribution(self, obs_history: torch.Tensor) -> None:
    actor_in = self._actor_input_from_history(obs_history)
    action_mean = self.actor(actor_in)
    self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

  def act(self, obs_history: torch.Tensor) -> torch.Tensor:
    self.update_distribution(obs_history)
    assert self.distribution is not None
    return self.distribution.sample()

  def act_inference_actor_obs(self, obs_history: torch.Tensor) -> torch.Tensor:
    actor_in = self._actor_input_from_history(obs_history)
    return self.actor(actor_in)

  def act_inference(self, obs: Any) -> torch.Tensor:
    obs_history = self.get_actor_obs(obs)
    return self.act_inference_actor_obs(obs_history)

  def evaluate_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
    if self.critic is None:
      raise RuntimeError("HIMActorCritic was constructed without a critic network.")
    return self.critic(critic_obs)

  def evaluate(self, obs: Any) -> torch.Tensor:
    critic_obs = self.get_critic_obs(obs)
    return self.evaluate_critic_obs(critic_obs)

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_estimator(
    self,
    obs_history: torch.Tensor,
    next_critic_obs: torch.Tensor,
    *,
    lr: float | None = None,
  ) -> tuple[float, float]:
    return self.estimator.update(
      obs_history[:, : self.actor_proprioceptive_obs_length],
      next_critic_obs[:, : self.critic_proprioceptive_obs_length],
      lr=lr,
    )
