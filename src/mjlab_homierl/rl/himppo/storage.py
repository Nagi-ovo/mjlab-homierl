from __future__ import annotations

from dataclasses import dataclass

import torch


class HIMRolloutStorage:
  @dataclass
  class Transition:
    observations: torch.Tensor | None = None
    critic_observations: torch.Tensor | None = None
    next_critic_observations: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    values: torch.Tensor | None = None
    actions_log_prob: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_sigma: torch.Tensor | None = None

    def clear(self) -> None:
      self.observations = None
      self.critic_observations = None
      self.next_critic_observations = None
      self.actions = None
      self.rewards = None
      self.dones = None
      self.values = None
      self.actions_log_prob = None
      self.action_mean = None
      self.action_sigma = None

  def __init__(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    obs_shape: tuple[int, ...],
    privileged_obs_shape: tuple[int, ...] | None,
    actions_shape: tuple[int, ...],
    device: torch.device,
    transitions_per_step: int = 2,
  ) -> None:
    self.device = device

    # With symmetry augmentation (transitions_per_step=2), each env step stores
    # the original and the mirrored transition.
    self.transitions_per_step = int(transitions_per_step)
    self.num_transitions_per_env = (
      int(num_transitions_per_env) * self.transitions_per_step
    )
    self.num_envs = int(num_envs)

    self.observations = torch.zeros(
      self.num_transitions_per_env, self.num_envs, *obs_shape, device=self.device
    )
    if privileged_obs_shape is not None:
      self.privileged_observations = torch.zeros(
        self.num_transitions_per_env,
        self.num_envs,
        *privileged_obs_shape,
        device=self.device,
      )
      self.next_privileged_observations = torch.zeros(
        self.num_transitions_per_env,
        self.num_envs,
        *privileged_obs_shape,
        device=self.device,
      )
    else:
      self.privileged_observations = None
      self.next_privileged_observations = None

    self.rewards = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )
    self.actions = torch.zeros(
      self.num_transitions_per_env, self.num_envs, *actions_shape, device=self.device
    )
    self.dones = torch.zeros(
      self.num_transitions_per_env,
      self.num_envs,
      1,
      device=self.device,
      dtype=torch.uint8,
    )

    self.actions_log_prob = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )
    self.values = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )
    self.returns = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )
    self.advantages = torch.zeros(
      self.num_transitions_per_env, self.num_envs, 1, device=self.device
    )
    self.mu = torch.zeros(
      self.num_transitions_per_env, self.num_envs, *actions_shape, device=self.device
    )
    self.sigma = torch.zeros(
      self.num_transitions_per_env, self.num_envs, *actions_shape, device=self.device
    )

    self.step = 0

  def add_transitions(self, transition: Transition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise AssertionError("Rollout buffer overflow")

    assert transition.observations is not None
    assert transition.actions is not None
    assert transition.rewards is not None
    assert transition.dones is not None
    assert transition.values is not None
    assert transition.actions_log_prob is not None
    assert transition.action_mean is not None
    assert transition.action_sigma is not None

    self.observations[self.step].copy_(transition.observations)

    if self.privileged_observations is not None:
      assert transition.critic_observations is not None
      self.privileged_observations[self.step].copy_(transition.critic_observations)
    if self.next_privileged_observations is not None:
      assert transition.next_critic_observations is not None
      self.next_privileged_observations[self.step].copy_(
        transition.next_critic_observations
      )

    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    self.mu[self.step].copy_(transition.action_mean)
    self.sigma[self.step].copy_(transition.action_sigma)
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def compute_returns(
    self, last_values: torch.Tensor, gamma: float, lam: float
  ) -> None:
    num_steps = self.num_transitions_per_env // self.transitions_per_step

    def _resize(x: torch.Tensor) -> torch.Tensor:
      return x.view(num_steps, self.transitions_per_step, -1, 1)

    advantage = 0.0
    for step in reversed(range(num_steps)):
      if step == num_steps - 1:
        next_values = last_values
      else:
        next_values = _resize(self.values)[step + 1]
      next_is_not_terminal = 1.0 - _resize(self.dones)[step].float()
      delta = (
        _resize(self.rewards)[step]
        + next_is_not_terminal * float(gamma) * next_values
        - _resize(self.values)[step]
      )
      advantage = delta + next_is_not_terminal * float(gamma) * float(lam) * advantage
      _resize(self.returns)[step] = advantage + _resize(self.values)[step]

    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (
      self.advantages.std() + 1e-8
    )

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // int(num_mini_batches)
    indices = torch.randperm(
      int(num_mini_batches) * mini_batch_size, requires_grad=False, device=self.device
    )

    observations = self.observations.flatten(0, 1)
    if self.privileged_observations is not None:
      assert self.next_privileged_observations is not None
      critic_observations = self.privileged_observations.flatten(0, 1)
      next_critic_observations = self.next_privileged_observations.flatten(0, 1)
    else:
      critic_observations = observations
      next_critic_observations = observations

    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_mu = self.mu.flatten(0, 1)
    old_sigma = self.sigma.flatten(0, 1)

    for _ in range(int(num_epochs)):
      for i in range(int(num_mini_batches)):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        batch_idx = indices[start:end]

        yield (
          observations[batch_idx],
          critic_observations[batch_idx],
          actions[batch_idx],
          next_critic_observations[batch_idx],
          values[batch_idx],
          advantages[batch_idx],
          returns[batch_idx],
          old_actions_log_prob[batch_idx],
          old_mu[batch_idx],
          old_sigma[batch_idx],
        )
