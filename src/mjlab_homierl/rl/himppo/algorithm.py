"""HIM-PPO: PPO with a self-supervised HIM estimator and symmetry augmentation.

Ported from OpenHomie's rsl_rl fork. Left/right mirror maps are derived from
joint names, so the algorithm works with any humanoid whose paired joints are
named ``left_*`` / ``right_*``.

Known deviation from OpenHomie: the estimator's next-step critic observation at
termination steps is the post-reset observation (mjlab computes observations
after resetting terminated environments). OpenHomie substitutes the pre-reset
observation there; this affects only the ~1/episode done transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.utils import resolve_obs_groups

from mjlab_homierl.rl.himppo.actor_critic import HIMActorCritic, HimObsLayout
from mjlab_homierl.rl.himppo.storage import HIMRolloutStorage

# Joints that flip sign under left/right mirroring but whose names carry no
# "yaw"/"roll" suffix (H1's single torso joint rotates about z).
_EXTRA_SIGN_FLIP_JOINTS = frozenset({"torso"})


@dataclass(frozen=True)
class _MirrorMaps:
  actor_index: torch.Tensor
  actor_sign: torch.Tensor
  critic_index: torch.Tensor
  critic_sign: torch.Tensor
  action_index: torch.Tensor
  action_sign: torch.Tensor


def _joint_mirror(
  names: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  """Left/right permutation and sign flips for an ordered joint list."""
  src = []
  for name in names:
    if "left" in name:
      partner = name.replace("left", "right")
    elif "right" in name:
      partner = name.replace("right", "left")
    else:
      partner = name
    if partner not in names:
      raise ValueError(f"Mirror partner '{partner}' of joint '{name}' not found.")
    src.append(list(names).index(partner))
  sign = [
    -1.0 if ("yaw" in n or "roll" in n or n in _EXTRA_SIGN_FLIP_JOINTS) else 1.0
    for n in names
  ]
  return (
    torch.tensor(src, device=device, dtype=torch.long),
    torch.tensor(sign, device=device, dtype=torch.float32),
  )


def _build_mirror_maps(
  obs_joint_names: Sequence[str],
  action_joint_names: Sequence[str],
  num_one_step_obs: int,
  num_one_step_critic_obs: int,
  device: torch.device,
) -> _MirrorMaps:
  """Mirror maps for the HIM one-step observation layout.

  Layout: [commands(4), ang_vel(3), gravity(3), dof_pos(N), dof_vel(N),
  actions(A)] with an optional base_lin_vel(3) tail on the critic.
  """
  num_dofs = len(obs_joint_names)
  num_actions = len(action_joint_names)
  expected = 10 + 2 * num_dofs + num_actions
  if num_one_step_obs != expected:
    raise ValueError(
      f"One-step obs dim {num_one_step_obs} does not match layout "
      f"10 + 2*{num_dofs} + {num_actions} = {expected}."
    )
  if num_one_step_critic_obs not in (expected, expected + 3):
    raise ValueError(
      f"One-step critic obs dim {num_one_step_critic_obs} must be {expected} "
      f"or {expected + 3} (with base_lin_vel)."
    )

  dof_map, dof_sign = _joint_mirror(obs_joint_names, device)
  act_map, act_sign = _joint_mirror(action_joint_names, device)

  def make_obs_maps(one_step_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(one_step_dim, device=device, dtype=torch.long)
    sign = torch.ones(one_step_dim, device=device, dtype=torch.float32)

    # Commands: [x, y, yaw, height] -> y and yaw flip.
    sign[1] = -1.0
    sign[2] = -1.0
    # Angular velocity: roll and yaw rates flip.
    sign[4] = -1.0
    sign[6] = -1.0
    # Projected gravity: y flips.
    sign[8] = -1.0

    dof_pos_start = 10
    dof_vel_start = 10 + num_dofs
    src[dof_pos_start : dof_pos_start + num_dofs] = dof_pos_start + dof_map
    sign[dof_pos_start : dof_pos_start + num_dofs] = dof_sign
    src[dof_vel_start : dof_vel_start + num_dofs] = dof_vel_start + dof_map
    sign[dof_vel_start : dof_vel_start + num_dofs] = dof_sign

    act_start = 10 + 2 * num_dofs
    src[act_start : act_start + num_actions] = act_start + act_map
    sign[act_start : act_start + num_actions] = act_sign

    if one_step_dim == expected + 3:
      sign[expected + 1] = -1.0  # base_lin_vel y flips.
    return src, sign

  actor_index, actor_sign = make_obs_maps(num_one_step_obs)
  critic_index, critic_sign = make_obs_maps(num_one_step_critic_obs)
  return _MirrorMaps(
    actor_index=actor_index,
    actor_sign=actor_sign,
    critic_index=critic_index,
    critic_sign=critic_sign,
    action_index=act_map,
    action_sign=act_sign,
  )


class HIMPPO:
  policy: HIMActorCritic

  def __init__(
    self,
    policy: HIMActorCritic,
    *,
    use_flip: bool = True,
    mirror_obs_joint_names: Sequence[str] | None = None,
    mirror_action_joint_names: Sequence[str] | None = None,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float | None = 0.01,
    device: str | torch.device = "cpu",
    symmetry_scale: float = 1.0,
    rnd_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    self.device = torch.device(device)
    self.use_flip = bool(use_flip)
    if rnd_cfg is not None:
      raise ValueError("HIMPPO does not support rnd_cfg.")
    if multi_gpu_cfg is not None:
      raise NotImplementedError("HIMPPO does not support multi-GPU training.")

    self.desired_kl = desired_kl
    self.schedule = str(schedule)
    self.learning_rate = float(learning_rate)

    self.policy = policy.to(self.device)
    self.storage: HIMRolloutStorage | None = None
    self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
    self.transition = HIMRolloutStorage.Transition()
    self.transition_sym = HIMRolloutStorage.Transition()

    self.symmetry_scale = float(symmetry_scale)
    self.clip_param = float(clip_param)
    self.num_learning_epochs = int(num_learning_epochs)
    self.num_mini_batches = int(num_mini_batches)
    self.value_loss_coef = float(value_loss_coef)
    self.entropy_coef = float(entropy_coef)
    self.gamma = float(gamma)
    self.lam = float(lam)
    self.max_grad_norm = float(max_grad_norm)
    self.use_clipped_value_loss = bool(use_clipped_value_loss)

    self._mirror_maps: _MirrorMaps | None = None
    if self.use_flip:
      if mirror_obs_joint_names is None or mirror_action_joint_names is None:
        raise ValueError(
          "use_flip=True requires mirror_obs_joint_names and mirror_action_joint_names."
        )
      self._mirror_maps = _build_mirror_maps(
        mirror_obs_joint_names,
        mirror_action_joint_names,
        policy.num_one_step_obs,
        policy.num_one_step_critic_obs,
        self.device,
      )

    # RSL-RL runner compatibility: OnPolicyRunner expects `.rnd` to exist.
    self.rnd = None
    self.rnd_optimizer = None
    self.intrinsic_rewards = None

  def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    actor_obs_shape: tuple[int, ...],
    critic_obs_shape: tuple[int, ...] | None,
    action_shape: tuple[int, ...],
  ) -> None:
    self.storage = HIMRolloutStorage(
      num_envs=num_envs,
      num_transitions_per_env=num_transitions_per_env,
      obs_shape=actor_obs_shape,
      privileged_obs_shape=critic_obs_shape,
      actions_shape=action_shape,
      device=self.device,
      transitions_per_step=2 if self.use_flip else 1,
    )

  def act(self, obs: object) -> torch.Tensor:
    actor_obs = self.policy.get_actor_obs(obs).to(self.device)
    critic_obs = self.policy.get_critic_obs(obs).to(self.device)

    actions = self.policy.act(actor_obs).detach()
    values = self.policy.evaluate_critic_obs(critic_obs).detach()
    actions_log_prob = self.policy.get_actions_log_prob(actions).detach()

    self.transition.actions = actions
    self.transition.values = values
    self.transition.actions_log_prob = actions_log_prob
    self.transition.action_mean = self.policy.action_mean.detach()
    self.transition.action_sigma = self.policy.action_std.detach()
    self.transition.observations = actor_obs
    self.transition.critic_observations = critic_obs

    if self.use_flip:
      obs_sym = self._flip_actor_obs(actor_obs)
      critic_obs_sym = self._flip_critic_obs(critic_obs)
      actions_sym = self.policy.act(obs_sym).detach()
      values_sym = self.policy.evaluate_critic_obs(critic_obs_sym).detach()
      actions_log_prob_sym = self.policy.get_actions_log_prob(actions_sym).detach()

      self.transition_sym.actions = actions_sym
      self.transition_sym.values = values_sym
      self.transition_sym.actions_log_prob = actions_log_prob_sym
      self.transition_sym.action_mean = self.policy.action_mean.detach()
      self.transition_sym.action_sigma = self.policy.action_std.detach()
      self.transition_sym.observations = obs_sym
      self.transition_sym.critic_observations = critic_obs_sym

    return actions

  def process_env_step(
    self, obs: object, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    assert self.storage is not None

    critic_obs_next = self.policy.get_critic_obs(obs).to(self.device)

    # OpenHomie substitutes the PRE-reset (terminal) critic observation for
    # terminated envs (him_on_policy_runner.py:144) so the estimator's targets
    # on done transitions come from the terminal state, not the post-reset
    # one. The TerminalCriticObsRecorder captures it into extras.
    terminal = extras.pop("terminal_critic_obs", None) if extras else None
    if terminal is not None:
      terminal_ids, terminal_obs = terminal
      critic_obs_next = critic_obs_next.clone()
      critic_obs_next[terminal_ids.to(self.device)] = terminal_obs.to(self.device)

    step_rewards = rewards.to(self.device).detach().clone()
    self.transition.next_critic_observations = critic_obs_next.detach()
    self.transition.rewards = step_rewards
    self.transition.dones = dones.to(self.device)

    if self.use_flip:
      self.transition_sym.next_critic_observations = self._flip_critic_obs(
        critic_obs_next
      ).detach()
      self.transition_sym.rewards = step_rewards.clone()
      self.transition_sym.dones = self.transition.dones

    # Bootstrapping on timeouts.
    if "time_outs" in extras:
      time_outs = extras["time_outs"].to(self.device)
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * time_outs.unsqueeze(1), 1
      )
      if self.use_flip:
        self.transition_sym.rewards += self.gamma * torch.squeeze(
          self.transition_sym.values * time_outs.unsqueeze(1), 1
        )

    self.storage.add_transitions(self.transition)
    if self.use_flip:
      self.storage.add_transitions(self.transition_sym)

    self.transition.clear()
    self.transition_sym.clear()
    self.policy.reset(dones)

  def compute_returns(self, obs: object) -> None:
    assert self.storage is not None
    last_values = self.policy.evaluate(obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def train_mode(self) -> None:
    self.policy.train()

  def eval_mode(self) -> None:
    self.policy.eval()

  def save(self) -> dict[str, Any]:
    return {
      "policy_state_dict": self.policy.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "estimator_optimizer_state_dict": self.policy.estimator.optimizer.state_dict(),
    }

  def load(
    self, loaded_dict: dict[str, Any], load_cfg: dict | None, strict: bool
  ) -> bool:
    if load_cfg is None:
      load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

    policy_state_dict = loaded_dict.get("policy_state_dict")
    if policy_state_dict is None and "model_state_dict" in loaded_dict:
      policy_state_dict = loaded_dict["model_state_dict"]

    if policy_state_dict is not None and (
      load_cfg.get("actor") or load_cfg.get("critic")
    ):
      self.policy.load_state_dict(policy_state_dict, strict=strict)

    if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
      if "estimator_optimizer_state_dict" in loaded_dict:
        self.policy.estimator.optimizer.load_state_dict(
          loaded_dict["estimator_optimizer_state_dict"]
        )

    return bool(load_cfg.get("iteration", False))

  def get_policy(self) -> HIMActorCritic:
    return self.policy

  def broadcast_parameters(self) -> None:
    raise NotImplementedError("HIMPPO does not support distributed parameter sync.")

  def reduce_parameters(self) -> None:
    raise NotImplementedError("HIMPPO does not support distributed gradient sync.")

  @staticmethod
  def construct_algorithm(obs: Any, env: Any, cfg: dict, device: str) -> "HIMPPO":
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    obs_groups = cfg["obs_groups"]

    env_unwrapped = getattr(env, "unwrapped", None)
    if env_unwrapped is None:
      raise RuntimeError("HIMPPO expects the wrapped env to expose `.unwrapped`.")

    actor_group_names = obs_groups["actor"]
    critic_group_names = obs_groups["critic"]
    if len(actor_group_names) != 1 or len(critic_group_names) != 1:
      raise ValueError(
        "HIMPPO expects exactly one actor group and one critic group. "
        f"Got actor={actor_group_names}, critic={critic_group_names}."
      )

    actor_group = actor_group_names[0]
    critic_group = critic_group_names[0]
    actor_terms = list(env_unwrapped.cfg.observations[actor_group].terms.keys())
    critic_terms = list(env_unwrapped.cfg.observations[critic_group].terms.keys())
    if len(actor_terms) != 1 or len(critic_terms) != 1:
      raise ValueError(
        "HIMPPO requires one term per observation group to preserve flattened "
        f"history. Got actor_terms={actor_terms}, critic_terms={critic_terms}."
      )

    actor_term_cfg = env_unwrapped.observation_manager.get_term_cfg(
      actor_group, actor_terms[0]
    )
    critic_term_cfg = env_unwrapped.observation_manager.get_term_cfg(
      critic_group, critic_terms[0]
    )

    actor_history_length = max(1, int(actor_term_cfg.history_length))
    critic_history_length = max(1, int(critic_term_cfg.history_length))

    actor_obs_dim = sum(obs[name].shape[-1] for name in actor_group_names)
    critic_obs_dim = sum(obs[name].shape[-1] for name in critic_group_names)
    if actor_obs_dim % actor_history_length != 0:
      raise ValueError(
        f"Actor obs dim {actor_obs_dim} not divisible by history "
        f"{actor_history_length}."
      )
    if critic_obs_dim % critic_history_length != 0:
      raise ValueError(
        f"Critic obs dim {critic_obs_dim} not divisible by history "
        f"{critic_history_length}."
      )

    # Resolve the joint orderings used by the observations and actions; the
    # mirror maps for symmetry augmentation are derived from these names.
    joint_asset_cfg = actor_term_cfg.params["joint_asset_cfg"]
    entity = env_unwrapped.scene[joint_asset_cfg.name]
    _, obs_joint_names = entity.find_joints(
      joint_asset_cfg.joint_names, preserve_order=True
    )
    action_term = env_unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._target_names)

    actor_cfg = dict(cfg["actor"])
    critic_cfg = dict(cfg["critic"])
    alg_cfg = dict(cfg["algorithm"])
    actor_cfg.pop("class_name", None)
    critic_cfg.pop("class_name", None)
    alg_cfg.pop("class_name", None)

    actor_hidden_dims = tuple(actor_cfg.pop("hidden_dims", (512, 256, 256)))
    critic_hidden_dims = tuple(critic_cfg.pop("hidden_dims", (512, 256, 256)))
    activation = actor_cfg.pop("activation", critic_cfg.pop("activation", "elu"))
    distribution_cfg = actor_cfg.pop("distribution_cfg", None) or {}
    init_noise_std = float(distribution_cfg.get("init_std", 1.0))
    dynamic_latent_dim = int(actor_cfg.pop("dynamic_latent_dim", 32))
    terrain_latent_dim = int(actor_cfg.pop("terrain_latent_dim", 32))

    layout = HimObsLayout(
      num_one_step_obs=actor_obs_dim // actor_history_length,
      num_one_step_privileged_obs=critic_obs_dim // critic_history_length,
      actor_history_length=actor_history_length,
      critic_history_length=critic_history_length,
    )

    policy = HIMActorCritic(
      obs,
      obs_groups,
      env.num_actions,
      layout=layout,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=critic_hidden_dims,
      activation=activation,
      init_noise_std=init_noise_std,
      dynamic_latent_dim=dynamic_latent_dim,
      terrain_latent_dim=terrain_latent_dim,
    ).to(device)

    alg = HIMPPO(
      policy,
      device=device,
      mirror_obs_joint_names=tuple(obs_joint_names),
      mirror_action_joint_names=tuple(action_joint_names),
      **alg_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.init_storage(
      env.num_envs,
      cfg["num_steps_per_env"],
      actor_obs_shape=(actor_obs_dim,),
      critic_obs_shape=(critic_obs_dim,),
      action_shape=(env.num_actions,),
    )
    return alg

  def update(self) -> dict[str, float]:  # noqa: C901
    assert self.storage is not None

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_estimation_loss = 0.0
    mean_swap_loss = 0.0
    mean_actor_sym_loss = 0.0
    mean_critic_sym_loss = 0.0

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )

    for (
      obs_batch,
      critic_obs_batch,
      actions_batch,
      next_critic_obs_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
    ) in generator:
      self.policy.act(obs_batch)
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate_critic_obs(critic_obs_batch)
      mu_batch = self.policy.action_mean
      sigma_batch = self.policy.action_std
      entropy_batch = self.policy.entropy

      # Adaptive learning rate via KL.
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      # Estimator update (self-supervised env feature extraction).
      flipped_obs_batch: torch.Tensor | None = None
      if self.use_flip:
        flipped_obs_batch = self._flip_actor_obs(obs_batch)
        flipped_next_critic_obs_batch = self._flip_critic_obs(next_critic_obs_batch)
        estimator_obs_batch = torch.cat((obs_batch, flipped_obs_batch), dim=0)
        estimator_next_batch = torch.cat(
          (next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0
        )
      else:
        estimator_obs_batch = obs_batch
        estimator_next_batch = next_critic_obs_batch

      estimation_loss, swap_loss = self.policy.update_estimator(
        estimator_obs_batch, estimator_next_batch, lr=self.learning_rate
      )

      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )

      actor_sym_loss = torch.tensor(0.0, device=self.device)
      critic_sym_loss = torch.tensor(0.0, device=self.device)
      if self.use_flip:
        assert flipped_obs_batch is not None
        flipped_critic_obs_batch = self._flip_critic_obs(critic_obs_batch)
        actor_sym_loss = self.symmetry_scale * torch.mean(
          torch.sum(
            torch.square(
              self.policy.act_inference_actor_obs(flipped_obs_batch)
              - self._flip_actions(self.policy.act_inference_actor_obs(obs_batch))
            ),
            dim=-1,
          )
        )
        critic_sym_loss = self.symmetry_scale * torch.mean(
          torch.square(
            self.policy.evaluate_critic_obs(flipped_critic_obs_batch)
            - self.policy.evaluate_critic_obs(critic_obs_batch).detach()
          )
        )
        loss = loss + actor_sym_loss + critic_sym_loss

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += float(value_loss.item())
      mean_surrogate_loss += float(surrogate_loss.item())
      mean_estimation_loss += float(estimation_loss)
      mean_swap_loss += float(swap_loss)
      if self.use_flip:
        mean_actor_sym_loss += float(actor_sym_loss.item())
        mean_critic_sym_loss += float(critic_sym_loss.item())

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_estimation_loss /= num_updates
    mean_swap_loss /= num_updates
    if self.use_flip:
      mean_actor_sym_loss /= num_updates
      mean_critic_sym_loss /= num_updates

    self.storage.clear()

    loss_dict: dict[str, float] = {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "estimation": mean_estimation_loss,
      "swap": mean_swap_loss,
    }
    if self.use_flip:
      loss_dict["actor_sym"] = mean_actor_sym_loss
      loss_dict["critic_sym"] = mean_critic_sym_loss
    return loss_dict

  # Mirror transforms.

  def _flip_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
    maps = self._mirror_maps
    assert maps is not None
    b = obs.shape[0]
    h = self.policy.actor_history_length
    d = self.policy.num_one_step_obs
    x = obs.view(b, h, d)
    y = x[:, :, maps.actor_index] * maps.actor_sign
    return y.reshape(b, h * d).detach()

  def _flip_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
    maps = self._mirror_maps
    assert maps is not None
    b = critic_obs.shape[0]
    h = self.policy.critic_history_length
    d = self.policy.num_one_step_critic_obs
    x = critic_obs.view(b, h, d)
    y = x[:, :, maps.critic_index] * maps.critic_sign
    return y.reshape(b, h * d).detach()

  def _flip_actions(self, actions: torch.Tensor) -> torch.Tensor:
    maps = self._mirror_maps
    assert maps is not None
    return (actions[:, maps.action_index] * maps.action_sign).detach()
