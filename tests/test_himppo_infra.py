"""Equivalence tests for the HIM-PPO infra optimizations.

The v8.1 speedups must not change the algorithm:
- rollout act() batches the original+mirrored forward (same math, fewer calls);
- the symmetry-loss references reuse mu_batch / value_batch instead of
  recomputing them (bitwise identical);
- the estimator trains on obs_batch alone (the storage already holds each
  transition's mirror, so the old cat(obs, flip(obs)) fed every sample twice).
"""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Normal

from mjlab_homierl.rl.himppo.actor_critic import HIMActorCritic, HimObsLayout
from mjlab_homierl.rl.himppo.algorithm import HIMPPO

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
)
NUM_DOFS = 4
NUM_ACTIONS = 4
NUM_COMMANDS = 4
ONE_STEP_OBS = NUM_COMMANDS + 6 + 2 * NUM_DOFS + NUM_ACTIONS  # 22
ONE_STEP_CRITIC_OBS = ONE_STEP_OBS + 3  # base_lin_vel tail
HISTORY = 3
ACTOR_OBS = HISTORY * ONE_STEP_OBS
CRITIC_OBS = HISTORY * ONE_STEP_CRITIC_OBS
NUM_ENVS = 8


def _make_obs(generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
  return {
    "actor": torch.randn(NUM_ENVS, ACTOR_OBS, generator=generator),
    "critic": torch.randn(NUM_ENVS, CRITIC_OBS, generator=generator),
  }


@pytest.fixture
def alg() -> HIMPPO:
  torch.manual_seed(0)
  policy = HIMActorCritic(
    _make_obs(),
    {"actor": ["actor"], "critic": ["critic"]},
    NUM_ACTIONS,
    layout=HimObsLayout(
      num_one_step_obs=ONE_STEP_OBS,
      num_one_step_privileged_obs=ONE_STEP_CRITIC_OBS,
      actor_history_length=HISTORY,
      critic_history_length=HISTORY,
    ),
    actor_hidden_dims=(32, 32),
    critic_hidden_dims=(32, 32),
    dynamic_latent_dim=8,
    terrain_latent_dim=8,
    estimator_hidden_dims=(16, 16),
  )
  alg = HIMPPO(
    policy,
    use_flip=True,
    mirror_obs_joint_names=JOINT_NAMES,
    mirror_action_joint_names=JOINT_NAMES,
    num_learning_epochs=2,
    num_mini_batches=2,
    device="cpu",
  )
  alg.init_storage(NUM_ENVS, 4, (ACTOR_OBS,), (CRITIC_OBS,), (NUM_ACTIONS,))
  return alg


def test_flip_is_an_involution(alg: HIMPPO) -> None:
  x = torch.randn(NUM_ENVS, ACTOR_OBS)
  assert torch.equal(alg._flip_actor_obs(alg._flip_actor_obs(x)), x)
  c = torch.randn(NUM_ENVS, CRITIC_OBS)
  assert torch.equal(alg._flip_critic_obs(alg._flip_critic_obs(c)), c)


def test_sym_loss_references_are_bitwise_identical(alg: HIMPPO) -> None:
  """update() now reuses mu_batch / value_batch for the symmetry-loss
  references; assert reuse equals the recomputation the old code did."""
  policy = alg.policy
  obs = torch.randn(NUM_ENVS, ACTOR_OBS)
  critic_obs = torch.randn(NUM_ENVS, CRITIC_OBS)

  policy.act(obs)
  mu_batch = policy.action_mean.detach()
  value_batch = policy.evaluate_critic_obs(critic_obs).detach()

  with torch.no_grad():
    recomputed_mu = policy.act_inference_actor_obs(obs)
    recomputed_value = policy.evaluate_critic_obs(critic_obs)

  assert torch.equal(mu_batch, recomputed_mu)
  assert torch.equal(value_batch, recomputed_value)


def test_batched_act_matches_separate_forwards(alg: HIMPPO) -> None:
  policy = alg.policy
  obs = _make_obs()
  actor_obs = obs["actor"]
  critic_obs = obs["critic"]

  alg.act(obs)
  t, ts = alg.transition, alg.transition_sym

  assert torch.equal(t.observations, actor_obs)
  assert torch.equal(ts.observations, alg._flip_actor_obs(actor_obs))
  assert torch.equal(ts.critic_observations, alg._flip_critic_obs(critic_obs))

  with torch.no_grad():
    mean_orig = policy.act_inference_actor_obs(actor_obs)
    mean_sym = policy.act_inference_actor_obs(alg._flip_actor_obs(actor_obs))
    values_orig = policy.evaluate_critic_obs(critic_obs)
    values_sym = policy.evaluate_critic_obs(alg._flip_critic_obs(critic_obs))

  assert torch.allclose(t.action_mean, mean_orig, atol=1e-6)
  assert torch.allclose(ts.action_mean, mean_sym, atol=1e-6)
  assert torch.allclose(t.values, values_orig, atol=1e-6)
  assert torch.allclose(ts.values, values_sym, atol=1e-6)

  # Stored log-probs must be consistent with the stored distribution/actions.
  for tr in (t, ts):
    dist = Normal(tr.action_mean, tr.action_sigma)
    assert torch.allclose(
      dist.log_prob(tr.actions).sum(dim=-1), tr.actions_log_prob, atol=1e-5
    )


def test_full_rollout_and_update_runs(alg: HIMPPO) -> None:
  torch.manual_seed(1)
  est_before = [p.clone() for p in alg.policy.estimator.parameters()]

  for _ in range(4):
    obs = _make_obs()
    alg.act(obs)
    alg.process_env_step(
      obs,
      rewards=torch.randn(NUM_ENVS),
      dones=torch.zeros(NUM_ENVS, dtype=torch.uint8),
      extras={},
    )
  alg.compute_returns(obs)
  loss_dict = alg.update()

  expected_keys = {
    "value_function",
    "surrogate",
    "estimation",
    "swap",
    "actor_sym",
    "critic_sym",
  }
  assert set(loss_dict) == expected_keys
  for key, value in loss_dict.items():
    assert isinstance(value, float), key
    assert torch.isfinite(torch.tensor(value)), key

  # The estimator optimizer actually stepped.
  assert any(
    not torch.equal(before, after)
    for before, after in zip(est_before, alg.policy.estimator.parameters())
  )
  assert alg.storage is not None and alg.storage.step == 0
