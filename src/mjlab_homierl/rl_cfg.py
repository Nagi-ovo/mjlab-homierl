"""RL configuration for the HOMIE tasks."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class HomieHimActorCfg:
  # OpenHomie HIMActorCritic hidden dims.
  hidden_dims: tuple[int, ...] = (512, 256, 256)
  activation: str = "elu"
  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )
  dynamic_latent_dim: int = 32
  terrain_latent_dim: int = 32
  estimator_hidden_dims: tuple[int, ...] = (256, 256)
  class_name: str = "mjlab_homierl.rl.himppo.actor_critic.HIMActorCritic"


@dataclass
class HomieHimCriticCfg:
  hidden_dims: tuple[int, ...] = (512, 256, 256)
  activation: str = "elu"
  class_name: str = "mjlab_homierl.rl.himppo.actor_critic.HIMActorCritic"


@dataclass
class HomieHimPpoAlgorithmCfg:
  use_flip: bool = True
  symmetry_scale: float = 1.0
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float | None = 0.01
  clip_param: float = 0.2
  entropy_coef: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  rnd_cfg: dict[str, Any] | None = None
  class_name: str = "mjlab_homierl.rl.himppo.algorithm.HIMPPO"


@dataclass
class HomieHimOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  actor: HomieHimActorCfg = field(default_factory=HomieHimActorCfg)
  critic: HomieHimCriticCfg = field(default_factory=HomieHimCriticCfg)
  algorithm: HomieHimPpoAlgorithmCfg = field(default_factory=HomieHimPpoAlgorithmCfg)


def homie_himppo_runner_cfg(experiment_name: str) -> HomieHimOnPolicyRunnerCfg:
  """HIM-PPO runner configuration shared by the HOMIE tasks."""
  return HomieHimOnPolicyRunnerCfg(
    experiment_name=experiment_name,
    save_interval=200,
    num_steps_per_env=50,
    max_iterations=30_000,
    # Keep W&B metric logging but never upload checkpoints/ONNX artifacts.
    upload_model=False,
  )


def unitree_g1_homie_himppo_runner_cfg() -> HomieHimOnPolicyRunnerCfg:
  return homie_himppo_runner_cfg("g1_homie_himppo")


def unitree_h1_homie_himppo_runner_cfg() -> HomieHimOnPolicyRunnerCfg:
  return homie_himppo_runner_cfg("h1_homie_himppo")
