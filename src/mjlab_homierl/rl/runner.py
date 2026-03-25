import os
from dataclasses import dataclass
from typing import Any

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab_homierl.rl.exporter import (
  attach_onnx_metadata,
  export_homie_policy_as_onnx,
)
from mjlab_homierl.rl.himppo.actor_critic import HIMActorCritic, HimObsLayout


@dataclass
class _InferenceOnlyAlgorithm:
  policy: HIMActorCritic

  def eval_mode(self) -> None:
    self.policy.eval()

  def get_policy(self) -> HIMActorCritic:
    return self.policy


class HomieHimOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self._inference_only = False
    obs = env.get_observations()
    if "critic" not in obs.keys():
      self._init_inference_only(env, train_cfg, obs, device)
      return
    super().__init__(env, train_cfg, log_dir, device)

  def _init_inference_only(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    obs: Any,
    device: str,
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.current_learning_iteration = 0
    self.logger = None
    self._inference_only = True

    env_unwrapped = env.unwrapped
    actor_group = "actor"
    actor_terms = list(env_unwrapped.cfg.observations[actor_group].terms.keys())
    if len(actor_terms) != 1:
      raise ValueError(
        "HOMIE inference-only play requires exactly one actor observation term. "
        f"Got actor_terms={actor_terms}."
      )

    actor_term_cfg = env_unwrapped.observation_manager.get_term_cfg(actor_group, actor_terms[0])
    actor_history_length = max(1, int(actor_term_cfg.history_length))
    actor_obs_dim = int(obs[actor_group].shape[-1])
    if actor_obs_dim % actor_history_length != 0:
      raise ValueError(
        f"Actor obs dim {actor_obs_dim} is not divisible by history length {actor_history_length}."
      )

    actor_cfg = dict(train_cfg["actor"])
    critic_cfg = dict(train_cfg.get("critic", {}))
    actor_cfg.pop("class_name", None)
    actor_hidden_dims = tuple(actor_cfg.pop("hidden_dims", (512, 256, 128)))
    activation = actor_cfg.pop("activation", critic_cfg.pop("activation", "elu"))
    distribution_cfg = actor_cfg.pop("distribution_cfg", None) or {}
    init_noise_std = float(distribution_cfg.get("init_std", 1.0))
    dynamic_latent_dim = int(actor_cfg.pop("dynamic_latent_dim", 32))
    terrain_latent_dim = int(actor_cfg.pop("terrain_latent_dim", 32))

    policy = HIMActorCritic(
      obs,
      {"actor": [actor_group]},
      env.num_actions,
      layout=HimObsLayout(
        num_one_step_obs=actor_obs_dim // actor_history_length,
        num_one_step_privileged_obs=0,
        actor_history_length=actor_history_length,
        critic_history_length=0,
      ),
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=(1,),
      activation=activation,
      init_noise_std=init_noise_std,
      dynamic_latent_dim=dynamic_latent_dim,
      terrain_latent_dim=terrain_latent_dim,
    ).to(device)
    policy.eval()
    self.alg = _InferenceOnlyAlgorithm(policy)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    export_homie_policy_as_onnx(
      self.alg.get_policy(),
      normalizer=None,
      path=path,
      filename=filename,
      verbose=verbose,
    )

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)

    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      attach_onnx_metadata(
        self.env.unwrapped,
        run_name,
        path=policy_path,
        filename=filename,
      )
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as e:
      print(f"[WARN] HOMIE ONNX export failed (training continues): {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if not self._inference_only:
      return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    policy_state_dict = loaded_dict.get("policy_state_dict") or loaded_dict.get("model_state_dict")
    if policy_state_dict is None:
      raise KeyError("Checkpoint must contain `policy_state_dict` or `model_state_dict`.")

    policy = self.alg.get_policy()
    model_keys = set(policy.state_dict().keys())
    filtered_state_dict = {k: v for k, v in policy_state_dict.items() if k in model_keys}
    missing_keys, unexpected_keys = policy.load_state_dict(filtered_state_dict, strict=False)
    if strict:
      missing_keys = [k for k in missing_keys if not k.startswith("critic.")]
      unexpected_keys = [k for k in unexpected_keys if not k.startswith("critic.")]
      if missing_keys or unexpected_keys:
        raise RuntimeError(
          "Inference-only HOMIE policy load mismatch. "
          f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    return loaded_dict

  def get_inference_policy(self, device: str | None = None) -> HIMActorCritic:
    if not self._inference_only:
      return super().get_inference_policy(device=device)
    self.alg.eval_mode()
    return self.alg.get_policy().to(device)
