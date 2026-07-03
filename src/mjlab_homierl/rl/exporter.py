import copy
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


class _HimOnnxPolicyExporter(torch.nn.Module):
  """ONNX wrapper for HOMIE HIMActorCritic policies.

  The standard Isaac-Lab-style exporter only serializes `policy.actor`, but
  HIMActorCritic requires estimator-based preprocessing from the raw actor
  observation history. This wrapper exports the full inference path:

    obs_history -> normalizer -> HIMActorCritic.act_inference_actor_obs -> actions
  """

  def __init__(self, actor_critic: object, normalizer: object | None, verbose: bool):
    super().__init__()
    self.verbose = bool(verbose)

    # NOTE: Do not deepcopy the full actor_critic module.
    # HIMPPO policies cache non-leaf tensors (e.g., action distribution mean/std)
    # which PyTorch does not allow to be deep-copied in newer versions.
    self.actor = copy.deepcopy(actor_critic.actor)
    self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
    self.num_actor_obs = int(actor_critic.num_actor_obs)
    self.num_one_step_obs = int(actor_critic.num_one_step_obs)
    self.actor_proprioceptive_obs_length = int(
      actor_critic.actor_proprioceptive_obs_length
    )
    self.num_height_points = int(getattr(actor_critic, "num_height_points", 0))
    self.actor_use_height = bool(getattr(actor_critic, "actor_use_height", False))
    if self.actor_use_height:
      self.terrain_encoder = copy.deepcopy(actor_critic.terrain_encoder)
    else:
      self.terrain_encoder = None

    self.actor.eval()
    self.estimator.eval()
    if self.terrain_encoder is not None:
      self.terrain_encoder.eval()

    if normalizer is not None:
      self.normalizer = copy.deepcopy(normalizer)
    else:
      self.normalizer = torch.nn.Identity()

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    obs = self.normalizer(obs)

    parts = self.estimator(obs[:, : self.actor_proprioceptive_obs_length])
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2.0)

    if self.actor_use_height:
      assert self.terrain_encoder is not None
      terrain_in = obs[:, -(self.num_height_points + self.num_one_step_obs) :]
      terrain_latent = self.terrain_encoder(terrain_in)
      last_step = obs[
        :, -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points
      ]
      actor_in = torch.cat((last_step, vel, z, terrain_latent), dim=-1)
    else:
      last_step = obs[:, -self.num_one_step_obs :]
      actor_in = torch.cat((last_step, vel, z), dim=-1)

    return self.actor(actor_in)

  def export(self, path: str, filename: str) -> None:
    self.to("cpu")
    self.eval()

    obs = torch.zeros(1, self.num_actor_obs, dtype=torch.float32)
    torch.onnx.export(
      self,
      obs,
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )


def export_homie_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

  if not (
    hasattr(actor_critic, "act_inference_actor_obs")
    and hasattr(actor_critic, "estimator")
  ):
    raise TypeError("HOMIE ONNX export expects a HIMActorCritic-compatible policy.")

  policy_exporter = _HimOnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


def _train_repo_commit() -> str:
  """Git commit of this training repo at export time (provenance)."""
  try:
    return (
      subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
      ).stdout.strip()
      or "unknown"
    )
  except Exception:
    return "unknown"


def homie_extra_metadata(env: ManagerBasedRlEnv) -> dict:
  """HOMIE-specific ONNX metadata beyond mjlab's base fields.

  Everything a downstream controller plugin needs to run the policy without
  hardcoding conventions: the paired init keyframe base height, the height
  command semantics, the observation scaling/layout, and training provenance.
  """
  metadata: dict = {"train_repo_commit": _train_repo_commit()}

  # Init keyframe: joint pose (base metadata `default_joint_pos`) and base
  # height must travel as a pair.
  robot_cfg = env.cfg.scene.entities["robot"]
  init_state = robot_cfg.init_state
  if init_state is not None and init_state.pos is not None:
    metadata["init_base_height"] = float(init_state.pos[2])

  # Height command semantics (relative pelvis height above the lowest foot).
  try:
    height_cfg = env.command_manager.get_term("height").cfg
    metadata["height_command_range"] = list(height_cfg.ranges.height)
    metadata["standing_height"] = float(height_cfg.standing_height)
  except KeyError:
    pass

  # Observation scaling and layout of the flattened actor history.
  obs_term_cfg = env.observation_manager.get_term_cfg("actor", "him_obs")
  obs_scales = obs_term_cfg.params.get("obs_scales")
  if obs_scales is not None:
    for key, value in obs_scales.items():
      metadata[f"obs_scale_{key}"] = float(value)
  history_length = max(1, int(obs_term_cfg.history_length))
  metadata["obs_history_length"] = history_length
  actor_dim = env.observation_manager.group_obs_dim["actor"][0]
  metadata["num_one_step_obs"] = int(actor_dim) // history_length

  return metadata


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach base + HOMIE-specific metadata to an exported ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)
  metadata.update(homie_extra_metadata(env))
  attach_metadata_to_onnx(onnx_path, metadata)
