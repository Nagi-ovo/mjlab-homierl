import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path just in case
sys.path.append(str(Path(__file__).parent / "src"))

from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class ObservationNormalizer(nn.Module):
  def __init__(self, dim=97):
    super().__init__()
    self.register_buffer("_mean", torch.zeros(dim))
    self.register_buffer("_std", torch.ones(dim))

  def load_from_state_dict(self, state_dict, prefix="actor_obs_normalizer."):
    # Squeeze to ensure shapes match [97]
    self._mean.copy_(state_dict[prefix + "_mean"].squeeze())
    self._std.copy_(state_dict[prefix + "_std"].squeeze())

  def forward(self, obs):
    return (obs - self._mean) / (self._std + 1e-8)


class H1Actor(nn.Module):
  """
  Unitree H1 Actor Network Structure

  Inputs (97 dims):
  - Base Lin/Ang Vel & Projected Gravity (9)
  - All Joint Pos & Vel (37x2 = 74)
  - Previous Actions (10)
  - Velocity & Height Commands (3+1 = 4)

  Outputs (10 dims):
  - Lower body joint targets (Hip Yaw, Roll, Pitch, Knee, Ankle x 2)
  """

  def __init__(self, input_dim=97, output_dim=10, hidden_dims=[512, 256, 128]):
    super().__init__()
    self.normalizer = ObservationNormalizer(input_dim)
    layers = []
    curr_dim = input_dim
    for h_dim in hidden_dims:
      layers.append(nn.Linear(curr_dim, h_dim))
      layers.append(nn.ELU())
      curr_dim = h_dim
    layers.append(nn.Linear(curr_dim, output_dim))
    self.actor = nn.Sequential(*layers)

  def forward(self, obs):
    if isinstance(obs, (dict, TensorDict)):
      obs = obs["policy"]
    with torch.no_grad():
      obs = self.normalizer(obs)
      return self.actor(obs)


def run_demo(checkpoint_path="homie_rl.pt", viewer_type="auto", num_envs=1):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[INFO] Using device: {device}")

  # 1. Load the model
  print(f"[INFO] Loading model from {checkpoint_path}...")
  if not os.path.exists(checkpoint_path):
    print(f"[ERROR] {checkpoint_path} not found.")
    return

  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint.get("model_state_dict", checkpoint)

  actor = H1Actor(input_dim=97, output_dim=10).to(device)

  # Map actor weights
  actor_state_dict = {}
  for k, v in state_dict.items():
    if k.startswith("actor.") and "normalizer" not in k:
      actor_state_dict[k.replace("actor.", "")] = v
  actor.actor.load_state_dict(actor_state_dict)

  # Load normalizer
  actor.normalizer.load_from_state_dict(state_dict, prefix="actor_obs_normalizer.")

  actor.eval()
  print("[INFO] Model and Normalizer loaded successfully.")

  # 2. Setup the environment
  task_id = "Mjlab-Walk-Unitree-H1-with_hands"
  print(f"[INFO] Setting up environment: {task_id} with {num_envs} envs...")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  # 3. Handle Viewer
  if viewer_type == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_type = "native" if has_display else "viser"

  print(f"[INFO] Using viewer: {viewer_type}")
  if viewer_type == "native":
    viewer = NativeMujocoViewer(env, actor)
  elif viewer_type == "viser":
    viewer = ViserPlayViewer(env, actor)
  else:
    print(f"[ERROR] Unknown viewer type: {viewer_type}")
    env.close()
    return

  # 4. Run loop (handled by viewer)
  print("[INFO] Starting demo. Press Ctrl+C to stop.")
  try:
    viewer.run()
  except KeyboardInterrupt:
    print("\n[INFO] Demo stopped by user.")
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run H1 RL Demo")
  parser.add_argument(
    "--checkpoint", type=str, default="homie_rl.pt", help="Path to checkpoint"
  )
  parser.add_argument(
    "--viewer",
    type=str,
    default="auto",
    choices=["auto", "native", "viser"],
    help="Viewer type",
  )
  parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
  args = parser.parse_args()

  run_demo(
    checkpoint_path=args.checkpoint, viewer_type=args.viewer, num_envs=args.num_envs
  )
