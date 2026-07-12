from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation(act_name: str) -> nn.Module:
  if act_name == "elu":
    return nn.ELU()
  if act_name == "selu":
    return nn.SELU()
  if act_name == "relu":
    return nn.ReLU()
  if act_name == "crelu":
    return nn.ReLU()
  if act_name == "silu":
    return nn.SiLU()
  if act_name == "lrelu":
    return nn.LeakyReLU()
  if act_name == "tanh":
    return nn.Tanh()
  if act_name == "sigmoid":
    return nn.Sigmoid()
  raise ValueError(f"Invalid activation function: {act_name}")


class HIMEstimator(nn.Module):
  def __init__(
    self,
    temporal_steps: int,
    num_one_step_obs: int,
    num_height_points: int,
    *,
    enc_hidden_dims: tuple[int, ...] = (256, 256),
    tar_hidden_dims: tuple[int, ...] = (256, 256),
    latent_dim: int = 32,
    activation: str = "elu",
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
    num_prototype: int = 64,
    temperature: float = 3.0,
  ) -> None:
    super().__init__()

    act = get_activation(activation)

    self.temporal_steps = int(temporal_steps)
    self.num_one_step_obs = int(num_one_step_obs)
    self.num_height_points = int(num_height_points)
    self.num_latent = int(latent_dim)
    self.max_grad_norm = float(max_grad_norm)
    self.temperature = float(temperature)

    # Encoder
    enc_input_dim = self.temporal_steps * self.num_one_step_obs + self.num_height_points
    enc_layers: list[nn.Module] = []
    for hidden_dim in enc_hidden_dims:
      enc_layers.extend([nn.Linear(enc_input_dim, hidden_dim), act])
      enc_input_dim = int(hidden_dim)
    enc_layers.append(nn.Linear(enc_input_dim, 3 + self.num_latent))
    self.encoder = nn.Sequential(*enc_layers)

    # Target
    tar_input_dim = self.num_one_step_obs
    tar_layers: list[nn.Module] = []
    for hidden_dim in tar_hidden_dims:
      tar_layers.extend([nn.Linear(tar_input_dim, hidden_dim), act])
      tar_input_dim = int(hidden_dim)
    tar_layers.append(nn.Linear(tar_input_dim, self.num_latent))
    self.target = nn.Sequential(*tar_layers)

    # Prototype
    self.proto = nn.Embedding(int(num_prototype), self.num_latent)

    # Optimizer
    self.learning_rate = float(learning_rate)
    self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

  def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    parts = self.encoder(obs_history.detach())
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2)
    return vel.detach(), z.detach()

  def update(
    self,
    obs_history: torch.Tensor,
    next_critic_obs: torch.Tensor,
    lr: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if lr is not None:
      self.learning_rate = float(lr)
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate

    # Target velocity is privileged base linear velocity at the end of privileged obs.
    vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3].detach()
    # Next-step observation for the target network excludes velocity commands (first 3 dims).
    next_obs = next_critic_obs.detach()[:, 3 : self.num_one_step_obs + 3]

    z_s = self.encoder(obs_history.detach())
    z_t = self.target(next_obs)
    pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

    z_s = F.normalize(z_s, dim=-1, p=2)
    z_t = F.normalize(z_t, dim=-1, p=2)

    with torch.no_grad():
      w = self.proto.weight.data.clone()
      w = F.normalize(w, dim=-1, p=2)
      self.proto.weight.copy_(w)

    score_s = z_s @ self.proto.weight.T
    score_t = z_t @ self.proto.weight.T

    with torch.no_grad():
      q_s = sinkhorn(score_s)
      q_t = sinkhorn(score_t)

    log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

    swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
    estimation_loss = F.mse_loss(pred_vel, vel)
    losses = estimation_loss + swap_loss

    self.optimizer.zero_grad()
    losses.backward()
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()

    # Detached GPU tensors: callers accumulate on-device and sync once per
    # update instead of forcing a GPU->CPU sync every minibatch.
    return estimation_loss.detach(), swap_loss.detach()


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  q = torch.exp(out / eps).T
  k, b = q.shape[0], q.shape[1]
  q /= q.sum()

  for _ in range(int(iters)):
    q /= torch.sum(q, dim=1, keepdim=True)
    q /= k
    q /= torch.sum(q, dim=0, keepdim=True)
    q /= b
  return (q * b).T
