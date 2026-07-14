"""HOMIE command terms: coupled twist and relative-height commands.

At every resampling instant each environment draws one of three mutually
exclusive modes (OpenHomie scheme):

- squat  (p = 1/3): zero twist, random height target
- walk   (p = 1/2): random twist, standing height target
- stand  (p = 1/6): zero twist, standing height target

The twist command samples the mode and exposes it via :attr:`mode`; the height
command couples to it. Both commands must share the same resampling interval,
and the twist command must precede the height command in the commands dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

MODE_STAND = 0
MODE_WALK = 1
MODE_SQUAT = 2


class UniformVelocityCommand(CommandTerm):
  """HOMIE twist command with three-mode sampling."""

  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.mode = torch.full(
      (self.num_envs,), MODE_STAND, device=self.device, dtype=torch.int64
    )

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    u = torch.rand(len(env_ids), device=self.device)
    is_squat = u < (1.0 / 3.0)
    is_walk = u > (1.0 / 2.0)

    self.mode[env_ids] = MODE_STAND
    self.mode[env_ids[is_walk]] = MODE_WALK
    self.mode[env_ids[is_squat]] = MODE_SQUAT

    self.vel_command_b[env_ids] = 0.0
    walk_ids = env_ids[is_walk]
    if walk_ids.numel() > 0:
      r = torch.empty(len(walk_ids), device=self.device)
      self.vel_command_b[walk_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
      self.vel_command_b[walk_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
      self.vel_command_b[walk_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

  def _update_command(self) -> None:
    pass

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw commanded and actual velocity arrows for the selected env."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    cmd = self.command[batch].cpu().numpy()
    base_pos_w = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    base_mat_w = matrix_from_quat(self.robot.data.root_link_quat_w)[batch].cpu().numpy()
    lin_vel_b = self.robot.data.root_link_lin_vel_b[batch].cpu().numpy()
    ang_vel_b = self.robot.data.root_link_ang_vel_b[batch].cpu().numpy()

    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    def local_to_world(vec: np.ndarray) -> np.ndarray:
      return base_pos_w + base_mat_w @ vec

    scale = self.cfg.viz.scale
    origin = np.array([0, 0, self.cfg.viz.z_offset]) * scale

    arrows = (
      (np.array([cmd[0], cmd[1], 0.0]), (0.2, 0.2, 0.6, 0.6)),  # cmd lin (blue)
      (np.array([0.0, 0.0, cmd[2]]), (0.2, 0.6, 0.2, 0.6)),  # cmd ang (green)
      (np.array([lin_vel_b[0], lin_vel_b[1], 0.0]), (0.0, 0.6, 1.0, 0.7)),  # actual lin
      (np.array([0.0, 0.0, ang_vel_b[2]]), (0.0, 1.0, 0.4, 0.7)),  # actual ang
    )
    for vec, color in arrows:
      visualizer.add_arrow(
        local_to_world(origin),
        local_to_world((origin / scale + vec) * scale),
        color=color,
        width=0.015,
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)


class RelativeHeightCommand(CommandTerm):
  """Base-height command relative to the support (lowest) foot site.

  Couples to the mode sampled by :class:`UniformVelocityCommand`: squat-mode
  envs draw a random height target; walk/stand envs get ``standing_height``.
  The command steps to the sampled target instantly (OpenHomie semantics).
  """

  cfg: RelativeHeightCommandCfg

  def __init__(self, cfg: RelativeHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    site_ids, _ = self.robot.find_sites(cfg.foot_site_names, preserve_order=True)
    if len(site_ids) == 0:
      raise ValueError(
        f"RelativeHeightCommand: no foot sites matched {cfg.foot_site_names}."
      )
    self._foot_site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)

    self.height_command = torch.full(
      (self.num_envs, 1), cfg.standing_height, device=self.device
    )
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _twist_mode(self) -> torch.Tensor:
    term = self._env.command_manager.get_term(self.cfg.twist_command_name)
    if not isinstance(term, UniformVelocityCommand):
      raise TypeError(
        f"Command '{self.cfg.twist_command_name}' must be UniformVelocityCommand."
      )
    return term.mode

  def _compute_relative_height(self) -> torch.Tensor:
    base_z = self.robot.data.root_link_pos_w[:, 2]
    support_z = torch.min(
      self.robot.data.site_pos_w[:, self._foot_site_ids, 2], dim=1
    ).values
    return base_z - support_z

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    error = torch.abs(self.height_command[:, 0] - self._compute_relative_height())
    self.metrics["error_height"] += error / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    is_squat = self._twist_mode()[env_ids] == MODE_SQUAT
    self.height_command[env_ids, 0] = float(self.cfg.standing_height)
    squat_ids = env_ids[is_squat]
    if squat_ids.numel() > 0:
      r = torch.empty(len(squat_ids), device=self.device)
      self.height_command[squat_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    pass

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw target and actual relative height for the selected env."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    base_pos_w = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    foot_pos_w = self.robot.data.site_pos_w[batch, self._foot_site_ids].cpu().numpy()
    support_z = float(np.min(foot_pos_w[:, 2]))
    target_h = float(self.height_command[batch, 0].cpu().item())

    start = np.array([base_pos_w[0], base_pos_w[1], support_z], dtype=np.float32)
    actual_end = base_pos_w.astype(np.float32)
    target_end = np.array(
      [base_pos_w[0], base_pos_w[1], support_z + target_h], dtype=np.float32
    )

    visualizer.add_arrow(
      start,
      target_end,
      color=self.cfg.viz.target_color,
      width=0.015,
      label="height_target",
    )
    visualizer.add_arrow(
      start,
      actual_end,
      color=self.cfg.viz.actual_color,
      width=0.015,
      label="height_actual",
    )
    visualizer.add_sphere(
      center=target_end,
      radius=self.cfg.viz.target_sphere_radius,
      color=self.cfg.viz.target_color,
      label="height_target_point",
    )


@dataclass(kw_only=True)
class RelativeHeightCommandCfg(CommandTermCfg):
  """Configuration for the relative-height command term."""

  entity_name: str
  foot_site_names: tuple[str, ...]
  standing_height: float
  """Height target used for non-squat (walk/stand) modes."""
  twist_command_name: str = "twist"
  """Name of the coupled :class:`UniformVelocityCommand` term. That term must
  come before this one in the commands dict so its mode is fresh when this term
  resamples."""

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    target_sphere_radius: float = 0.03
    target_color: tuple[float, float, float, float] = (0.7, 0.2, 0.7, 0.6)
    actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.7)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> RelativeHeightCommand:
    return RelativeHeightCommand(self, env)
