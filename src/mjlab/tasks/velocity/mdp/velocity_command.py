from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
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
    # Optionally gate this command to a specific env group (others are forced to stand still).
    if self.cfg.active_env_group is not None:
      try:
        group_mask = self._env.get_env_group_mask(self.cfg.active_env_group)
      except Exception:
        group_mask = None
      if group_mask is not None:
        active_mask = group_mask[env_ids]
        active_env_ids = env_ids[active_mask]
        inactive_env_ids = env_ids[~active_mask]
        if inactive_env_ids.numel() > 0:
          self.vel_command_b[inactive_env_ids, :] = 0.0
          self.heading_target[inactive_env_ids] = 0.0
          self.heading_error[inactive_env_ids] = 0.0
          self.is_heading_env[inactive_env_ids] = False
          self.is_standing_env[inactive_env_ids] = True
        env_ids = active_env_ids
        if env_ids.numel() == 0:
          return

    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows.

    Note: Only visualizes the selected environment (visualizer.env_idx).
    """
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    # Helper to transform local to world coordinates.
    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command linear velocity arrow (blue).
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

    # Command angular velocity arrow (green).
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # Actual linear velocity arrow (cyan).
    act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
    )

    # Actual angular velocity arrow (light green).
    act_ang_from = act_lin_from
    act_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
    )
    visualizer.add_arrow(
      act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  active_env_group: str | None = None
  class_type: type[CommandTerm] = UniformVelocityCommand

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )


class RelativeHeightCommand(CommandTerm):
  """Command a target base height relative to the support foot (lowest foot site).

  The tracked quantity is:
    h_rel = base_z - min_i(foot_site_z[i])
  which approximates "body height above ground" and works on uneven terrain.
  """

  cfg: RelativeHeightCommandCfg

  def __init__(self, cfg: RelativeHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]

    site_ids, site_names = self.robot.find_sites(
      cfg.foot_site_names, preserve_order=cfg.preserve_order
    )
    if len(site_ids) == 0:
      raise ValueError(
        "RelativeHeightCommand requires at least one foot site. "
        f"Got foot_site_names={cfg.foot_site_names} which matched none."
      )
    self._foot_site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    self._foot_site_names = site_names

    # Commanded relative height (meters). Shape: (num_envs, 1).
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _compute_relative_height(self) -> torch.Tensor:
    base_z = self.robot.data.root_link_pos_w[:, 2]
    foot_z = self.robot.data.site_pos_w[:, self._foot_site_ids, 2]
    support_z = torch.min(foot_z, dim=1).values
    return base_z - support_z

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    height = self._compute_relative_height()
    error = torch.abs(self.height_command[:, 0] - height)
    self.metrics["error_height"] += error / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Optionally gate this command to a specific env group (others get a fixed height target).
    if self.cfg.active_env_group is not None:
      try:
        group_mask = self._env.get_env_group_mask(self.cfg.active_env_group)
      except Exception:
        group_mask = None
      if group_mask is not None:
        active_mask = group_mask[env_ids]
        active_env_ids = env_ids[active_mask]
        inactive_env_ids = env_ids[~active_mask]
        if inactive_env_ids.numel() > 0:
          inactive_h = (
            float(self.cfg.inactive_height)
            if self.cfg.inactive_height is not None
            else 0.5 * (self.cfg.ranges.height[0] + self.cfg.ranges.height[1])
          )
          self.height_command[inactive_env_ids, 0] = inactive_h
        env_ids = active_env_ids
        if env_ids.numel() == 0:
          return

    r = torch.empty(len(env_ids), device=self.device)
    self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    # Command is held constant until resampling.
    pass

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw target and actual relative height for the selected environment."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    base_pos_w = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    foot_pos_w = (
      self.robot.data.site_pos_w[batch, self._foot_site_ids].cpu().numpy()
    )  # [N, 3]
    if foot_pos_w.size == 0:
      return

    support_z = float(np.min(foot_pos_w[:, 2]))
    target_h = float(self.height_command[batch, 0].cpu().item())

    xy_offset = np.array(self.cfg.viz.xy_offset, dtype=np.float32)
    z_offset = float(self.cfg.viz.z_offset)

    start = np.array([base_pos_w[0], base_pos_w[1], support_z], dtype=np.float32)
    start[:2] += xy_offset
    start[2] += z_offset

    actual_end = np.array(
      [base_pos_w[0], base_pos_w[1], base_pos_w[2]], dtype=np.float32
    )
    actual_end[:2] += xy_offset

    target_end = np.array(
      [base_pos_w[0], base_pos_w[1], support_z + target_h], dtype=np.float32
    )
    target_end[:2] += xy_offset
    target_end[2] += z_offset

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

  asset_name: str
  foot_site_names: tuple[str, ...]
  preserve_order: bool = False
  active_env_group: str | None = None
  inactive_height: float | None = None
  class_type: type[CommandTerm] = RelativeHeightCommand

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    xy_offset: tuple[float, float] = (0.0, 0.0)
    z_offset: float = 0.0
    target_sphere_radius: float = 0.03
    target_color: tuple[float, float, float, float] = (0.7, 0.2, 0.7, 0.6)
    actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.7)

  viz: VizCfg = field(default_factory=VizCfg)
