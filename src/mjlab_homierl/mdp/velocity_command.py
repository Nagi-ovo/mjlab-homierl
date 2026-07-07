"""HOMIE command terms: coupled twist and relative-height commands.

At every resampling instant each environment draws one of three mutually
exclusive modes (OpenHomie scheme):

- squat  (p = 1/3): zero twist, random height target
- walk   (p = 1/2): random twist, standing height target
- stand  (p = 1/6): zero twist, standing height target

Optionally, ``inplace_prob`` converts that fraction of walk-mode resamples
into in-place locomotion commands: vx = 0 with the sampled (vy, wz) kept and
the dominant of the two clamped away from zero. This is an extension over
OpenHomie: its sampler draws vx/vy/wz jointly, so "strafe or rotate without
advancing" has measure zero and trained policies gate their gait on |vx|
alone — probes showed pure-turn AND pure-strafe commands leave the robot
standing at 100% double support. Default 0.0 = exact OpenHomie parity.

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
      if self.cfg.inplace_prob > 0.0:
        is_ip = (
          torch.rand(len(walk_ids), device=self.device) < self.cfg.inplace_prob
        )
        ip_ids = walk_ids[is_ip]
        if ip_ids.numel() > 0:
          self.vel_command_b[ip_ids, 0] = 0.0
          vy = self.vel_command_b[ip_ids, 1]
          wz = self.vel_command_b[ip_ids, 2]
          # Clamp the dominant axis away from zero so every in-place env
          # actually strafes and/or turns; the other axis keeps its sampled
          # value, so pure-strafe, pure-turn, and combos all occur.
          min_mag = float(self.cfg.inplace_min_cmd)
          vy_c = torch.where(
            vy.abs() < min_mag, min_mag * torch.where(vy < 0.0, -1.0, 1.0), vy
          )
          wz_c = torch.where(
            wz.abs() < min_mag, min_mag * torch.where(wz < 0.0, -1.0, 1.0), wz
          )
          wz_dominant = wz.abs() >= vy.abs()
          self.vel_command_b[ip_ids, 1] = torch.where(wz_dominant, vy, vy_c)
          self.vel_command_b[ip_ids, 2] = torch.where(wz_dominant, wz_c, wz)

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

  inplace_prob: float = 0.0
  """Fraction of walk-mode resamples converted to in-place locomotion
  (vx = 0, sampled vy/wz kept, dominant axis clamped to ``inplace_min_cmd``).

  0.0 (default) reproduces OpenHomie's sampler exactly. In-place envs keep
  walk-mode semantics everywhere else (standing height target, twist-gated
  reward terms see a nonzero command norm).
  """

  inplace_min_cmd: float = 0.3
  """Minimum magnitude for the dominant in-place axis (vy or wz); smaller
  draws are pushed to this value (sign preserved) so the mode never
  degenerates into standing."""

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

  With ``max_rate_range`` set, the exposed command is a SLEWED setpoint that
  moves toward the sampled target at a per-env rate drawn from that range.
  OpenHomie feeds the target as an instantaneous step, but height is a
  position-type command: a step creates a physically unreachable error window
  whose exp-tracking gradient rewards ballistic descent (the v3 policy
  crash-squatted onto its knees). The slewed setpoint defines a well-executed
  descent at every instant, the endpoint is unobservable to the policy (no
  incentive to race ahead), and it matches deployment, where stick commands
  ramp. ``None`` (default) keeps OpenHomie's step semantics.
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
    self._height_target = torch.full(
      (self.num_envs,), cfg.standing_height, device=self.device
    )
    self._slew_rate = torch.zeros(self.num_envs, device=self.device)
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

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # Episodes spawn at the default (standing) pose; re-anchor the slewed
    # setpoint so a fresh episode never inherits the dying episode's height.
    assert isinstance(env_ids, torch.Tensor)  # matches the base-class contract
    self.height_command[env_ids, 0] = float(self.cfg.standing_height)
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    is_squat = self._twist_mode()[env_ids] == MODE_SQUAT
    self._height_target[env_ids] = float(self.cfg.standing_height)
    squat_ids = env_ids[is_squat]
    if squat_ids.numel() > 0:
      r = torch.empty(len(squat_ids), device=self.device)
      self._height_target[squat_ids] = r.uniform_(*self.cfg.ranges.height)
    if self.cfg.max_rate_range is None:
      # OpenHomie parity: the command steps to the target instantly.
      self.height_command[env_ids, 0] = self._height_target[env_ids]
    else:
      r = torch.empty(len(env_ids), device=self.device)
      self._slew_rate[env_ids] = r.uniform_(*self.cfg.max_rate_range)

  def _update_command(self) -> None:
    if self.cfg.max_rate_range is None:
      return
    step = self._slew_rate * self._env.step_dt
    delta = self._height_target - self.height_command[:, 0]
    self.height_command[:, 0] += torch.clamp(delta, -step, step)

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

  max_rate_range: tuple[float, float] | None = None
  """Per-env slew-rate range [m/s] for the exposed height setpoint, drawn at
  each resample. ``None`` (default) = OpenHomie parity: the command steps to
  the sampled target instantly. E.g. ``(0.25, 0.75)`` spans a gentle to a
  brisk human squat descent; deployment picks any rate inside the trained
  envelope without retraining."""

  @dataclass
  class VizCfg:
    target_sphere_radius: float = 0.03
    target_color: tuple[float, float, float, float] = (0.7, 0.2, 0.7, 0.6)
    actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.7)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> RelativeHeightCommand:
    return RelativeHeightCommand(self, env)


class TorsoPitchCommand(CommandTerm):
  """HOMIE+ torso-pitch command: a waist_pitch joint-angle target (rad, +fwd).

  Couples to the mode sampled by :class:`UniformVelocityCommand`, with the
  pitch law keyed on moving vs stationary rather than the mode name:

  - walk  envs (moving): 0 with p = ``walk_zero_prob``, else U(``walk_range``)
    — covers "look down / reach while walking";
  - squat/stand envs (stationary): 0 with p = ``squat_zero_prob``, else
    U(``squat_range``) — squat + lean is the pick-from-floor work case, and
    stand + lean is the reach-over-a-table case (the most common teleop
    manipulation pose; v3 covered it only via the shallow-squat corner).

  The command is a joint-space target so upstream IK/teleop can drive it
  directly, with no attitude-estimation loop (homie_plus_plan.md §2.1).
  """

  cfg: "TorsoPitchCommandCfg"

  def __init__(self, cfg: "TorsoPitchCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    joint_ids, _ = self.robot.find_joints((cfg.joint_name,), preserve_order=True)
    if len(joint_ids) != 1:
      raise ValueError(
        f"TorsoPitchCommand: joint '{cfg.joint_name}' not found or ambiguous."
      )
    self._joint_id = int(joint_ids[0])

    self.pitch_command = torch.zeros(self.num_envs, 1, device=self.device)
    self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.pitch_command

  @property
  def joint_id(self) -> int:
    return self._joint_id

  def _twist_mode(self) -> torch.Tensor:
    term = self._env.command_manager.get_term(self.cfg.twist_command_name)
    if not isinstance(term, UniformVelocityCommand):
      raise TypeError(
        f"Command '{self.cfg.twist_command_name}' must be UniformVelocityCommand."
      )
    return term.mode

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    actual = self.robot.data.joint_pos[:, self._joint_id]
    error = torch.abs(self.pitch_command[:, 0] - actual)
    self.metrics["error_pitch"] += error / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    mode = self._twist_mode()[env_ids]
    self.pitch_command[env_ids, 0] = 0.0
    for mode_id, zero_prob, rng in (
      (MODE_WALK, self.cfg.walk_zero_prob, self.cfg.walk_range),
      (MODE_SQUAT, self.cfg.squat_zero_prob, self.cfg.squat_range),
      # Stationary law: stand shares the squat pitch distribution.
      (MODE_STAND, self.cfg.squat_zero_prob, self.cfg.squat_range),
    ):
      ids = env_ids[mode == mode_id]
      if ids.numel() == 0:
        continue
      active = torch.rand(len(ids), device=self.device) >= zero_prob
      act_ids = ids[active]
      if act_ids.numel() > 0:
        r = torch.empty(len(act_ids), device=self.device)
        self.pitch_command[act_ids, 0] = r.uniform_(*rng)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass


@dataclass(kw_only=True)
class TorsoPitchCommandCfg(CommandTermCfg):
  """Configuration for the HOMIE+ torso-pitch command term."""

  entity_name: str
  joint_name: str = "waist_pitch_joint"
  twist_command_name: str = "twist"
  """Coupled mode source; must precede this term in the commands dict."""

  walk_zero_prob: float = 0.7
  walk_range: tuple[float, float] = (-0.15, 0.25)
  squat_zero_prob: float = 0.5
  squat_range: tuple[float, float] = (-0.2, 0.45)

  def build(self, env: ManagerBasedRlEnv) -> TorsoPitchCommand:
    return TorsoPitchCommand(self, env)
