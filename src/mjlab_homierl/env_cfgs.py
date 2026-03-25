"""Unitree H1 homie (humanoid walk) environment configurations."""

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_homierl import mdp
from mjlab_homierl.homie_env_cfg import make_homie_env_cfg
from mjlab_homierl.mdp import RelativeHeightCommandCfg, UniformVelocityCommandCfg
from mjlab_homierl.robots import H1_ACTION_SCALE, get_h1_robot_cfg
from mjlab_homierl.robots.unitree_h1 import (
  DEFAULT_2F85_XML,
  HandMountCfg,
  HandsCfg,
)


class UpperBodyPoseAction(ActionTerm):
  """Generate smooth, policy-free targets for upper-body joints.

  The action has zero policy dimensions; it internally tracks a goal pose and
  interpolates toward it every simulation step. New goals are sampled via an
  event hook to mimic upper-body motion while keeping the lower-body action
  space small.
  """

  def __init__(self, cfg: "UpperBodyPoseActionCfg", env):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints(
      cfg.joint_names, preserve_order=cfg.preserve_order
    )
    if len(joint_ids) == 0:
      raise ValueError(
        f"No upper-body joints matched patterns: {cfg.joint_names}. "
        "Ensure the expressions align with the H1 MJCF joint names."
      )

    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._action_dim = 0
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    defaults = self._entity.data.default_joint_pos[:, self._joint_ids]
    self._default_target = defaults.clone()
    self._current_target = defaults.clone()
    self._goal_target = defaults.clone()
    self._joint_limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids]

    self._interp_rate = cfg.interp_rate
    self._max_speed_rad_s = cfg.max_speed_rad_s
    self._target_range = cfg.target_range
    self._start_step = cfg.start_step
    self._use_sampled_ratio = cfg.use_sampled_ratio
    # Curriculum-controlled scale; when use_sampled_ratio=True, this is max_ratio.
    self._curriculum_ratio = torch.tensor(
      cfg.initial_ratio, device=self.device, dtype=torch.float32
    )
    # Per-env sampled ratio (only used when use_sampled_ratio=True)
    self._sampled_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # No policy-controlled upper-body actions; keep buffer shape consistent.
    if actions.numel() != 0:
      raise ValueError(
        f"UpperBodyPoseAction expects zero-dim actions, got shape {actions.shape}."
      )
    self._raw_actions[:] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._current_target[env_ids] = self._default_target[env_ids]
    self._goal_target[env_ids] = self._default_target[env_ids]
    # Clamp curriculum ratio to valid bounds.
    self._curriculum_ratio = self._curriculum_ratio.clamp(0.0, 1.0)

  def sample_new_goals(
    self,
    env_ids: torch.Tensor | slice | None = None,
    target_range: tuple[float, float] | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)

    defaults = self._default_target[env_ids]
    low, high = target_range if target_range is not None else self._target_range
    # Base noise from symmetric range, then scale by ratio.
    noise = torch.empty_like(defaults).uniform_(low, high)

    if self._use_sampled_ratio:
      # Sample per-env ratio from [0, curriculum_ratio (as max)]
      if isinstance(env_ids, slice):
        n_envs = self.num_envs
        sampled = torch.empty(n_envs, device=self.device).uniform_(0.0, self._curriculum_ratio.item())
        self._sampled_ratio[:] = sampled
      else:
        n_envs = len(env_ids)
        sampled = torch.empty(n_envs, device=self.device).uniform_(0.0, self._curriculum_ratio.item())
        self._sampled_ratio[env_ids] = sampled
      # Use per-env sampled ratio
      ratio = self._sampled_ratio[env_ids].unsqueeze(1)  # [N, 1]
    else:
      # Use global fixed ratio
      ratio = self._curriculum_ratio

    noise = noise * ratio
    targets = defaults + noise

    limits = self._joint_limits[env_ids]
    targets = torch.max(torch.min(targets, limits[..., 1]), limits[..., 0])
    self._goal_target[env_ids] = targets

  def apply_actions(self) -> None:
    # Hold default targets until curriculum start.
    if self._env.common_step_counter < self._start_step:
      self._goal_target[:] = self._default_target

    # Smoothly move current targets toward goal targets.
    proposed = torch.lerp(self._current_target, self._goal_target, self._interp_rate)

    # Optional hard rate limit (rad/s) to avoid overly fast pose target changes.
    if self._max_speed_rad_s is not None:
      if self._max_speed_rad_s <= 0.0:
        raise ValueError(
          f"UpperBodyPoseActionCfg.max_speed_rad_s must be > 0, got {self._max_speed_rad_s}."
        )
      max_delta = float(self._max_speed_rad_s) * float(self._env.physics_dt)
      delta = proposed - self._current_target
      delta = torch.clamp(delta, min=-max_delta, max=max_delta)
      self._current_target = self._current_target + delta
    else:
      self._current_target = proposed

    self._entity.set_joint_position_target(
      self._current_target, joint_ids=self._joint_ids
    )

  def set_curriculum_ratio(self, ratio: torch.Tensor | float) -> None:
    """Update the global curriculum ratio, clamped to [0, 1]."""
    ratio_tensor = torch.as_tensor(ratio, device=self.device, dtype=torch.float32)
    self._curriculum_ratio = ratio_tensor.clamp(0.0, 1.0)

  @property
  def curriculum_ratio(self) -> torch.Tensor:
    return self._curriculum_ratio


@dataclass(kw_only=True)
class UpperBodyPoseActionCfg(ActionTermCfg):
  """Configuration for the smooth upper-body target generator."""

  joint_names: Sequence[str]
  interp_rate: float = 0.05
  max_speed_rad_s: float | None = None
  """Optional max change rate for joint position targets (rad/s).

  When set, the internal target generator will clamp per-step target changes to
  ``max_speed_rad_s * physics_dt`` to avoid abrupt upper-body motion that can
  destabilize training.
  """
  target_range: tuple[float, float] = (-0.35, 0.35)
  start_step: int = 0
  initial_ratio: float = 0.0
  preserve_order: bool = False
  use_sampled_ratio: bool = True  # If True, ratio is sampled from [0, max] each time

  def build(self, env) -> UpperBodyPoseAction:
    return UpperBodyPoseAction(self, env)


class GripperActuatorAction(ActionTerm):
  """Drive XML-defined gripper actuators with internal random targets (policy-free)."""

  def __init__(self, cfg: "GripperActuatorActionCfg", env):
    super().__init__(cfg=cfg, env=env)
    act_ids, act_names = self._entity.find_actuators(
      cfg.actuator_names, preserve_order=True
    )
    if len(act_ids) == 0:
      raise ValueError(
        f"No actuators matched for gripper action: {cfg.actuator_names}."
      )
    self._actuator_ids = torch.tensor(act_ids, device=self.device, dtype=torch.long)
    self._actuator_names = act_names
    self._action_dim = 0  # policy has zero dims; internal random targets
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    # Pre-compute ctrl ranges.
    ctrl_mins = []
    ctrl_maxs = []
    for idx in act_ids:
      ctrlrange = self._entity.spec.actuators[idx].ctrlrange
      if ctrlrange is None:
        ctrl_mins.append(-1.0)
        ctrl_maxs.append(1.0)
      else:
        ctrl_mins.append(ctrlrange[0])
        ctrl_maxs.append(ctrlrange[1])
    self._ctrl_min = torch.tensor(
      ctrl_mins, device=self.device, dtype=torch.float32
    ).unsqueeze(0)
    self._ctrl_max = torch.tensor(
      ctrl_maxs, device=self.device, dtype=torch.float32
    ).unsqueeze(0)

    self._start_step = cfg.start_step
    self._target_range = cfg.target_range
    self._interp_rate = cfg.interp_rate
    self._current = torch.zeros(self.num_envs, len(act_ids), device=self.device)
    self._goal = torch.zeros_like(self._current)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.numel() != 0:
      raise ValueError(
        f"GripperActuatorAction expects zero-dim actions, got shape {actions.shape}."
      )
    self._raw_actions[:] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._current[env_ids] = 0.0
    self._goal[env_ids] = 0.0

  def sample_new_goals(
    self,
    env_ids: torch.Tensor | slice | None = None,
    target_range: tuple[float, float] | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    low, high = target_range if target_range is not None else self._target_range
    noise = torch.empty_like(self._goal[env_ids]).uniform_(low, high)
    self._goal[env_ids] = torch.clamp(noise, 0.0, 1.0)

  def apply_actions(self) -> None:
    if self._env.common_step_counter < self._start_step:
      self._goal[:] = 0.0

    self._current = torch.lerp(self._current, self._goal, self._interp_rate)
    # Map [0,1] to ctrl range.
    ctrl = self._ctrl_min + (self._ctrl_max - self._ctrl_min) * self._current
    self._entity.write_ctrl_to_sim(ctrl, ctrl_ids=self._actuator_ids)


@dataclass(kw_only=True)
class GripperActuatorActionCfg(ActionTermCfg):
  actuator_names: Sequence[str]
  target_range: tuple[float, float] = (0.0, 1.0)
  interp_rate: float = 0.05
  start_step: int = 0

  def build(self, env) -> GripperActuatorAction:
    return GripperActuatorAction(self, env)


def _default_hands_cfg(enable: bool) -> HandsCfg | None:
  if not enable:
    return None
  return HandsCfg(
    left=HandMountCfg(
      enable=True,
      enable_collision=False,
      mount_site="left_hand_site",
      model=DEFAULT_2F85_XML,
      # Rotate gripper so its +Z (pinch direction) aligns with H1 forearm (+X).
      # Euler order is XYZ in MuJoCo.
      offset_euler=(1.5707963267948966, 1.5707963267948966, 0.0),
      add_wrist_joint=True,
      wrist_axis=(0.0, 0.0, 1.0),
      wrist_range=(-1.0, 1.0),
      wrist_ctrlrange=(-1.0, 1.0),
      actuator_whitelist=("fingers_actuator",),
    ),
    right=HandMountCfg(
      enable=True,
      enable_collision=False,
      mount_site="right_hand_site",
      model=DEFAULT_2F85_XML,
      offset_euler=(1.5707963267948966, 1.5707963267948966, 0.0),
      add_wrist_joint=True,
      wrist_axis=(0.0, 0.0, 1.0),
      wrist_range=(-1.0, 1.0),
      wrist_ctrlrange=(-1.0, 1.0),
      actuator_whitelist=("fingers_actuator",),
    ),
  )


def _sample_upper_body_targets_with_curriculum(
  env,
  env_ids,
  action_name: str,
  target_range: tuple[float, float],
  start_step: int = 0,
) -> None:
  if env.common_step_counter < start_step:
    return

  action_term = env.action_manager.get_term(action_name)
  if not isinstance(action_term, UpperBodyPoseAction):
    raise ValueError(
      f"Action term '{action_name}' is expected to be UpperBodyPoseAction; "
      f"received {type(action_term)}."
    )

  action_term.sample_new_goals(env_ids=env_ids, target_range=target_range)


def _sample_gripper_targets(
  env, env_ids, action_name: str, target_range: tuple[float, float], start_step: int = 0
):
  if env.common_step_counter < start_step:
    return
  action_term = env.action_manager.get_term(action_name)
  if not isinstance(action_term, GripperActuatorAction):
    raise ValueError(
      f"Action term '{action_name}' is expected to be GripperActuatorAction; "
      f"received {type(action_term)}."
    )
  action_term.sample_new_goals(env_ids=env_ids, target_range=target_range)


class _StepScheduledExternalPush:
  """Apply periodic horizontal pushes via MuJoCo external wrenches.

  This avoids modifying framework code by running every env step (via an
  interval event with range (0, 0)) and internally scheduling the actual push
  cadence in control steps.
  """

  def __init__(self, cfg: EventTermCfg, env):
    self._env = env
    params = cfg.params

    self._asset_cfg: SceneEntityCfg = params["asset_cfg"]
    self._push_interval_s = float(params.get("push_interval_s", 4.0))
    self._max_push_vel_xy = float(params.get("max_push_vel_xy", 0.5))
    self._push_duration_s = float(params.get("push_duration_s", env.step_dt))
    self._zero_push_prob = float(params.get("zero_push_prob", 0.0))

    self._interval_steps = max(1, math.ceil(self._push_interval_s / env.step_dt))
    self._duration_steps = max(1, math.ceil(self._push_duration_s / env.step_dt))

    asset = env.scene[self._asset_cfg.name]
    default_body_mass = env.sim.get_default_field("body_mass")
    total_mass = default_body_mass[asset.indexing.body_ids].sum().item()
    self._max_force_xy = total_mass * self._max_push_vel_xy / max(self._push_duration_s, 1e-6)

    self._active_forces: torch.Tensor | None = None
    self._clear_at_step: int | None = None

  def _write_wrench(self, forces: torch.Tensor) -> None:
    asset = self._env.scene[self._asset_cfg.name]
    torques = torch.zeros_like(forces)
    asset.write_external_wrench_to_sim(
      forces, torques, env_ids=None, body_ids=self._asset_cfg.body_ids
    )

  def __call__(self, env, env_ids, **_kwargs) -> None:
    del env, env_ids  # Unused; scheduling is global.

    # Convert to a zero-based step index (first call happens after step 0).
    step_index = int(self._env.common_step_counter) - 1

    if self._active_forces is not None:
      assert self._clear_at_step is not None
      if step_index >= self._clear_at_step:
        self._write_wrench(torch.zeros_like(self._active_forces))
        self._active_forces = None
        self._clear_at_step = None
      else:
        # Re-apply to survive per-env resets that clear xfrc_applied.
        self._write_wrench(self._active_forces)

    if step_index % self._interval_steps != 0:
      return

    num_envs = self._env.num_envs
    device = self._env.device
    forces = torch.zeros((num_envs, 1, 3), device=device, dtype=torch.float32)
    forces[:, 0, 0] = torch.empty((num_envs,), device=device).uniform_(
      -self._max_force_xy, self._max_force_xy
    )
    forces[:, 0, 1] = torch.empty((num_envs,), device=device).uniform_(
      -self._max_force_xy, self._max_force_xy
    )

    if self._zero_push_prob > 0.0:
      zero_mask = torch.rand(num_envs, device=device) < self._zero_push_prob
      forces[zero_mask] = 0.0

    self._write_wrench(forces)
    self._active_forces = forces
    self._clear_at_step = step_index + self._duration_steps


class _HandLoadExternalWrench:
  """Apply a constant downward load on each hand via external wrenches.

  The load is specified as an equivalent mass in kilograms and converted to a
  world-frame force using gravity. Values are sampled once per environment and
  persist across episode resets (re-applied on reset since MuJoCo clears forces).
  """

  def __init__(self, cfg: EventTermCfg, env):
    self._env = env
    params = cfg.params

    self._asset_cfg: SceneEntityCfg = params["asset_cfg"]
    self._mass_range = tuple(params.get("mass_range", (0.0, 5.0)))
    self._zero_load_prob = float(params.get("zero_load_prob", 0.0))

    try:
      self._g = float(abs(env.sim.mj_model.opt.gravity[2]))
    except Exception:
      self._g = 9.81

    self._mass_kg = torch.full(
      (env.num_envs, 2), float("nan"), device=env.device, dtype=torch.float32
    )

  def __call__(self, env, env_ids, **_kwargs) -> None:
    del env  # Unused; use self._env.
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.int)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.int)
    else:
      env_ids = env_ids.to(device=self._env.device, dtype=torch.int)

    # Lazily sample once (respects env seeding done before reset events).
    existing = self._mass_kg[env_ids]
    needs_sample = torch.isnan(existing).any(dim=1)
    if needs_sample.any():
      ids_to_sample = env_ids[needs_sample]
      low, high = float(self._mass_range[0]), float(self._mass_range[1])
      samples = torch.empty(
        (len(ids_to_sample), 2), device=self._env.device, dtype=torch.float32
      ).uniform_(low, high)

      if self._zero_load_prob > 0.0:
        # Independent probability for each hand
        zero_mask = (
          torch.rand(samples.shape, device=self._env.device) < self._zero_load_prob
        )
        samples[zero_mask] = 0.0

      self._mass_kg[ids_to_sample] = samples

    mass_kg = self._mass_kg[env_ids]
    forces = torch.zeros((len(env_ids), 2, 3), device=self._env.device, dtype=torch.float32)
    forces[:, :, 2] = -mass_kg * self._g
    torques = torch.zeros_like(forces)

    asset = self._env.scene[self._asset_cfg.name]
    asset.write_external_wrench_to_sim(
      forces, torques, env_ids=env_ids, body_ids=self._asset_cfg.body_ids
    )


def unitree_h1_homie_env_cfg(
  play: bool = False,
  curriculum_start_step: int = 0,
  hands: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 homie (humanoid walk) task configuration.

  The homie task reduces the policy action space to the lower body. Upper-body
  joints (torso + arms) follow smooth, randomly sampled pose targets updated via
  a curriculum. This setup mirrors HOMIE: legs are policy-controlled, while the
  upper body moves through slow, interpolated targets to introduce realistic
  disturbances without enlarging the action space.
  """
  # Start from the base homie configuration.
  cfg = make_homie_env_cfg()
  # HOMIE runs on flat terrain, and the with-hands variant now disables gripper
  # collision geoms. The default CCD budget is sufficient here and matches
  # official flat-humanoid configs more closely.
  cfg.sim.mujoco.ccd_iterations = 50

  cfg.scene.entities = {"robot": get_h1_robot_cfg(hands=_default_hands_cfg(hands))}

  site_names = ("left_foot", "right_foot")
  # H1 foot collision geoms
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 4)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_link|right_ankle_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Hip and knee ground contact sensor for penalizing unwanted contacts
  hip_knee_ground_cfg = ContactSensorCfg(
    name="hip_knee_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left_hip_yaw_link|right_hip_yaw_link|left_hip_roll_link|right_hip_roll_link|left_hip_pitch_link|right_hip_pitch_link|left_knee_link|right_knee_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, hip_knee_ground_cfg)


  # Switch to flat terrain for homie task.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  if "terrain_levels" in cfg.curriculum:
    del cfg.curriculum["terrain_levels"]

  step_threshold = 0 if play else curriculum_start_step

  # Split actions: lower body is policy-controlled; upper body follows smooth targets.
  lower_body_joint_expr: tuple[str, ...] = (
    r"(left|right)_hip_yaw",
    r"(left|right)_hip_roll",
    r"(left|right)_hip_pitch",
    r"(left|right)_knee",
    r"(left|right)_ankle",
  )
  upper_body_joint_expr: tuple[str, ...] = (
    "torso",
    r"(left|right)_shoulder_pitch",
    r"(left|right)_shoulder_roll",
    r"(left|right)_shoulder_yaw",
    r"(left|right)_elbow",
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = lower_body_joint_expr
  # Limit scale dict to only lower-body joints to avoid regex match errors for upper-body patterns.
  lower_scale_keys = (
    ".*_hip_yaw",
    ".*_hip_roll",
    ".*_hip_pitch",
    ".*_knee",
    ".*_ankle",
  )
  joint_pos_action.scale = {k: H1_ACTION_SCALE[k] for k in lower_scale_keys}

  upper_action_name = "upper_body_pose"
  cfg.actions[upper_action_name] = UpperBodyPoseActionCfg(
    entity_name="robot",
    joint_names=upper_body_joint_expr,
    interp_rate=0.05,
    max_speed_rad_s=1.0,
    target_range=(-0.6, 0.6),
    start_step=step_threshold,
    initial_ratio=1.0 if play else 0.0,
  )

  # Optional gripper action (policy-free, random clamp).
  if hands:
    gripper_action_name = "gripper"
    cfg.actions[gripper_action_name] = GripperActuatorActionCfg(
      entity_name="robot",
      actuator_names=(r".*fingers_actuator.*",),
      target_range=(0.0, 1.0),
      interp_rate=0.05,
      start_step=step_threshold,
    )
    # Exclude gripper/wrist joints from posture reward to keep std shape consistent.
    pose_reward = cfg.rewards["pose"]
    pose_asset_cfg = pose_reward.params["asset_cfg"]
    pose_asset_cfg.joint_names = (
      r"(left|right)_hip_yaw",
      r"(left|right)_hip_roll",
      r"(left|right)_hip_pitch",
      r"(left|right)_knee",
      r"(left|right)_ankle",
      "torso",
      r"(left|right)_shoulder_pitch",
      r"(left|right)_shoulder_roll",
      r"(left|right)_shoulder_yaw",
      r"(left|right)_elbow",
    )
    cfg.events["gripper_random_targets"] = EventTermCfg(
      func=_sample_gripper_targets,
      mode="interval",
      interval_range_s=(0.75, 1.25),
      params={
        "action_name": gripper_action_name,
        "target_range": (0.0, 1.0),
        "start_step": step_threshold,
      },
    )

  # Update upper-body targets on a slower schedule to simulate purposeful motion.
  cfg.events["upper_body_random_targets"] = EventTermCfg(
    func=_sample_upper_body_targets_with_curriculum,
    mode="interval",
    interval_range_s=(2.0, 2.0),
    params={
      "action_name": upper_action_name,
      "target_range": (-3.0, 1.0),
      "start_step": step_threshold,
    },
  )

  cfg.viewer.body_name = "torso_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0  # H1 is taller than G1

  height_cmd = cfg.commands["height"]
  assert isinstance(height_cmd, RelativeHeightCommandCfg)
  height_cmd.foot_site_names = site_names
  # Squat range: target pelvis height relative to lowest foot (approx. "pelvis above ground").
  # Note: for H1 the `left_foot/right_foot` sites sit ~1-2cm above the lowest foot collision,
  # so `h_rel=0.33` roughly corresponds to pelvis ~0.34-0.35m above the actual contact plane.
  height_cmd.ranges.height = (0.4, 0.98)
  # For non-squat envs (velocity group), keep a stable standing target to avoid confusing the policy.
  height_cmd.inactive_height = 0.98

  cfg.rewards["track_height"].params["asset_cfg"].site_names = site_names
  cfg.rewards["knee_deviation"].params["foot_asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Random push: apply short horizontal external force pulses to the base.
  cfg.events["push_robot"] = EventTermCfg(
    func=_StepScheduledExternalPush,
    mode="interval",
    interval_range_s=(0.0, 0.0),  # Run every step; internal scheduler applies cadence.
    is_global_time=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",), preserve_order=True),
      "push_interval_s": 4.0,
      "max_push_vel_xy": 0.5,
      "zero_push_prob": 0.3,
    },
  )

  # Hand load: constant downward wrench equivalent to a 0-5kg payload per hand.
  hand_body_names = (
    ("left_wrist_link", "right_wrist_link")
    if hands
    else ("left_elbow_link", "right_elbow_link")
  )
  cfg.events["hand_load"] = EventTermCfg(
    func=_HandLoadExternalWrench,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=hand_body_names, preserve_order=True
      ),
      "mass_range": (0.0, 5.0),
      "zero_load_prob": 0.3,
    },
  )

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body (H1 has simpler leg structure than G1)
    r".*hip_yaw": 0.15,
    r".*hip_roll": 0.15,
    r".*hip_pitch": 0.3,
    r".*knee": 0.35,
    r".*ankle": 0.25,
    # Torso (H1 has single torso joint)
    "torso": 0.2,
    # Arms (H1 has simpler arm structure)
    r".*shoulder_pitch": 0.15,
    r".*shoulder_roll": 0.15,
    r".*shoulder_yaw": 0.1,
    r".*elbow": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body
    r".*hip_yaw": 0.2,
    r".*hip_roll": 0.2,
    r".*hip_pitch": 0.5,
    r".*knee": 0.6,
    r".*ankle": 0.35,
    # Torso
    "torso": 0.3,
    # Arms
    r".*shoulder_pitch": 0.5,
    r".*shoulder_roll": 0.2,
    r".*shoulder_yaw": 0.15,
    r".*elbow": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.05

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )
  # Wire the hip/knee contact penalty sensor
  cfg.rewards["hip_knee_contact"].params["sensor_name"] = hip_knee_ground_cfg.name


  # Configure feet parallel rewards.
  #
  # We use four corner sites per foot:
  #   fi = front-inner, fo = front-outer, ri = rear-inner, ro = rear-outer
  # These are defined in the H1 MJCF under each ankle link.
  left_foot_corner_sites = (
    "left_foot_fi",
    "left_foot_fo",
    "left_foot_ri",
    "left_foot_ro",
  )
  right_foot_corner_sites = (
    "right_foot_fi",
    "right_foot_fo",
    "right_foot_ri",
    "right_foot_ro",
  )

  cfg.rewards["feet_ground_parallel"].params["left_foot_sites"] = left_foot_corner_sites
  cfg.rewards["feet_ground_parallel"].params["right_foot_sites"] = (
    right_foot_corner_sites
  )

  # feet_parallel expects corresponding left/right sites to have the same local-frame
  # coordinates. Since "inner/outer" flips between left and right feet in +Y/-Y, we
  # reorder the right-foot corner sites to match local Y signs.
  cfg.rewards["feet_parallel"].params["left_foot_sites"] = left_foot_corner_sites
  cfg.rewards["feet_parallel"].params["right_foot_sites"] = (
    "right_foot_fo",
    "right_foot_fi",
    "right_foot_ro",
    "right_foot_ri",
  )

  # Curriculum: expand upper-body motion as velocity tracking improves.
  cfg.curriculum["upper_body_action"] = CurriculumTermCfg(
    func=mdp.upper_body_action_curriculum,
    params={
      "action_name": upper_action_name,
      "reward_name": "track_linear_velocity",
      "success_threshold": 0.8,
      "increment": 0.05,
      "max_ratio": 1.0,
      "start_step": step_threshold,
    },
  )

  # Ensure total_reward_clipping is always the last term.
  if "total_reward_clipping" in cfg.rewards:
    clipping_term = cfg.rewards.pop("total_reward_clipping")
    cfg.rewards["total_reward_clipping"] = clipping_term

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    # HOMIE play uses a custom actor-only inference path, so critic
    # observations, rewards, and curriculum are all unnecessary overhead.
    cfg.observations.pop("critic", None)
    cfg.rewards = {}
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    cfg.events.pop("hand_load", None)

    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.8, 1.2)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
