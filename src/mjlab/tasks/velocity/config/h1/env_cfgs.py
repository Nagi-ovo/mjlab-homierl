"""Unitree H1 velocity environment configurations."""

from dataclasses import dataclass
from typing import Sequence

import torch

from mjlab.asset_zoo.robots import H1_ACTION_SCALE, get_h1_robot_cfg
from mjlab.asset_zoo.robots.unitree_h1.h1_constants import (
  DEFAULT_2F85_XML,
  HandMountCfg,
  HandsCfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  RewardTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


class UpperBodyPoseAction(ActionTerm):
  """Generate smooth, policy-free targets for upper-body joints.

  The action has zero policy dimensions; it internally tracks a goal pose and
  interpolates toward it every simulation step. New goals are sampled via an
  event hook to mimic upper-body motion while keeping the lower-body action
  space small.
  """

  def __init__(self, cfg: "UpperBodyPoseActionCfg", env):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._asset.find_joints(
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

    defaults = self._asset.data.default_joint_pos[:, self._joint_ids]
    self._default_target = defaults.clone()
    self._current_target = defaults.clone()
    self._goal_target = defaults.clone()
    self._joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]

    self._interp_rate = cfg.interp_rate
    self._target_range = cfg.target_range
    self._start_step = cfg.start_step
    # Curriculum-controlled global scale in [0, 1]; 0 => no upper-body motion.
    self._curriculum_ratio = torch.tensor(
      cfg.initial_ratio, device=self.device, dtype=torch.float32
    )

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
    # Base noise from symmetric range, then scale by global curriculum ratio.
    noise = torch.empty_like(defaults).uniform_(low, high)
    noise = noise * self._curriculum_ratio
    targets = defaults + noise

    limits = self._joint_limits[env_ids]
    targets = torch.max(torch.min(targets, limits[..., 1]), limits[..., 0])
    self._goal_target[env_ids] = targets

  def apply_actions(self) -> None:
    # Hold default targets until curriculum start.
    if self._env.common_step_counter < self._start_step:
      self._goal_target[:] = self._default_target

    self._current_target = torch.lerp(
      self._current_target, self._goal_target, self._interp_rate
    )
    self._asset.set_joint_position_target(
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
  target_range: tuple[float, float] = (-0.35, 0.35)
  start_step: int = 0
  initial_ratio: float = 0.0
  preserve_order: bool = False
  class_type: type[ActionTerm] = UpperBodyPoseAction


class GripperActuatorAction(ActionTerm):
  """Drive XML-defined gripper actuators with internal random targets (policy-free)."""

  def __init__(self, cfg: "GripperActuatorActionCfg", env):
    super().__init__(cfg=cfg, env=env)
    act_ids, act_names = self._asset.find_actuators(cfg.actuator_names, preserve_order=True)
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
      ctrlrange = self._asset.spec.actuators[idx].ctrlrange
      if ctrlrange is None:
        ctrl_mins.append(-1.0)
        ctrl_maxs.append(1.0)
      else:
        ctrl_mins.append(ctrlrange[0])
        ctrl_maxs.append(ctrlrange[1])
    self._ctrl_min = torch.tensor(ctrl_mins, device=self.device, dtype=torch.float32).unsqueeze(0)
    self._ctrl_max = torch.tensor(ctrl_maxs, device=self.device, dtype=torch.float32).unsqueeze(0)

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
    self._asset.write_ctrl_to_sim(ctrl, ctrl_ids=self._actuator_ids)


@dataclass(kw_only=True)
class GripperActuatorActionCfg(ActionTermCfg):
  actuator_names: Sequence[str]
  target_range: tuple[float, float] = (0.0, 1.0)
  interp_rate: float = 0.05
  start_step: int = 0
  class_type: type[ActionTerm] = GripperActuatorAction


def _default_hands_cfg(enable: bool) -> HandsCfg | None:
  if not enable:
    return None
  return HandsCfg(
    left=HandMountCfg(
      enable=True,
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


def unitree_h1_rough_env_cfg(
  play: bool = False, hands: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0  # H1 is taller than G1

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

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
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_h1_flat_env_cfg(
  play: bool = False, hands: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 flat terrain velocity configuration."""
  cfg = unitree_h1_rough_env_cfg(play=play, hands=hands)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


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


def _sample_gripper_targets(env, env_ids, action_name: str, target_range: tuple[float, float], start_step: int = 0):
  if env.common_step_counter < start_step:
    return
  action_term = env.action_manager.get_term(action_name)
  if not isinstance(action_term, GripperActuatorAction):
    raise ValueError(
      f"Action term '{action_name}' is expected to be GripperActuatorAction; "
      f"received {type(action_term)}."
    )
  action_term.sample_new_goals(env_ids=env_ids, target_range=target_range)


def unitree_h1_walk_env_cfg(
  play: bool = False,
  curriculum_start_step: int = 500 * 24,
  hands: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 walk task with upper-body target curriculum.

  Based on ``unitree_h1_flat_env_cfg`` but reduces the policy action space to
  the lower body. Upper-body joints (torso + arms) follow smooth, randomly
  sampled pose targets that are updated via a curriculum. This mirrors the
  HOMIE setup: legs are policy-controlled, while the upper body moves through
  slow, interpolated targets to introduce realistic disturbances without
  enlarging the action space.
  """
  # Start from the standard flat velocity configuration.
  cfg = unitree_h1_flat_env_cfg(play=play, hands=hands)

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
  # 限定缩放表只包含下肢关节，避免上肢模式缺失造成的正则匹配报错
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
    asset_name="robot",
    joint_names=upper_body_joint_expr,
    interp_rate=0.05,
    target_range=(-0.6, 0.6),
    start_step=step_threshold,
    initial_ratio=1.0 if play else 0.0,
  )

  # Optional gripper action (policy-free, random clamp).
  if hands:
    gripper_action_name = "gripper"
    cfg.actions[gripper_action_name] = GripperActuatorActionCfg(
      asset_name="robot",
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
    interval_range_s=(0.75, 1.25),
    params={
      "action_name": upper_action_name,
      "target_range": (-1.2, 0.3),
      "start_step": step_threshold,
    },
  )

  # Curriculum: expand upper-body motion as velocity tracking improves.
  assert cfg.curriculum is not None
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

  return cfg
