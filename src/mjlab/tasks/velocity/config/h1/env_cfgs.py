"""Unitree H1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  H1_ACTION_SCALE,
  get_h1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_h1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_h1_robot_cfg()}

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


def unitree_h1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 flat terrain velocity configuration."""
  cfg = unitree_h1_rough_env_cfg(play=play)

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


def _upper_body_random_disturbance_with_curriculum(
  env,
  env_ids,
  force_range: tuple[float, float],
  torque_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  start_step: int = 0,
) -> None:
  """Apply random external wrenches to H1 upper body after a curriculum warmup.

  This wraps ``mdp.apply_external_force_torque`` but keeps the first
  ``start_step`` environment steps disturbance–free so the policy can first
  learn nominal walking, then adapt to upper–body motion.
  """
  # env.common_step_counter is a global training step counter incremented every
  # environment step; it is shared across all env instances.
  if env.common_step_counter < start_step:
    return

  mdp.apply_external_force_torque(
    env,
    env_ids,
    force_range=force_range,
    torque_range=torque_range,
    asset_cfg=asset_cfg,
  )


def unitree_h1_walk_env_cfg(
  play: bool = False,
  curriculum_start_step: int = 1_000 * 24,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 flat walk task with upper-body disturbance curriculum.

  The configuration is based on ``unitree_h1_flat_env_cfg`` (velocity tracking
  on flat terrain) and adds an event that applies random external wrenches to
  the torso and arm links. For training, the disturbance is disabled for the
  first ``curriculum_start_step`` environment steps (default: ``1000 * 24``,
  mirroring the velocity task curricula) so the robot can learn stable walking
  before adapting to random upper–body motion. In play mode the disturbance is
  enabled from the beginning.
  """
  # Start from the standard flat velocity configuration.
  cfg = unitree_h1_flat_env_cfg(play=play)

  # Configure which bodies receive external disturbances (torso + arms).
  upper_body_asset_cfg = SceneEntityCfg(
    "robot",
    body_names=(
      "torso_link",
      "left_shoulder_pitch_link",
      "left_shoulder_roll_link",
      "left_shoulder_yaw_link",
      "left_elbow_link",
      "right_shoulder_pitch_link",
      "right_shoulder_roll_link",
      "right_shoulder_yaw_link",
      "right_elbow_link",
    ),
  )

  # Add an interval event that injects random forces/torques on the upper body.
  # The strength is modest so that the task remains solvable but non-trivial.
  step_threshold = 0 if play else curriculum_start_step

  cfg.events["upper_body_random_disturbance"] = EventTermCfg(
    func=_upper_body_random_disturbance_with_curriculum,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "force_range": (-60.0, 60.0),
      "torque_range": (-30.0, 30.0),
      "asset_cfg": upper_body_asset_cfg,
      "start_step": step_threshold,
    },
  )

  return cfg
