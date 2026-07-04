"""Robot-specific HOMIE environment configurations (Unitree G1 and H1)."""

import os

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_homierl import mdp
from mjlab_homierl.homie_env_cfg import make_him_observations, make_homie_env_cfg
from mjlab_homierl.mdp import RelativeHeightCommandCfg, UniformVelocityCommandCfg
from mjlab_homierl.robots import get_h1_robot_cfg
from mjlab_homierl.robots.inspire_rh56 import (
  INSPIRE_MOUNT_BODY_PATTERN,
  attach_inspire_hands,
)
from mjlab_homierl.robots.unitree_dex3 import (
  DEX3_MOUNT_BODY_PATTERN,
  attach_dex3_hands,
)
from mjlab_homierl.robots.unitree_g1_deploy import (
  G1_DEPLOY_ACTION_SCALE,
  G1_DEPLOY_PD_GAINS,
  get_g1_deploy_robot_cfg,
)
from mjlab_homierl.robots.unitree_h1 import (
  DEFAULT_2F85_XML,
  HandMountCfg,
  HandsCfg,
  h1_constants,
)
from mjlab_homierl.robots.unitree_h1_deploy import (
  H1_DEPLOY_ACTION_SCALE,
  H1_DEPLOY_PD_GAINS,
  get_h1_deploy_robot_cfg,
)

##
# Shared helpers.
##


def _make_contact_sensors(
  feet_link_pattern: str, hip_knee_pattern: str
) -> tuple[ContactSensorCfg, ...]:
  """Contact sensors shared by all HOMIE robots."""
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=feet_link_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  hip_knee_ground = ContactSensorCfg(
    name="hip_knee_ground_contact",
    primary=ContactMatch(mode="body", pattern=hip_knee_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  return (feet_ground, self_collision, hip_knee_ground)


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Strip training-only machinery for actor-only play/inference."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  # HOMIE play uses an actor-only inference path: critic observations, rewards,
  # and curriculum are unnecessary overhead.
  cfg.observations.pop("critic", None)
  cfg.rewards = {}
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)

  # Upper-body disturbance amplitude during play. The play env has no
  # curriculum, so this ratio stays fixed. Default 0.0 holds the default pose
  # (isolates lower-body gait for inspection); the upstream play CLI exposes
  # no env overrides, so the value can be set via an environment variable:
  #   HOMIE_PLAY_UPPER_RATIO=1.0 uv run play ...   # deployment-like disturbance
  upper = cfg.actions["upper_body_pose"]
  assert isinstance(upper, mdp.UpperBodyPoseActionCfg)
  upper.initial_ratio = float(os.environ.get("HOMIE_PLAY_UPPER_RATIO", "0.0"))


##
# Unitree G1 (29 dof, 12 lower-body actions).
##

G1_LOWER_BODY_JOINTS: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

G1_UPPER_BODY_JOINTS: tuple[str, ...] = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_ALL_JOINTS: tuple[str, ...] = G1_LOWER_BODY_JOINTS + G1_UPPER_BODY_JOINTS

# Standing pelvis height above the foot sites at the HOME keyframe (measured).
G1_STANDING_HEIGHT = 0.78
# Squat depth of 0.5 m, matching OpenHomie's height command range on G1.
G1_HEIGHT_RANGE = (0.28, 0.78)
# Standing-gate threshold: just below the standing command (OpenHomie margin).
G1_STANDING_GATE = G1_STANDING_HEIGHT - 0.005
# Clearance gate keeps OpenHomie's 0.03 margin below standing.
G1_CLEARANCE_GATE = G1_STANDING_HEIGHT - 0.03

# Lower-body stiffness used to normalize the ``torques`` reward; must match
# the actuator gains of the selected variant.
G1_MJLAB_LOWER_STIFFNESS = {
  ".*_hip_pitch_joint": g1_constants.STIFFNESS_7520_14,
  ".*_hip_yaw_joint": g1_constants.STIFFNESS_7520_14,
  ".*_hip_roll_joint": g1_constants.STIFFNESS_7520_22,
  ".*_knee_joint": g1_constants.STIFFNESS_7520_22,
  ".*_ankle_.*_joint": g1_constants.STIFFNESS_5020 * 2,
}
G1_DEPLOY_LOWER_STIFFNESS = {
  pattern: kp
  for pattern, (kp, _) in G1_DEPLOY_PD_GAINS.items()
  if ("hip" in pattern or "knee" in pattern or "ankle" in pattern)
}
G1_LOWER_VELOCITY_LIMITS = {
  ".*_hip_pitch_joint": g1_constants.ACTUATOR_7520_14.velocity_limit,
  ".*_hip_yaw_joint": g1_constants.ACTUATOR_7520_14.velocity_limit,
  ".*_hip_roll_joint": g1_constants.ACTUATOR_7520_22.velocity_limit,
  ".*_knee_joint": g1_constants.ACTUATOR_7520_22.velocity_limit,
  ".*_ankle_.*_joint": g1_constants.ACTUATOR_5020.velocity_limit,
}
G1_LOWER_EFFORT_LIMITS = {
  ".*_hip_pitch_joint": g1_constants.ACTUATOR_7520_14.effort_limit,
  ".*_hip_yaw_joint": g1_constants.ACTUATOR_7520_14.effort_limit,
  ".*_hip_roll_joint": g1_constants.ACTUATOR_7520_22.effort_limit,
  ".*_knee_joint": g1_constants.ACTUATOR_7520_22.effort_limit,
  ".*_ankle_.*_joint": g1_constants.ACTUATOR_5020.effort_limit * 2,
}


def unitree_g1_homie_env_cfg(
  play: bool = False,
  curriculum_start_step: int = 0,
  gains: str = "deploy",
  hands: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 HOMIE task configuration.

  Args:
    gains: PD-gain variant.
      - ``"deploy"`` (default): deployment-grade gains matching HomieDeploy's
        real-robot low-level controller, with the uniform 0.25 action scale
        used by the deployed inference pipeline. Use this for sim2real.
      - ``"mjlab"``: mjlab asset-zoo first-principles gains (armature x
        natural-frequency) with per-joint effort/stiffness action scales.
        Sim-only / ablation variant.
    hands: Mount real hand models as inertial attachments and randomize an
      additional held-object payload. The observation/action interface is
      unchanged, so checkpoints remain compatible with the base task.
      - ``"dex3"``: Unitree Dex3 (~0.53 kg each; BiGym's G1 is G1-Dex3).
      - ``"inspire"``: Inspire RH56 (RH56DFX spec weight, 0.54 kg each).
  """
  if gains not in ("deploy", "mjlab"):
    raise ValueError(f"Unknown gains variant '{gains}'. Use 'deploy' or 'mjlab'.")
  cfg = make_homie_env_cfg()

  # Robot: mjlab asset-zoo G1 with the standing HOME keyframe as the default
  # pose (OpenHomie squats are commanded relative to standing).
  if gains == "deploy":
    robot_cfg = get_g1_deploy_robot_cfg()
  else:
    robot_cfg = g1_constants.get_g1_robot_cfg()
    robot_cfg.init_state = g1_constants.HOME_KEYFRAME

  # Wrist payload DR, sampled independently per hand: makes the lower-body
  # policy hand-agnostic. OpenHomie's reference config randomizes a hand
  # payload of [-0.1, 0.3] kg on its bare-wrist G1; the wider (0, 1.5) kg
  # envelope also covers mounted hands (Dex3 0.53 kg, Inspire RH56DFX
  # 0.54 kg) plus a held object, so one policy serves every hand option.
  cfg.events["hand_payload"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r".*_wrist_yaw_link",)),
      "operation": "add",
      "ranges": (0.0, 1.5),
    },
  )

  if hands is not None:
    try:
      attach_fn, mount_pattern = {
        "dex3": (attach_dex3_hands, DEX3_MOUNT_BODY_PATTERN),
        "inspire": (attach_inspire_hands, INSPIRE_MOUNT_BODY_PATTERN),
      }[hands]
    except KeyError:
      raise ValueError(
        f"Unknown hands variant '{hands}'. Use 'dex3' or 'inspire'."
      ) from None
    robot_cfg.spec_fn = lambda: attach_fn(g1_constants.get_spec())
    # Replaces the bare-wrist payload DR: the mounted hand supplies its real
    # mass, and the randomized remainder models a held object.
    cfg.events["hand_payload"] = EventTermCfg(
      mode="startup",
      func=mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(mount_pattern,)),
        "operation": "add",
        "ranges": (0.0, 1.0),
      },
    )
  cfg.scene.entities = {"robot": robot_cfg}

  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.sensors = _make_contact_sensors(
    feet_link_pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
    hip_knee_pattern=r"^(left|right)_(hip_(pitch|roll|yaw)|knee)_link$",
  ) + (
    ContactSensorCfg(
      name="torso_ground_contact",
      primary=ContactMatch(
        mode="body", pattern=r"^(torso_link|pelvis)$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  # OpenHomie terminates on torso contact.
  cfg.terminations["torso_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "torso_ground_contact"},
  )

  # Observations.
  cfg.observations = make_him_observations(
    joint_names=G1_ALL_JOINTS, num_actions=len(G1_LOWER_BODY_JOINTS)
  )

  # Actions.
  step_threshold = 0 if play else curriculum_start_step
  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = G1_LOWER_BODY_JOINTS
  if gains == "deploy":
    joint_pos.scale = G1_DEPLOY_ACTION_SCALE
  else:
    joint_pos.scale = {
      k: v
      for k, v in g1_constants.G1_ACTION_SCALE.items()
      if ("hip" in k or "knee" in k or "ankle" in k)
    }
  upper = cfg.actions["upper_body_pose"]
  assert isinstance(upper, mdp.UpperBodyPoseActionCfg)
  upper.joint_names = G1_UPPER_BODY_JOINTS
  cfg.events["upper_body_goals"].params["start_step"] = step_threshold
  cfg.curriculum["upper_body_action"].params["start_step"] = step_threshold

  # Commands.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.15
  height = cfg.commands["height"]
  assert isinstance(height, RelativeHeightCommandCfg)
  height.foot_site_names = ("left_foot", "right_foot")
  height.standing_height = G1_STANDING_HEIGHT
  height.ranges.height = G1_HEIGHT_RANGE

  # Events.
  foot_geoms = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["payload_mass"].params["asset_cfg"].body_names = ("torso_link",)

  # Rewards: wire robot-specific parameters.
  site_names = ("left_foot", "right_foot")
  for name in (
    "track_height",
    "feet_clearance",
    "feet_slip",
    "contact_momentum",
    "feet_distance_lateral",
  ):
    cfg.rewards[name].params["asset_cfg"].site_names = site_names
  cfg.rewards["knee_deviation"].params["foot_asset_cfg"].site_names = site_names

  for name in ("lin_vel_z", "deviation_hip_joint", "deviation_ankle_joint"):
    cfg.rewards[name].params["min_height"] = G1_STANDING_GATE
  for name in (
    "feet_parallel",
    "feet_distance_lateral",
    "knee_distance_lateral",
    "stand_still",
  ):
    cfg.rewards[name].params["min_height"] = G1_STANDING_GATE
  cfg.rewards["feet_clearance"].params["min_height"] = G1_CLEARANCE_GATE
  cfg.rewards["feet_clearance"].params["target_height"] = 0.14

  lower_cfg = SceneEntityCfg("robot", joint_names=G1_LOWER_BODY_JOINTS)
  for name in (
    "dof_pos_limits",
    "torques",
    "dof_vel",
    "dof_vel_limits",
    "torque_limits",
    "joint_tracking_error",
  ):
    cfg.rewards[name].params["asset_cfg"] = lower_cfg
  cfg.rewards["torques"].params["stiffness"] = (
    G1_DEPLOY_LOWER_STIFFNESS if gains == "deploy" else G1_MJLAB_LOWER_STIFFNESS
  )
  cfg.rewards["dof_vel_limits"].params["velocity_limits"] = G1_LOWER_VELOCITY_LIMITS
  cfg.rewards["torque_limits"].params["effort_limits"] = G1_LOWER_EFFORT_LIMITS

  # OpenHomie measures foot-parallelism on sampled foot surface points; the
  # mjlab G1 has no per-corner sites, so the foot collision spheres are used.
  cfg.rewards["feet_ground_parallel"].params.update(
    left_foot_points=tuple(f"left_foot{i}_collision" for i in range(1, 8)),
    right_foot_points=tuple(f"right_foot{i}_collision" for i in range(1, 8)),
    point_type="geom",
  )
  cfg.rewards["feet_parallel"].params.update(
    left_foot_points=tuple(f"left_foot{i}_collision" for i in range(1, 8)),
    right_foot_points=tuple(f"right_foot{i}_collision" for i in range(1, 8)),
    point_type="geom",
  )
  # OpenHomie uses (knee, hip_yaw) link pairs for the lateral knee distance.
  cfg.rewards["knee_distance_lateral"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    body_names=(
      "left_knee_link",
      "left_hip_yaw_link",
      "right_knee_link",
      "right_hip_yaw_link",
    ),
    preserve_order=True,
  )

  # OpenHomie G1 trains with self-collision disabled (IsaacGym
  # ``self_collision = 1``) and has no self-collision penalty in its reward
  # scales; leg separation is shaped by the lateral-distance terms instead.
  # On the G1 the hanging wrists rest against the hip links (wrist kp 5-10
  # cannot hold them clear), so a self-collision penalty fires on ~25-30% of
  # steps at the default pose and intensifies as the thighs rise -- an
  # anti-squat gradient that walled the 2026-07-03 run at ~0.67 m. Physical
  # self-contacts remain simulated; only the penalty is dropped.
  del cfg.rewards["self_collisions"]

  cfg.viewer.body_name = "torso_link"

  if play:
    _apply_play_overrides(cfg)

  return cfg


##
# Unitree H1 (19 dof, 10 lower-body actions).
##

H1_LOWER_BODY_JOINTS: tuple[str, ...] = (
  "left_hip_yaw",
  "left_hip_roll",
  "left_hip_pitch",
  "left_knee",
  "left_ankle",
  "right_hip_yaw",
  "right_hip_roll",
  "right_hip_pitch",
  "right_knee",
  "right_ankle",
)

H1_UPPER_BODY_JOINTS: tuple[str, ...] = (
  "torso",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
)

H1_ALL_JOINTS: tuple[str, ...] = H1_LOWER_BODY_JOINTS + H1_UPPER_BODY_JOINTS

# H1 stands taller than G1: standing height and squat range scale accordingly.
H1_STANDING_HEIGHT = 0.98
H1_HEIGHT_RANGE = (0.4, 0.98)
H1_STANDING_GATE = H1_STANDING_HEIGHT - 0.005
H1_CLEARANCE_GATE = H1_STANDING_HEIGHT - 0.03

# Lower-body stiffness used to normalize the ``torques`` reward; must match
# the actuator gains of the selected variant.
H1_MJLAB_LOWER_STIFFNESS = {
  ".*_hip_.*": h1_constants.STIFFNESS_HIP_KNEE,
  ".*_knee": h1_constants.STIFFNESS_HIP_KNEE,
  ".*_ankle": h1_constants.STIFFNESS_ANKLE_TORSO,
}
H1_DEPLOY_LOWER_STIFFNESS = {
  pattern: kp
  for pattern, (kp, _) in H1_DEPLOY_PD_GAINS.items()
  if ("hip" in pattern or "knee" in pattern or "ankle" in pattern)
}
H1_LOWER_VELOCITY_LIMITS = {
  ".*_hip_.*": h1_constants.ACTUATOR_HIP_KNEE.velocity_limit,
  ".*_knee": h1_constants.ACTUATOR_HIP_KNEE.velocity_limit,
  ".*_ankle": h1_constants.ACTUATOR_ANKLE_TORSO.velocity_limit,
}
H1_LOWER_EFFORT_LIMITS = {
  ".*_hip_.*": h1_constants.ACTUATOR_HIP_KNEE.effort_limit,
  ".*_knee": h1_constants.ACTUATOR_HIP_KNEE.effort_limit,
  ".*_ankle": h1_constants.ACTUATOR_ANKLE_TORSO.effort_limit,
}


def _default_hands_cfg(enable: bool) -> HandsCfg | None:
  if not enable:
    return None

  def mount(site: str) -> HandMountCfg:
    return HandMountCfg(
      enable=True,
      enable_collision=False,
      mount_site=site,
      model=DEFAULT_2F85_XML,
      # Rotate gripper so its +Z (pinch direction) aligns with the H1 forearm
      # (+X). Euler order is XYZ in MuJoCo.
      offset_euler=(1.5707963267948966, 1.5707963267948966, 0.0),
      add_wrist_joint=True,
      wrist_axis=(0.0, 0.0, 1.0),
      wrist_range=(-1.0, 1.0),
      wrist_ctrlrange=(-1.0, 1.0),
      actuator_whitelist=("fingers_actuator",),
    )

  return HandsCfg(left=mount("left_hand_site"), right=mount("right_hand_site"))


def unitree_h1_homie_env_cfg(
  play: bool = False,
  curriculum_start_step: int = 0,
  hands: bool = False,
  gains: str = "deploy",
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree H1 HOMIE task configuration.

  Args:
    gains: PD-gain variant.
      - ``"deploy"`` (default): gains from Unitree's official RL stack
        (unitree_rl_gym h1_config.py) with the uniform 0.25 action scale.
        Use this for sim2real.
      - ``"mjlab"``: mjlab-style first-principles gains (armature x natural
        frequency). Sim-only / ablation variant.
  """
  if gains not in ("deploy", "mjlab"):
    raise ValueError(f"Unknown gains variant '{gains}'. Use 'deploy' or 'mjlab'.")
  cfg = make_homie_env_cfg()

  # Standing HOME keyframe as the default pose (OpenHomie squats are commanded
  # relative to standing; deviation rewards reference the default pose).
  if gains == "deploy":
    robot_cfg = get_h1_deploy_robot_cfg(hands=_default_hands_cfg(hands))
  else:
    robot_cfg = get_h1_robot_cfg(hands=_default_hands_cfg(hands))
    robot_cfg.init_state = h1_constants.HOME_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg}
  cfg.sim.mujoco.ccd_iterations = 50

  cfg.scene.sensors = _make_contact_sensors(
    feet_link_pattern=r"^(left_ankle_link|right_ankle_link)$",
    hip_knee_pattern=r"^(left|right)_(hip_(yaw|roll|pitch)|knee)_link$",
  )

  # Observations.
  cfg.observations = make_him_observations(
    joint_names=H1_ALL_JOINTS, num_actions=len(H1_LOWER_BODY_JOINTS)
  )

  # Actions.
  step_threshold = 0 if play else curriculum_start_step
  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = H1_LOWER_BODY_JOINTS
  if gains == "deploy":
    joint_pos.scale = H1_DEPLOY_ACTION_SCALE
  else:
    joint_pos.scale = {
      k: v
      for k, v in h1_constants.H1_ACTION_SCALE.items()
      if ("hip" in k or "knee" in k or "ankle" in k)
    }
  upper = cfg.actions["upper_body_pose"]
  assert isinstance(upper, mdp.UpperBodyPoseActionCfg)
  upper.joint_names = H1_UPPER_BODY_JOINTS
  cfg.events["upper_body_goals"].params["start_step"] = step_threshold
  cfg.curriculum["upper_body_action"].params["start_step"] = step_threshold

  # Optional policy-free gripper motion (with-hands variant).
  if hands:
    cfg.actions["gripper"] = mdp.GripperActuatorActionCfg(
      entity_name="robot",
      actuator_names=(r".*fingers_actuator.*",),
      target_range=(0.0, 1.0),
      interp_rate=0.05,
    )
    cfg.events["gripper_goals"] = EventTermCfg(
      func=mdp.sample_gripper_goals,
      mode="interval",
      interval_range_s=(0.75, 1.25),
      params={"action_name": "gripper", "start_step": step_threshold},
    )
    # Random hand payload (equivalent of carrying an object).
    cfg.events["hand_payload"] = EventTermCfg(
      mode="startup",
      func=mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("left_wrist_link", "right_wrist_link")
        ),
        "operation": "add",
        "ranges": (0.0, 2.0),
      },
    )

  # Commands.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.0
  height = cfg.commands["height"]
  assert isinstance(height, RelativeHeightCommandCfg)
  height.foot_site_names = ("left_foot", "right_foot")
  height.standing_height = H1_STANDING_HEIGHT
  height.ranges.height = H1_HEIGHT_RANGE

  # Events.
  foot_geoms = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 4)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["payload_mass"].params["asset_cfg"].body_names = ("torso_link",)

  # Rewards.
  site_names = ("left_foot", "right_foot")
  for name in (
    "track_height",
    "feet_clearance",
    "feet_slip",
    "contact_momentum",
    "feet_distance_lateral",
  ):
    cfg.rewards[name].params["asset_cfg"].site_names = site_names
  cfg.rewards["knee_deviation"].params["foot_asset_cfg"].site_names = site_names

  for name in ("lin_vel_z", "deviation_hip_joint", "deviation_ankle_joint"):
    cfg.rewards[name].params["min_height"] = H1_STANDING_GATE
  for name in (
    "feet_parallel",
    "feet_distance_lateral",
    "knee_distance_lateral",
    "stand_still",
  ):
    cfg.rewards[name].params["min_height"] = H1_STANDING_GATE
  cfg.rewards["feet_clearance"].params["min_height"] = H1_CLEARANCE_GATE
  cfg.rewards["feet_clearance"].params["target_height"] = 0.1

  lower_cfg = SceneEntityCfg("robot", joint_names=H1_LOWER_BODY_JOINTS)
  for name in (
    "dof_pos_limits",
    "torques",
    "dof_vel",
    "dof_vel_limits",
    "torque_limits",
    "joint_tracking_error",
  ):
    cfg.rewards[name].params["asset_cfg"] = lower_cfg
  cfg.rewards["torques"].params["stiffness"] = (
    H1_DEPLOY_LOWER_STIFFNESS if gains == "deploy" else H1_MJLAB_LOWER_STIFFNESS
  )
  cfg.rewards["dof_vel_limits"].params["velocity_limits"] = H1_LOWER_VELOCITY_LIMITS
  cfg.rewards["torque_limits"].params["effort_limits"] = H1_LOWER_EFFORT_LIMITS

  # H1 has explicit per-corner foot sites.
  left_corners = ("left_foot_fi", "left_foot_fo", "left_foot_ri", "left_foot_ro")
  right_corners = ("right_foot_fi", "right_foot_fo", "right_foot_ri", "right_foot_ro")
  cfg.rewards["feet_ground_parallel"].params.update(
    left_foot_points=left_corners,
    right_foot_points=right_corners,
    point_type="site",
  )
  # feet_parallel pairs corresponding left/right points; inner/outer flips
  # between feet, so the right-foot corners are reordered to match Y signs.
  cfg.rewards["feet_parallel"].params.update(
    left_foot_points=left_corners,
    right_foot_points=(
      "right_foot_fo",
      "right_foot_fi",
      "right_foot_ro",
      "right_foot_ri",
    ),
    point_type="site",
  )
  cfg.rewards["knee_distance_lateral"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    body_names=(
      "left_knee_link",
      "left_hip_yaw_link",
      "right_knee_link",
      "right_hip_yaw_link",
    ),
    preserve_order=True,
  )

  cfg.viewer.body_name = "torso_link"

  if play:
    _apply_play_overrides(cfg)
    twist.ranges.lin_vel_x = (-0.8, 1.2)
    twist.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
