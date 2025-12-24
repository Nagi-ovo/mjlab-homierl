"""Unitree H1 constants."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_h1" / "xmls" / "h1.xml"
)
assert H1_XML.exists()

# Default Robotiq 2F85 path.
DEFAULT_2F85_XML = (
  MJLAB_SRC_PATH / "asset_zoo" / "hands" / "robotiq_2f85" / "xmls" / "2f85.xml"
)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_XML.parent / "assets", meshdir)
  return assets


@dataclass(kw_only=True)
class HandMountCfg:
  """Configuration for mounting an external gripper onto the wrist."""

  enable: bool = False
  mount_site: str = ""
  model: Path | None = None
  offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  offset_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
  add_wrist_joint: bool = False
  wrist_joint_type: str = "hinge"
  wrist_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
  wrist_range: tuple[float, float] = (-1.0, 1.0)
  wrist_ctrlrange: tuple[float, float] = (-1.0, 1.0)
  actuator_whitelist: Sequence[str] = ("fingers_actuator",)
  pinch_site: str | None = "pinch"


@dataclass(kw_only=True)
class HandsCfg:
  left: HandMountCfg = field(
    default_factory=lambda: HandMountCfg(
      enable=False,
      mount_site="left_hand_site",
    )
  )
  right: HandMountCfg = field(
    default_factory=lambda: HandMountCfg(
      enable=False,
      mount_site="right_hand_site",
    )
  )


def _find_or_create_site(
  model: mujoco.MjSpec, site_name: str, fallback_body: str
) -> mujoco.MjsSite:
  try:
    site = model.site(site_name)
    if site is not None:
      return site
  except KeyError:
    # Will create below.
    ...

  body = model.body(fallback_body)
  return body.add_site(name=site_name, pos=(0.0, 0.0, 0.0))


def _maybe_add_wrist_and_joint(
  model: mujoco.MjSpec, side: str, cfg: HandMountCfg
) -> mujoco.MjsSite:
  site_parent = None
  site_pos = None
  created_site = False

  site = None
  try:
    site = model.site(cfg.mount_site)
  except KeyError:
    site = None

  if site is not None:
    site_parent = site.parent
    site_pos = site.pos.copy()
  else:
    # Fallback: use hand collision geom pose as anchor.
    hand_geom = None
    try:
      hand_geom = model.geom(f"{side}_hand_collision")
    except KeyError:
      hand_geom = None

    if hand_geom is not None:
      site_parent = hand_geom.parent
      site_pos = hand_geom.pos.copy()
    else:
      # Last resort: create on elbow link origin.
      site_parent = model.body(f"{side}_elbow_link")
      site_pos = np.array((0.0, 0.0, 0.0))
    site = site_parent.add_site(name=cfg.mount_site, pos=site_pos)
    created_site = True

  wrist_body = site_parent.add_body(name=f"{side}_wrist_link", pos=site_pos)
  if created_site:
    # Rename the temp site to avoid duplication, then create the wrist site with the desired name.
    site.name = f"{cfg.mount_site}_legacy_wrist"
    wrist_site = wrist_body.add_site(name=cfg.mount_site, pos=site.pos, quat=site.quat)
  else:
    # Avoid duplicate names: rename original, then create a new site with the original name under wrist.
    original_site_name = site.name
    site.name = f"{original_site_name}_legacy_wrist"
    wrist_site = wrist_body.add_site(
      name=original_site_name, pos=site.pos, quat=site.quat
    )

  if cfg.add_wrist_joint:
    joint_type_map = {
      "hinge": mujoco.mjtJoint.mjJNT_HINGE,
      "slide": mujoco.mjtJoint.mjJNT_SLIDE,
      "ball": mujoco.mjtJoint.mjJNT_BALL,
      "free": mujoco.mjtJoint.mjJNT_FREE,
    }
    joint_type = joint_type_map.get(cfg.wrist_joint_type, mujoco.mjtJoint.mjJNT_HINGE)
    joint = wrist_body.add_joint(
      type=joint_type,
      name=f"{side}_wrist",
      axis=cfg.wrist_axis,
      range=cfg.wrist_range,
    )
    actuator = model.add_actuator(name=f"{side}_wrist", target=joint.name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.ctrllimited = True
    actuator.ctrlrange[:] = cfg.wrist_ctrlrange

  return wrist_site


def _attach_gripper(
  model: mujoco.MjSpec, side: str, cfg: HandMountCfg
) -> tuple[list[str], list[str]]:
  """Attach external gripper MJCF under wrist site; returns (actuator_names, joint_names)."""
  if not cfg.enable or cfg.model is None:
    return [], []

  wrist_site = _maybe_add_wrist_and_joint(model, side, cfg)

  gripper_spec = mujoco.MjSpec.from_file(str(cfg.model))
  gripper_frame = wrist_site.parent.add_frame(
    pos=cfg.offset_pos, euler=cfg.offset_euler
  )
  prefix = f"{side}_gripper/"
  model.attach(gripper_spec, frame=gripper_frame, prefix=prefix)

  actuators: list[str] = []
  joints: list[str] = []
  for act in gripper_spec.actuators:
    name = act.name or act.tendon or act.joint or ""
    full_name = prefix + name if name else ""
    if any(key in full_name for key in cfg.actuator_whitelist):
      actuators.append(full_name)
  for j in gripper_spec.joints:
    joints.append(prefix + j.name)
  return actuators, joints


def get_spec(hands: HandsCfg | None = None) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_XML))
  assets = get_assets(spec.meshdir)
  if hands is not None:
    for side, hand_cfg in (("left", hands.left), ("right", hands.right)):
      _attach_gripper(spec, side, hand_cfg)
    if hands.left.enable and hands.left.model is not None:
      update_assets(assets, hands.left.model.parent / "assets", meshdir="assets")
    if hands.right.enable and hands.right.model is not None:
      update_assets(assets, hands.right.model.parent / "assets", meshdir="assets")

  spec.assets = assets
  return spec


##
# Actuator config.
##

# Motor specs (based on Unitree H1 specifications)
# H1 uses similar actuator technology to G1 but with different configurations
# Using conservative estimates based on G1's actuator specs

# Rotor inertias and gear ratios for different joint types
# These are approximations and should be refined based on actual H1 specifications

# Hip and knee actuators (larger, more powerful)
ROTOR_INERTIAS_HIP_KNEE = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_HIP_KNEE = (
  1,
  4.5,
  5,
)
ARMATURE_HIP_KNEE = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_HIP_KNEE, GEARS_HIP_KNEE
)

# Ankle and torso actuators (medium)
ROTOR_INERTIAS_ANKLE_TORSO = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_ANKLE_TORSO = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_ANKLE_TORSO = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_ANKLE_TORSO, GEARS_ANKLE_TORSO
)

# Shoulder and elbow actuators (smaller)
ROTOR_INERTIAS_ARM = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_ARM = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_ARM = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_ARM, GEARS_ARM)

# Actuator specifications
ACTUATOR_HIP_KNEE = ElectricActuator(
  reflected_inertia=ARMATURE_HIP_KNEE,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_ANKLE_TORSO = ElectricActuator(
  reflected_inertia=ARMATURE_ANKLE_TORSO,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_ARM = ElectricActuator(
  reflected_inertia=ARMATURE_ARM,
  velocity_limit=37.0,
  effort_limit=25.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE_TORSO = ARMATURE_ANKLE_TORSO * NATURAL_FREQ**2
STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_ANKLE_TORSO = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_TORSO * NATURAL_FREQ
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ

H1_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    ".*_hip_pitch",
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_knee",
  ),
  stiffness=STIFFNESS_HIP_KNEE,
  damping=DAMPING_HIP_KNEE,
  effort_limit=ACTUATOR_HIP_KNEE.effort_limit,
  armature=ACTUATOR_HIP_KNEE.reflected_inertia,
)

H1_ACTUATOR_ANKLE_TORSO = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    ".*_ankle",
    "torso",
  ),
  stiffness=STIFFNESS_ANKLE_TORSO,
  damping=DAMPING_ANKLE_TORSO,
  effort_limit=ACTUATOR_ANKLE_TORSO.effort_limit,
  armature=ACTUATOR_ANKLE_TORSO.reflected_inertia,
)

H1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
    ".*_elbow",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=ACTUATOR_ARM.effort_limit,
  armature=ACTUATOR_ARM.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.98),  # H1 is about 1.8m tall, standing height around 0.98m
  joint_pos={
    ".*_hip_pitch": -0.1,
    ".*_knee": 0.3,
    ".*_ankle": -0.2,
    ".*_shoulder_pitch": 0.3,
    ".*_elbow": 1.0,
    "left_shoulder_roll": 0.2,
    "right_shoulder_roll": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.92),
  joint_pos={
    ".*_hip_pitch": -0.35,
    ".*_knee": 0.7,
    ".*_ankle": -0.35,
    ".*_elbow": 0.8,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.3,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.3,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Foot geom names
_foot_regex = r"^(left|right)_foot[1-3]_collision$"

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
  contype=0,
  conaffinity=1,
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

H1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_ACTUATOR_HIP_KNEE,
    H1_ACTUATOR_ANKLE_TORSO,
    H1_ACTUATOR_ARM,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_robot_cfg(hands: HandsCfg | None = None) -> EntityCfg:
  """Get a fresh H1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=lambda: get_spec(hands=hands),
    articulation=H1_ARTICULATION,
  )


H1_ACTION_SCALE: dict[str, float] = {}
for a in H1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  assert e is not None
  for n in names:
    H1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_robot_cfg())

  viewer.launch(robot.spec.compile())
