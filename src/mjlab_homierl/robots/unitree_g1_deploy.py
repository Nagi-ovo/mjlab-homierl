"""Unitree G1 deployment-grade PD gains — single source of truth.

The stiffness/damping values follow OpenHomie's real-robot deployment stack
(``HomieDeploy/unitree_sdk2/g1_control.cpp``): these are the gains the onboard
low-level controller runs. Training with the same gains keeps the simulated
closed loop aligned with the deployed one, which is the property that actually
matters for sim2real.

Motor properties (armature, effort limits) still come from mjlab's asset-zoo
constants — they are physical and independent of the controller choice. The
exported ONNX embeds the effective per-joint stiffness/damping via mjlab's
``get_base_metadata``, so deployment code can read the table from the policy
file instead of duplicating it.
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# (kp [N·m/rad], kd [N·m·s/rad]) per joint — HomieDeploy g1_control.cpp.
G1_DEPLOY_PD_GAINS: dict[str, tuple[float, float]] = {
  ".*_hip_pitch_joint": (150.0, 2.0),
  ".*_hip_roll_joint": (150.0, 2.0),
  ".*_hip_yaw_joint": (150.0, 2.0),
  ".*_knee_joint": (300.0, 4.0),
  ".*_ankle_pitch_joint": (40.0, 2.0),
  ".*_ankle_roll_joint": (40.0, 2.0),
  "waist_yaw_joint": (300.0, 5.0),
  "waist_roll_joint": (300.0, 5.0),
  "waist_pitch_joint": (300.0, 5.0),
  ".*_shoulder_pitch_joint": (150.0, 4.0),
  ".*_shoulder_roll_joint": (150.0, 4.0),
  ".*_shoulder_yaw_joint": (150.0, 4.0),
  ".*_elbow_joint": (100.0, 1.0),
  ".*_wrist_roll_joint": (10.0, 0.5),
  ".*_wrist_pitch_joint": (10.0, 0.5),
  ".*_wrist_yaw_joint": (5.0, 0.5),
}

# Deployment action scale (HomieDeploy ``ACTION_SCALE``): position target =
# default + 0.25 * action, uniform across joints.
G1_DEPLOY_ACTION_SCALE = 0.25


def _kp(pattern: str) -> float:
  return G1_DEPLOY_PD_GAINS[pattern][0]


def _kd(pattern: str) -> float:
  return G1_DEPLOY_PD_GAINS[pattern][1]


def _actuator(
  patterns: tuple[str, ...],
  armature: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  """Actuator group with deploy gains; all patterns must share one (kp, kd)."""
  gains = {G1_DEPLOY_PD_GAINS[p] for p in patterns}
  if len(gains) != 1:
    raise ValueError(f"Actuator group {patterns} mixes different PD gains: {gains}.")
  return BuiltinPositionActuatorCfg(
    target_names_expr=patterns,
    stiffness=_kp(patterns[0]),
    damping=_kd(patterns[0]),
    effort_limit=effort_limit,
    armature=armature,
  )


# Groups are the intersection of the gain table with mjlab's motor types
# (armature / effort limits per motor).
G1_DEPLOY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _actuator(
      (".*_hip_pitch_joint", ".*_hip_yaw_joint"),
      armature=g1_constants.ARMATURE_7520_14,
      effort_limit=g1_constants.ACTUATOR_7520_14.effort_limit,
    ),
    _actuator(
      (".*_hip_roll_joint",),
      armature=g1_constants.ARMATURE_7520_22,
      effort_limit=g1_constants.ACTUATOR_7520_22.effort_limit,
    ),
    _actuator(
      (".*_knee_joint",),
      armature=g1_constants.ARMATURE_7520_22,
      effort_limit=g1_constants.ACTUATOR_7520_22.effort_limit,
    ),
    _actuator(
      (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      armature=g1_constants.ARMATURE_5020 * 2,
      effort_limit=g1_constants.ACTUATOR_5020.effort_limit * 2,
    ),
    _actuator(
      ("waist_yaw_joint",),
      armature=g1_constants.ARMATURE_7520_14,
      effort_limit=g1_constants.ACTUATOR_7520_14.effort_limit,
    ),
    _actuator(
      ("waist_roll_joint", "waist_pitch_joint"),
      armature=g1_constants.ARMATURE_5020 * 2,
      effort_limit=g1_constants.ACTUATOR_5020.effort_limit * 2,
    ),
    _actuator(
      (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
      ),
      armature=g1_constants.ARMATURE_5020,
      effort_limit=g1_constants.ACTUATOR_5020.effort_limit,
    ),
    _actuator(
      (".*_elbow_joint",),
      armature=g1_constants.ARMATURE_5020,
      effort_limit=g1_constants.ACTUATOR_5020.effort_limit,
    ),
    _actuator(
      (".*_wrist_roll_joint",),
      armature=g1_constants.ARMATURE_5020,
      effort_limit=g1_constants.ACTUATOR_5020.effort_limit,
    ),
    _actuator(
      (".*_wrist_pitch_joint",),
      armature=g1_constants.ARMATURE_4010,
      effort_limit=g1_constants.ACTUATOR_4010.effort_limit,
    ),
    _actuator(
      (".*_wrist_yaw_joint",),
      armature=g1_constants.ARMATURE_4010,
      effort_limit=g1_constants.ACTUATOR_4010.effort_limit,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_deploy_robot_cfg() -> EntityCfg:
  """G1 with deployment-grade PD gains and the standing HOME keyframe."""
  return EntityCfg(
    init_state=g1_constants.HOME_KEYFRAME,
    collisions=(g1_constants.FULL_COLLISION,),
    spec_fn=g1_constants.get_spec,
    articulation=G1_DEPLOY_ARTICULATION,
  )
