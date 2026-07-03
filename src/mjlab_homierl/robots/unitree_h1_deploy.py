"""Unitree H1 deployment-grade PD gains — single source of truth.

The stiffness/damping values follow Unitree's official RL stack
(``unitreerobotics/unitree_rl_gym``, ``legged_gym/envs/h1/h1_config.py``):

  stiffness = {hip_yaw: 150, hip_roll: 150, hip_pitch: 150, knee: 200,
               ankle: 40, torso: 300, shoulder: 150, elbow: 100}
  damping   = {hip_yaw: 2, hip_roll: 2, hip_pitch: 2, knee: 4,
               ankle: 2, torso: 6, shoulder: 2, elbow: 2}
  action_scale = 0.25

Note: Unitree's own ``deploy/deploy_real/configs/h1.yaml`` agrees with this
table for the legs but differs for the upper body (torso kd 3, shoulder kp
100, elbow-region kp 50). The legs — the policy-controlled joints — are
identical in both; if your deployment stack sends the deploy_real upper-body
values, adjust this table to match it.

Motor properties (armature, effort limits) come from this package's H1
constants — they are physical and independent of the controller choice.
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_homierl.robots.unitree_h1 import h1_constants
from mjlab_homierl.robots.unitree_h1.h1_constants import HandsCfg

# (kp [N·m/rad], kd [N·m·s/rad]) per joint — unitree_rl_gym h1_config.py.
H1_DEPLOY_PD_GAINS: dict[str, tuple[float, float]] = {
  ".*_hip_yaw": (150.0, 2.0),
  ".*_hip_roll": (150.0, 2.0),
  ".*_hip_pitch": (150.0, 2.0),
  ".*_knee": (200.0, 4.0),
  ".*_ankle": (40.0, 2.0),
  "torso": (300.0, 6.0),
  ".*_shoulder_pitch": (150.0, 2.0),
  ".*_shoulder_roll": (150.0, 2.0),
  ".*_shoulder_yaw": (150.0, 2.0),
  ".*_elbow": (100.0, 2.0),
}

# Deployment action scale (unitree_rl_gym ``action_scale``).
H1_DEPLOY_ACTION_SCALE = 0.25


def _actuator(
  patterns: tuple[str, ...],
  armature: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  """Actuator group with deploy gains; all patterns must share one (kp, kd)."""
  gains = {H1_DEPLOY_PD_GAINS[p] for p in patterns}
  if len(gains) != 1:
    raise ValueError(f"Actuator group {patterns} mixes different PD gains: {gains}.")
  kp, kd = gains.pop()
  return BuiltinPositionActuatorCfg(
    target_names_expr=patterns,
    stiffness=kp,
    damping=kd,
    effort_limit=effort_limit,
    armature=armature,
  )


H1_DEPLOY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _actuator(
      (".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch"),
      armature=h1_constants.ACTUATOR_HIP_KNEE.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_HIP_KNEE.effort_limit,
    ),
    _actuator(
      (".*_knee",),
      armature=h1_constants.ACTUATOR_HIP_KNEE.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_HIP_KNEE.effort_limit,
    ),
    _actuator(
      (".*_ankle",),
      armature=h1_constants.ACTUATOR_ANKLE_TORSO.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_ANKLE_TORSO.effort_limit,
    ),
    _actuator(
      ("torso",),
      armature=h1_constants.ACTUATOR_ANKLE_TORSO.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_ANKLE_TORSO.effort_limit,
    ),
    _actuator(
      (".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw"),
      armature=h1_constants.ACTUATOR_ARM.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_ARM.effort_limit,
    ),
    _actuator(
      (".*_elbow",),
      armature=h1_constants.ACTUATOR_ARM.reflected_inertia,
      effort_limit=h1_constants.ACTUATOR_ARM.effort_limit,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_deploy_robot_cfg(hands: HandsCfg | None = None) -> EntityCfg:
  """H1 with deployment-grade PD gains and the standing HOME keyframe."""
  return EntityCfg(
    init_state=h1_constants.HOME_KEYFRAME,
    collisions=(h1_constants.FULL_COLLISION,),
    spec_fn=lambda: h1_constants.get_spec(hands=hands),
    articulation=H1_DEPLOY_ARTICULATION,
  )
