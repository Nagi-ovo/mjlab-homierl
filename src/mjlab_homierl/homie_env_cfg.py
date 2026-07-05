"""HOMIE task configuration (robot-agnostic skeleton).

This module provides the base HOMIE task configuration: a squat + velocity
tracking + standing task for humanoids where only the lower body is policy
controlled and the upper body follows curriculum-driven random pose targets.

Reward terms, weights, and command sampling follow the OpenHomie reference
(HomieRL/legged_gym). Robot-specific values (joint lists, sites, actuator
limits, height ranges) are filled in by the configs in ``env_cfgs.py``.
"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_homierl import mdp
from mjlab_homierl.mdp import RelativeHeightCommandCfg, UniformVelocityCommandCfg

# Observation scaling constants (legged_gym defaults).
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
NOISE_SCALES = {"dof_pos": 0.02, "dof_vel": 2.0, "ang_vel": 0.5, "gravity": 0.05}


def make_him_observations(
  joint_names: tuple[str, ...], num_actions: int
) -> dict[str, ObservationGroupCfg]:
  """Build HIM actor/critic observation groups for a robot's joint layout.

  Actor: 6-step history of the one-step observation, with additive uniform
  noise. Critic: single noiseless step plus base linear velocity.
  """
  joint_asset_cfg = SceneEntityCfg(
    "robot", joint_names=joint_names, preserve_order=True
  )

  num_dofs = len(joint_names)
  one_step_dim = 10 + 2 * num_dofs + num_actions
  noise_vec = torch.zeros(one_step_dim, dtype=torch.float32)
  noise_vec[0:4] = 0.0  # Commands.
  noise_vec[4:7] = NOISE_SCALES["ang_vel"] * OBS_SCALES["ang_vel"]
  noise_vec[7:10] = NOISE_SCALES["gravity"]
  noise_vec[10 : 10 + num_dofs] = NOISE_SCALES["dof_pos"] * OBS_SCALES["dof_pos"]
  noise_vec[10 + num_dofs : 10 + 2 * num_dofs] = (
    NOISE_SCALES["dof_vel"] * OBS_SCALES["dof_vel"]
  )
  noise_vec[10 + 2 * num_dofs :] = 0.0  # Previous actions.

  common_params = {
    "command_name": "twist",
    "height_command_name": "height",
    "joint_asset_cfg": joint_asset_cfg,
    "obs_scales": OBS_SCALES,
  }

  return {
    "actor": ObservationGroupCfg(
      terms={
        "him_obs": ObservationTermCfg(
          func=mdp.him_actor_one_step_obs,
          params=dict(common_params),
          noise=Unoise(
            n_min=tuple((-noise_vec).tolist()),
            n_max=tuple(noise_vec.tolist()),
          ),
          clip=(-100.0, 100.0),
          history_length=6,
          flatten_history_dim=True,
        ),
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "him_privileged_obs": ObservationTermCfg(
          func=mdp.him_critic_one_step_obs,
          params=dict(common_params),
          clip=(-100.0, 100.0),
          history_length=1,
          flatten_history_dim=True,
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_homie_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base HOMIE task configuration.

  Placeholders marked "Set per-robot" must be filled by robot-specific configs
  (see ``env_cfgs.py``), including the observation groups, which depend on the
  robot's joint layout.
  """

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    # Randomized actuation latency (OpenHomie domain_rand.delay): 0..3 physics
    # substeps per policy step. Disabled in play (see _apply_play_overrides).
    "joint_pos": mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(),  # Set per-robot (lower body only).
      scale=0.25,  # Set per-robot.
      use_default_offset=True,
    ),
    "upper_body_pose": mdp.UpperBodyPoseActionCfg(
      entity_name="robot",
      joint_names=(),  # Set per-robot (upper body).
      interval_s=1.0,
    ),
  }

  ##
  # Commands (HOMIE 3-mode: 1/3 squat, 1/2 walk, 1/6 stand)
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 4.0),
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.8, 1.2),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.8, 0.8),
      ),
    ),
    "height": RelativeHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 4.0),
      debug_vis=True,
      foot_site_names=(),  # Set per-robot.
      standing_height=0.0,  # Set per-robot.
      ranges=RelativeHeightCommandCfg.Ranges(height=(0.0, 0.0)),  # Set per-robot.
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Global horizontal kick every 4 s, as in OpenHomie.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 4.0),
      is_global_time=True,
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
    # Upper-body goal resampling: global 1 s cadence (OpenHomie upper_interval_s).
    "upper_body_goals": EventTermCfg(
      func=mdp.sample_upper_body_goals,
      mode="interval",
      interval_range_s=(1.0, 1.0),
      is_global_time=True,
      params={"action_name": "upper_body_pose", "start_step": 0},
    ),
    # Domain randomization.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        # OpenHomie friction_range. Effective because mjlab feet geoms carry
        # priority=1, overriding the terrain's friction in contact mixing.
        "ranges": (0.1, 3.0),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        # Approximates OpenHomie's joint_injection + actuation_offset (each
        # +-0.05 rad): mjlab's encoder bias shifts observations and position
        # targets coherently, modeling a joint zero-calibration error.
        "bias_range": (-0.05, 0.05),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        # OpenHomie body_displacement: torso CoM shifted by up to +-0.1 m.
        "ranges": {0: (-0.1, 0.1), 1: (-0.1, 0.1), 2: (-0.1, 0.1)},
      },
    ),
    "link_mass": EventTermCfg(
      mode="startup",
      func=mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    ),
    "payload_mass": EventTermCfg(
      mode="startup",
      func=mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot (torso).
        "operation": "add",
        # OpenHomie payload_mass_range.
        "ranges": (-5.0, 10.0),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="reset",
      func=mdp.dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
  }

  ##
  # Rewards (weights follow OpenHomie's G1 config)
  ##

  lower_body_cfg = SceneEntityCfg("robot", joint_names=())  # Set per-robot.

  rewards = {
    # Command tracking.
    "track_lin_vel_x": RewardTermCfg(
      func=mdp.track_lin_vel_x_exp,
      weight=1.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_lin_vel_y": RewardTermCfg(
      func=mdp.track_lin_vel_y_exp,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_height": RewardTermCfg(
      func=mdp.track_relative_height,
      weight=2.0,
      params={
        "command_name": "height",
        "scale": 4.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Base motion.
    "lin_vel_z": RewardTermCfg(
      func=mdp.lin_vel_z_penalty,
      weight=-0.5,
      params={"height_command_name": "height", "min_height": 0.0},  # Set per-robot.
    ),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_penalty, weight=-0.025),
    "orientation": RewardTermCfg(func=mdp.orientation_penalty, weight=-1.5),
    # Pose deviations (gated to standing height).
    "deviation_hip_joint": RewardTermCfg(
      func=mdp.joint_deviation_gated,
      weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip.*",)),
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    "deviation_ankle_joint": RewardTermCfg(
      func=mdp.joint_deviation_gated,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*ankle.*",)),
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    "knee_deviation": RewardTermCfg(
      func=mdp.knee_deviation,
      weight=-0.75,
      params={
        "command_name": "height",
        "knee_asset_cfg": SceneEntityCfg("robot", joint_names=(r".*knee.*",)),
        "foot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Joint-space regularization.
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": lower_body_cfg},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "smoothness": RewardTermCfg(func=mdp.action_smoothness_l2, weight=-0.05),
    "dof_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_power": RewardTermCfg(
      func=mdp.joint_power,
      weight=-2e-5,
      params={"command_name": "twist"},
    ),
    "torques": RewardTermCfg(
      func=mdp.joint_torques_l2_normalized,
      weight=-2.5e-6,
      params={"asset_cfg": lower_body_cfg, "stiffness": {}},  # Set per-robot.
    ),
    "dof_vel": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": lower_body_cfg},
    ),
    "dof_vel_limits": RewardTermCfg(
      func=mdp.joint_vel_limits_cost,
      weight=-2e-3,
      params={
        "asset_cfg": lower_body_cfg,
        "velocity_limits": {},  # Set per-robot.
        "soft_factor": 0.8,
      },
    ),
    "torque_limits": RewardTermCfg(
      func=mdp.joint_torque_limits_cost,
      weight=-0.1,
      params={
        "asset_cfg": lower_body_cfg,
        "effort_limits": {},  # Set per-robot.
        "soft_factor": 0.95,
      },
    ),
    "joint_tracking_error": RewardTermCfg(
      func=mdp.joint_tracking_error,
      weight=-0.1,
      params={"asset_cfg": lower_body_cfg},
    ),
    "action_vanish": RewardTermCfg(
      func=mdp.action_out_of_bounds,
      weight=-1.0,
      params={"action_name": "joint_pos"},
    ),
    # Feet and contacts.
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.05,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "no_fly": RewardTermCfg(
      func=mdp.no_fly,
      weight=0.75,
      params={
        "sensor_name": "feet_ground_contact",
        "force_threshold": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "feet_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.25,
      params={
        "target_height": 0.14,  # Set per-robot.
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "force_threshold": 1.0,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-1.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.feet_contact_forces,
      weight=-0.00025,
      params={"sensor_name": "feet_ground_contact", "max_force": 400.0},
    ),
    "contact_momentum": RewardTermCfg(
      func=mdp.contact_momentum,
      weight=2.5e-4,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "force_threshold": 50.0,
      },
    ),
    "feet_ground_parallel": RewardTermCfg(
      func=mdp.feet_ground_parallel,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "left_foot_points": (),  # Set per-robot.
        "right_foot_points": (),  # Set per-robot.
        "point_type": "site",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "feet_parallel": RewardTermCfg(
      func=mdp.feet_parallel,
      weight=-3.0,
      params={
        "left_foot_points": (),  # Set per-robot.
        "right_foot_points": (),  # Set per-robot.
        "point_type": "site",
        "asset_cfg": SceneEntityCfg("robot"),
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    "feet_distance_lateral": RewardTermCfg(
      func=mdp.feet_distance_lateral,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "least_distance": 0.2,
        "most_distance": 0.35,
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    "knee_distance_lateral": RewardTermCfg(
      func=mdp.knee_distance_lateral,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "least_distance": 0.2,
        "most_distance": 0.35,
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-0.15,
      params={
        "sensor_name": "feet_ground_contact",
        "force_threshold": 0.1,
        "command_name": "twist",
        "command_threshold": 0.1,
        "height_command_name": "height",
        "min_height": 0.0,  # Set per-robot.
      },
    ),
    # Contact penalties (replace IsaacGym contact-based terminations; see docs).
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision"},
    ),
    "hip_knee_contact": RewardTermCfg(
      func=mdp.penalize_body_contacts,
      weight=-1.0,
      params={"sensor_name": "hip_knee_ground_contact"},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # OpenHomie: ||projected_gravity_xy|| > 0.8  <=>  tilt > asin(0.8).
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.asin(0.8)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "upper_body_action": CurriculumTermCfg(
      func=mdp.upper_body_action_curriculum,
      params={
        "action_name": "upper_body_pose",
        "reward_name": "track_lin_vel_x",
        "success_threshold": 0.8,
        "increment": 0.05,
        "max_ratio": 1.0,
        "start_step": 0,
      },
    ),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations={},  # Set per-robot via make_him_observations().
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    # OpenHomie feeds the estimator the PRE-reset critic observation on done
    # transitions (him_on_policy_runner.py:144); this recorder captures it.
    recorders={
      "terminal_critic_obs": RecorderTermCfg(func=mdp.TerminalCriticObsRecorder),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
