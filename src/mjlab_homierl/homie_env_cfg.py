"""Homie (humanoid walk) task configuration.

This module provides a factory function to create the homie task config.
The homie task is designed for humanoid robots with upper-body motion capabilities,
combining velocity tracking and squatting tasks with curriculum-driven complexity.
"""

import math
from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab_homierl import mdp
from mjlab_homierl.mdp import (
  RelativeHeightCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_homie_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base homie (humanoid walk) task configuration."""

  ##
  # Observations
  ##

  # HIMPPO observation layout (aligned with legged_gym):
  #   one_step_obs = [cmd(3), height_cmd(1), imu_ang_vel(3), gravity(3),
  #                   dof_pos(N), dof_vel(N), actions(A)]
  #   actor obs = actor_history * one_step_obs
  #   critic obs = critic_history * (one_step_obs + base_lin_vel(3))
  h1_joint_names = (
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
  joint_asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=h1_joint_names,
    preserve_order=True,
  )

  # Deterministic scaling constants (matches legged_gym defaults).
  obs_scales = {"lin_vel": 2.0, "ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
  noise_scales = {"dof_pos": 0.02, "dof_vel": 2.0, "ang_vel": 0.5, "gravity": 0.05}
  noise_level = 1.0
  num_dofs = len(h1_joint_names)
  num_actions = 10  # Lower-body action dim for H1 homie.
  one_step_obs_dim = 10 + 2 * num_dofs + num_actions
  noise_vec = torch.zeros(one_step_obs_dim, dtype=torch.float32)
  noise_vec[0:4] = 0.0  # commands
  noise_vec[4:7] = noise_scales["ang_vel"] * noise_level * obs_scales["ang_vel"]
  noise_vec[7:10] = noise_scales["gravity"] * noise_level
  noise_vec[10 : 10 + num_dofs] = (
    noise_scales["dof_pos"] * noise_level * obs_scales["dof_pos"]
  )
  noise_vec[10 + num_dofs : 10 + 2 * num_dofs] = (
    noise_scales["dof_vel"] * noise_level * obs_scales["dof_vel"]
  )
  noise_vec[10 + 2 * num_dofs :] = 0.0  # previous actions

  policy_terms = {
    "him_obs": ObservationTermCfg(
      func=mdp.him_actor_one_step_obs,
      params={
        "command_name": "twist",
        "height_command_name": "height",
        "imu_ang_vel_sensor_name": "robot/imu_ang_vel",
        "joint_asset_cfg": joint_asset_cfg,
        "obs_scales": obs_scales,
      },
      noise=Unoise(
        n_min=tuple((-noise_vec).cpu().tolist()),
        n_max=tuple((noise_vec).cpu().tolist()),
      ),
      history_length=6,
      flatten_history_dim=True,
    ),
  }

  critic_terms = {
    "him_privileged_obs": ObservationTermCfg(
      func=mdp.him_critic_one_step_obs,
      params={
        "command_name": "twist",
        "height_command_name": "height",
        "imu_ang_vel_sensor_name": "robot/imu_ang_vel",
        "joint_asset_cfg": joint_asset_cfg,
        "obs_scales": obs_scales,
      },
      history_length=1,
      flatten_history_dim=True,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 5.0),  # Fixed 5s resampling
      # HOMIE (IsaacGym) 3-mode mutually exclusive commands:
      #   1/3 squat (height), 1/2 velocity tracking, 1/6 standing.
      homie_three_mode=True,
      # The remaining parameters are unused when homie_three_mode=True, but
      # kept for backwards compatibility.
      rel_standing_envs=1.0 / 6.0,
      rel_heading_envs=0.0,
      rel_pure_lin_vel_x_envs=0.0,
      rel_pure_lin_vel_y_envs=0.0,
      heading_command=False,
      heading_control_stiffness=0.5,
      active_env_group=None,
      avoid_consecutive_standing=True,  # Don't sample standing twice in a row
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.8, 1.2),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.8, 0.8),
        heading=None,
      ),
    ),
    "height": RelativeHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      active_env_group=None,
      # Smooth height-command transitions (avoid step changes at resampling).
      interp_rate=0.02,
      homie_three_mode=True,
      foot_site_names=(),  # Set per-robot.
      ranges=RelativeHeightCommandCfg.Ranges(height=(0.6, 1.0)),
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
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=3.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.5),
      },
    ),
    "track_height": RewardTermCfg(
      func=mdp.track_relative_height,
      weight=2.0,
      params={
        "command_name": "height",
        "std": math.sqrt(0.02),
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "knee_deviation": RewardTermCfg(
      func=mdp.knee_deviation_reward,
      weight=-0.75,
      params={
        "command_name": "height",
        "knee_asset_cfg": SceneEntityCfg(
          "robot",
          # Default covers humanoids ("knee") and quadrupeds ("calf").
          joint_names=(r".*(knee|calf).*",),
        ),
        "foot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=0.5,  # Disabled - replaced by orientation_penalty
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "orientation_penalty": RewardTermCfg(
      func=mdp.orientation_penalty,
      weight=-3.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),  # Uses root link by default.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
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
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.05,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.25, # -2.0
      params={
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "force_threshold": 1.0,
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "feet_ground_parallel": RewardTermCfg(
      func=mdp.feet_ground_parallel,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "left_foot_sites": (),  # Set per-robot (e.g., left_toe, left_heel).
        "right_foot_sites": (),  # Set per-robot (e.g., right_toe, right_heel).
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "feet_parallel": RewardTermCfg(
      func=mdp.feet_parallel,
      weight=-3.0,
      params={
        "left_foot_sites": (),  # Set per-robot.
        "right_foot_sites": (),  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
        "height_command_name": "height",
        "min_height": 0.735,
      },
    ),
    "deviation_ankle_joint": RewardTermCfg(
      func=mdp.deviation_ankle_joint,
      weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*ankle.*",)),
        "height_command_name": "height",
        "min_height": 0.735,
      },
    ),
    "deviation_hip_joint": RewardTermCfg(
      func=mdp.deviation_hip_joint,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip.*",)),
        "height_command_name": "height",
        "min_height": 0.735,
      },
    ),
    "knee_distance_lateral": RewardTermCfg(
      func=mdp.knee_distance_lateral,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(r".*knee.*",)),
        "min_distance": 0.1,
        "max_distance": 0.4,
        "height_command_name": "height",
        "min_height": 0.735,
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
        "min_height": 0.735,
      },
    ),
    "hip_knee_contact": RewardTermCfg(
      func=mdp.penalize_body_contacts,
      weight=-1.0,
      params={
        "sensor_name": "",  # Set per-robot in config (e.g., "hip_knee_ground_contact").
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # Velocity curriculum disabled - using fixed ranges
    # "command_vel": CurriculumTermCfg(
    #   func=mdp.commands_vel,
    #   params={
    #     "command_name": "twist",
    #     "velocity_stages": [
    #       {"step": 0, "lin_vel_x": (-0.8, 1.2), "ang_vel_z": (-0.8, 0.8)},
    #     ],
    #   },
    # ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
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
