import math

import pytest

from mjlab_homierl.env_cfgs import (
  unitree_g1_homie_env_cfg,
  unitree_h1_homie_env_cfg,
)


@pytest.mark.parametrize(
  "make_cfg", [unitree_g1_homie_env_cfg, unitree_h1_homie_env_cfg]
)
def test_homie_env_cfg_uses_actor_critic_groups(make_cfg) -> None:
  cfg = make_cfg()
  assert "actor" in cfg.observations
  assert "critic" in cfg.observations
  assert "joint_pos" in cfg.actions
  assert "upper_body_pose" in cfg.actions


@pytest.mark.parametrize(
  "make_cfg", [unitree_g1_homie_env_cfg, unitree_h1_homie_env_cfg]
)
def test_homie_reward_weights_follow_openhomie(make_cfg) -> None:
  cfg = make_cfg()
  rewards = cfg.rewards
  # OpenHomie G1 config weights.
  assert rewards["track_lin_vel_x"].weight == 1.5
  assert rewards["track_lin_vel_y"].weight == 1.0
  assert rewards["track_ang_vel"].weight == 2.0
  assert rewards["track_ang_vel"].params["std"] == pytest.approx(math.sqrt(0.25))
  assert rewards["deviation_hip_joint"].weight == -0.2
  assert rewards["deviation_ankle_joint"].weight == -0.5
  assert rewards["orientation"].weight == -1.5
  assert rewards["action_rate"].weight == -0.01
  assert rewards["feet_slip"].weight == -0.25
  assert rewards["dof_pos_limits"].weight == -2.0


@pytest.mark.parametrize(
  "make_cfg", [unitree_g1_homie_env_cfg, unitree_h1_homie_env_cfg]
)
def test_homie_commands_resample_every_4s(make_cfg) -> None:
  cfg = make_cfg()
  assert cfg.commands["twist"].resampling_time_range == (4.0, 4.0)
  assert cfg.commands["height"].resampling_time_range == (4.0, 4.0)


def test_homie_env_cfg_with_hands_adds_gripper_action() -> None:
  cfg = unitree_h1_homie_env_cfg(hands=True)
  assert "gripper" in cfg.actions
  assert "hand_payload" in cfg.events
  assert cfg.sim.mujoco.ccd_iterations == 50


@pytest.mark.parametrize(
  "make_cfg", [unitree_g1_homie_env_cfg, unitree_h1_homie_env_cfg]
)
def test_homie_play_cfg_strips_training_only_work(make_cfg) -> None:
  cfg = make_cfg(play=True)
  assert "actor" in cfg.observations
  assert "critic" not in cfg.observations
  assert cfg.rewards == {}
  assert cfg.curriculum == {}
  assert "push_robot" not in cfg.events


def test_g1_terminates_on_torso_contact() -> None:
  cfg = unitree_g1_homie_env_cfg()
  assert "torso_contact" in cfg.terminations


def test_g1_gain_variants() -> None:
  from mjlab_homierl.robots.unitree_g1_deploy import G1_DEPLOY_PD_GAINS

  deploy = unitree_g1_homie_env_cfg(gains="deploy")
  # Deployment pipeline uses a uniform 0.25 action scale.
  assert deploy.actions["joint_pos"].scale == 0.25
  # Torque-reward normalization must match the deploy gain table.
  stiffness = deploy.rewards["torques"].params["stiffness"]
  assert stiffness[".*_knee_joint"] == G1_DEPLOY_PD_GAINS[".*_knee_joint"][0] == 300.0

  mjlab_variant = unitree_g1_homie_env_cfg(gains="mjlab")
  assert isinstance(mjlab_variant.actions["joint_pos"].scale, dict)

  with pytest.raises(ValueError):
    unitree_g1_homie_env_cfg(gains="unknown")
