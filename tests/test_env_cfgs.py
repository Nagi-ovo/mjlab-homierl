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


def test_g1_has_no_self_collision_penalty() -> None:
  # OpenHomie G1 trains with self-collision disabled (IsaacGym
  # self_collision=1) and no such penalty in its reward scales; the term
  # walled squatting via permanent wrist-hip contacts (see env cfg comment).
  assert "self_collisions" not in unitree_g1_homie_env_cfg().rewards
  assert "self_collisions" in unitree_h1_homie_env_cfg().rewards


def test_g1_dex3_variant() -> None:
  cfg = unitree_g1_homie_env_cfg(dex3=True)
  assert "hand_payload" in cfg.events
  # Interface must stay identical to the base task (checkpoint-compatible).
  base = unitree_g1_homie_env_cfg()
  assert (
    cfg.actions["joint_pos"].actuator_names == base.actions["joint_pos"].actuator_names
  )
  spec = cfg.scene.entities["robot"].spec_fn()
  dex3_bodies = [b.name for b in spec.bodies if "dex3" in b.name]
  assert "left_dex3/left_dex3_mount" in dex3_bodies
  assert "right_dex3/right_dex3_mount" in dex3_bodies


def test_h1_gain_variants() -> None:
  from mjlab_homierl.robots.unitree_h1_deploy import H1_DEPLOY_PD_GAINS

  deploy = unitree_h1_homie_env_cfg(gains="deploy")
  assert deploy.actions["joint_pos"].scale == 0.25
  stiffness = deploy.rewards["torques"].params["stiffness"]
  assert stiffness[".*_knee"] == H1_DEPLOY_PD_GAINS[".*_knee"][0] == 200.0

  mjlab_variant = unitree_h1_homie_env_cfg(gains="mjlab")
  assert isinstance(mjlab_variant.actions["joint_pos"].scale, dict)

  with pytest.raises(ValueError):
    unitree_h1_homie_env_cfg(gains="unknown")
