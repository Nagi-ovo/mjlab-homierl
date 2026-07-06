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


def test_g1_waist_variants() -> None:
  # Default = OpenHomie 27-dof parity: waist_roll/pitch held at the default
  # pose, only waist_yaw in the disturbance set.
  locked = unitree_g1_homie_env_cfg()
  locked_joints = locked.actions["upper_body_pose"].joint_names
  assert "waist_yaw_joint" in locked_joints
  assert "waist_roll_joint" not in locked_joints
  assert "waist_pitch_joint" not in locked_joints
  assert len(locked_joints) == 15

  free = unitree_g1_homie_env_cfg(waist="free")
  free_joints = free.actions["upper_body_pose"].joint_names
  assert len(free_joints) == 17
  # Interface unchanged: same policy joints either way (checkpoint-compatible).
  assert (
    locked.actions["joint_pos"].actuator_names
    == free.actions["joint_pos"].actuator_names
  )

  with pytest.raises(ValueError):
    unitree_g1_homie_env_cfg(waist="unknown")


def test_g1_homie_plus_deployment_extensions() -> None:
  from mjlab_homierl import mdp

  cfg = unitree_g1_homie_env_cfg(
    torso_pitch=True, inplace_prob=1.0 / 3.0, floor="compliant"
  )

  # In-place locomotion sampling (pure strafe/turn corners).
  assert cfg.commands["twist"].inplace_prob == 1.0 / 3.0
  # Foot contact-compliance DR (soft floors): in-episode re-randomization at
  # ~0.5 s average, per arXiv:2504.13619.
  assert cfg.events["foot_compliance"].params["ranges"][0] == (0.02, 0.1)
  assert cfg.events["foot_compliance"].mode == "interval"
  assert cfg.events["foot_compliance"].interval_range_s == (0.3, 0.7)
  with pytest.raises(ValueError):
    unitree_g1_homie_env_cfg(floor="unknown")

  # 5th command dim: torso_pitch term, coupled to the twist mode.
  assert isinstance(cfg.commands["torso_pitch"], mdp.TorsoPitchCommandCfg)
  assert cfg.commands["torso_pitch"].resampling_time_range == (4.0, 4.0)
  # Command-driven waist_pitch, zero policy dims; policy interface unchanged.
  assert isinstance(cfg.actions["torso_pitch"], mdp.TorsoPitchActionCfg)
  assert len(cfg.actions["joint_pos"].actuator_names) == 12
  # waist_pitch must not also be in the random disturbance set.
  assert "waist_pitch_joint" not in cfg.actions["upper_body_pose"].joint_names
  # Rewards: pitch tracking added, hip-deviation gate lifted while leaning.
  assert "track_torso_pitch" in cfg.rewards
  assert (
    cfg.rewards["deviation_hip_joint"].params["pitch_command_name"] == "torso_pitch"
  )
  # Observation command segment is 5-dim (one-step 81 = 5 + 6 + 2*29 + 12).
  actor_term = cfg.observations["actor"].terms["him_obs"]
  assert actor_term.params["pitch_command_name"] == "torso_pitch"
  assert len(actor_term.noise.n_max) == 81

  # Base task is untouched (4-dim command, parity sampler, rigid floor).
  base = unitree_g1_homie_env_cfg()
  assert "torso_pitch" not in base.commands
  assert "torso_pitch" not in base.actions
  assert "track_torso_pitch" not in base.rewards
  assert len(base.observations["actor"].terms["him_obs"].noise.n_max) == 80
  assert base.commands["twist"].inplace_prob == 0.0
  assert "foot_compliance" not in base.events

  with pytest.raises(ValueError):
    unitree_g1_homie_env_cfg(torso_pitch=True, waist="free")


def test_g1_has_no_self_collision_penalty() -> None:
  # OpenHomie G1 trains with self-collision disabled (IsaacGym
  # self_collision=1) and no such penalty in its reward scales; the term
  # walled squatting via permanent wrist-hip contacts (see env cfg comment).
  assert "self_collisions" not in unitree_g1_homie_env_cfg().rewards
  assert "self_collisions" in unitree_h1_homie_env_cfg().rewards


@pytest.mark.parametrize(
  "make_cfg", [unitree_g1_homie_env_cfg, unitree_h1_homie_env_cfg]
)
def test_homie_dr_follows_openhomie_ranges(make_cfg) -> None:
  from mjlab_homierl import mdp

  cfg = make_cfg()
  if make_cfg is unitree_g1_homie_env_cfg:
    # Deliberate deviation: OpenHomie's (-5, +10) is a field outlier (peers
    # use (-1, +3)) and our model under-weighs the battery by ~2 kg; see the
    # env cfg comment.
    assert cfg.events["payload_mass"].params["ranges"] == (-1.0, 5.0)
  else:
    assert cfg.events["payload_mass"].params["ranges"] == (-5.0, 10.0)
  assert cfg.events["foot_friction"].params["ranges"] == (0.1, 3.0)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.05, 0.05)
  # Actuation latency: training randomizes 0..decimation-1 substeps; play
  # runs the plant without it.
  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, mdp.DelayedJointPositionActionCfg)
  assert joint_pos.max_delay_substeps is None
  play = make_cfg(play=True)
  assert play.actions["joint_pos"].max_delay_substeps == 0


def test_g1_base_task_randomizes_wrist_payload() -> None:
  # Hand-agnostic training: the bare-wrist payload envelope covers Dex3
  # (0.53 kg), Inspire RH56DFX (0.54 kg), and a held object.
  cfg = unitree_g1_homie_env_cfg()
  params = cfg.events["hand_payload"].params
  assert params["ranges"] == (0.0, 1.5)
  assert params["asset_cfg"].body_names == (r".*_wrist_yaw_link",)


def test_h1_base_task_randomizes_wrist_payload() -> None:
  # Bare H1 arms end at the elbow (forearm) link; the envelope covers an
  # Inspire hand or 2F85 gripper plus a held object.
  cfg = unitree_h1_homie_env_cfg()
  params = cfg.events["hand_payload"].params
  assert params["ranges"] == (0.0, 2.0)
  assert params["asset_cfg"].body_names == (r".*_elbow_link",)
  # The mounted-gripper variant replaces it with a payload on the wrist links.
  hands_cfg = unitree_h1_homie_env_cfg(hands=True)
  assert hands_cfg.events["hand_payload"].params["asset_cfg"].body_names == (
    "left_wrist_link",
    "right_wrist_link",
  )


@pytest.mark.parametrize("hands", ["dex3", "inspire"])
def test_g1_hand_variants(hands) -> None:
  cfg = unitree_g1_homie_env_cfg(hands=hands)
  # The mounted hand replaces the bare-wrist payload DR (real hand mass +
  # held-object remainder on the mount body).
  assert "hand_payload" in cfg.events
  assert cfg.events["hand_payload"].params["ranges"] == (0.0, 1.0)
  # Interface must stay identical to the base task (checkpoint-compatible).
  base = unitree_g1_homie_env_cfg()
  assert (
    cfg.actions["joint_pos"].actuator_names == base.actions["joint_pos"].actuator_names
  )
  spec = cfg.scene.entities["robot"].spec_fn()
  hand_bodies = [b.name for b in spec.bodies if hands in b.name]
  assert f"left_{hands}/left_{hands}_mount" in hand_bodies
  assert f"right_{hands}/right_{hands}_mount" in hand_bodies

  with pytest.raises(ValueError):
    unitree_g1_homie_env_cfg(hands="unknown")


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
