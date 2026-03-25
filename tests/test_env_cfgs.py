from mjlab_homierl.env_cfgs import unitree_h1_homie_env_cfg


def test_homie_env_cfg_uses_actor_critic_groups() -> None:
  cfg = unitree_h1_homie_env_cfg()
  assert "actor" in cfg.observations
  assert "critic" in cfg.observations
  assert "policy" not in cfg.observations
  assert "joint_pos" in cfg.actions


def test_homie_env_cfg_with_hands_adds_gripper_action() -> None:
  cfg = unitree_h1_homie_env_cfg(hands=True)
  assert "gripper" in cfg.actions
  assert cfg.sim.mujoco.ccd_iterations == 50


def test_homie_env_cfg_without_hands_raises_ccd_budget_above_default() -> None:
  cfg = unitree_h1_homie_env_cfg()
  assert cfg.sim.mujoco.ccd_iterations == 50


def test_homie_play_cfg_strips_training_only_work() -> None:
  cfg = unitree_h1_homie_env_cfg(play=True, hands=True)
  assert "actor" in cfg.observations
  assert "critic" not in cfg.observations
  assert cfg.rewards == {}
  assert cfg.curriculum == {}
