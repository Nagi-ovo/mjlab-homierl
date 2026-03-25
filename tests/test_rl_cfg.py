from mjlab_homierl.rl_cfg import unitree_h1_homie_himppo_runner_cfg


def test_himppo_runner_cfg_uses_custom_algorithm() -> None:
  cfg = unitree_h1_homie_himppo_runner_cfg()
  assert cfg.algorithm.class_name == "mjlab_homierl.rl.himppo.algorithm.HIMPPO"
  assert cfg.actor.hidden_dims == (512, 256, 128)
  assert cfg.critic.hidden_dims == (512, 256, 128)
