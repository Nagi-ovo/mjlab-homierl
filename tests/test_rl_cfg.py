from mjlab_homierl.rl_cfg import (
  unitree_g1_homie_himppo_runner_cfg,
  unitree_h1_homie_himppo_runner_cfg,
)


def test_himppo_runner_cfg_uses_custom_algorithm() -> None:
  for cfg in (
    unitree_g1_homie_himppo_runner_cfg(),
    unitree_h1_homie_himppo_runner_cfg(),
  ):
    assert cfg.algorithm.class_name == "mjlab_homierl.rl.himppo.algorithm.HIMPPO"
    # OpenHomie HIMActorCritic hidden dims.
    assert cfg.actor.hidden_dims == (512, 256, 256)
    assert cfg.critic.hidden_dims == (512, 256, 256)
    # Checkpoints must never be uploaded to W&B.
    assert cfg.upload_model is False
