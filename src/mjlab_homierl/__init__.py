from mjlab.tasks.registry import register_mjlab_task

from mjlab_homierl.env_cfgs import (
  unitree_g1_homie_env_cfg,
  unitree_h1_homie_env_cfg,
)
from mjlab_homierl.rl import HomieHimOnPolicyRunner
from mjlab_homierl.rl_cfg import (
  unitree_g1_homie_himppo_runner_cfg,
  unitree_h1_homie_himppo_runner_cfg,
)

# Default G1 task trains with deployment-grade PD gains (sim2real).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1",
  env_cfg=unitree_g1_homie_env_cfg(),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True),
  rl_cfg=unitree_g1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

# Ablation variant with mjlab's first-principles actuator gains (sim-only).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-mjlab_gains",
  env_cfg=unitree_g1_homie_env_cfg(gains="mjlab"),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, gains="mjlab"),
  rl_cfg=unitree_g1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

# Default H1 task trains with deployment-grade PD gains (sim2real).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-H1",
  env_cfg=unitree_h1_homie_env_cfg(),
  play_env_cfg=unitree_h1_homie_env_cfg(play=True),
  rl_cfg=unitree_h1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

# Ablation variant with mjlab's first-principles actuator gains (sim-only).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-H1-mjlab_gains",
  env_cfg=unitree_h1_homie_env_cfg(gains="mjlab"),
  play_env_cfg=unitree_h1_homie_env_cfg(play=True, gains="mjlab"),
  rl_cfg=unitree_h1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-H1-with_hands",
  env_cfg=unitree_h1_homie_env_cfg(hands=True),
  play_env_cfg=unitree_h1_homie_env_cfg(play=True, hands=True),
  rl_cfg=unitree_h1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)
