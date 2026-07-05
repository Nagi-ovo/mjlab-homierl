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

# Opt-in extension: 1/3 of walk-mode resamples become in-place-turn commands
# (vx=vy=0, |wz|>=0.3; ~1/6 of all envs). The faithful OpenHomie sampler never
# produces pure rotation, so base-task policies stand through yaw-only
# commands; train (or resume) on this variant if step-turning in place is a
# deployment requirement. Interface-identical; checkpoints load both ways.
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-turn_mode",
  env_cfg=unitree_g1_homie_env_cfg(turn_prob=1.0 / 3.0),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, turn_prob=1.0 / 3.0),
  rl_cfg=unitree_g1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

# Superset variant: waist_roll/pitch join the random upper-body disturbance
# (the default task locks them at the default pose, matching OpenHomie's
# 27-dof G1). Interface-identical; checkpoints load both ways.
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-free_waist",
  env_cfg=unitree_g1_homie_env_cfg(waist="free"),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, waist="free"),
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

# G1 with real hand models mounted (inertial attachments; same obs/action
# interface, so base-task checkpoints load into these variants and vice versa).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-with_dex3",
  env_cfg=unitree_g1_homie_env_cfg(hands="dex3"),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, hands="dex3"),
  rl_cfg=unitree_g1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-with_inspire",
  env_cfg=unitree_g1_homie_env_cfg(hands="inspire"),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, hands="inspire"),
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
