from mjlab.tasks.registry import register_mjlab_task

from mjlab_homierl.env_cfgs import (
  unitree_g1_homie_env_cfg,
  unitree_h1_homie_env_cfg,
)
from mjlab_homierl.rl import HomieHimOnPolicyRunner
from mjlab_homierl.rl_cfg import (
  unitree_g1_homie_himppo_runner_cfg,
  unitree_g1_homie_plus_himppo_runner_cfg,
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

# Frozen OpenHomie-parity preset: the diff between this cfg and the default
# task is the authoritative ledger of our deliberate deviations (payload DR,
# wrist payload DR, hip/knee contact penalty). Reference baseline — never
# iterate on this entry.
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-native",
  env_cfg=unitree_g1_homie_env_cfg(native=True),
  play_env_cfg=unitree_g1_homie_env_cfg(play=True, native=True),
  rl_cfg=unitree_g1_homie_himppo_runner_cfg(),
  runner_cls=HomieHimOnPolicyRunner,
)

# HOMIE+ (homie_plus_plan.md): the deployment fork. Commanded torso pitch
# (5-dim command, one-step obs 81, actor 486 -- checkpoints do NOT interchange
# with the base task; waist_pitch command-driven, waist_roll locked, waist_yaw
# disturbed), plus two deployment-motivated extensions over OpenHomie parity:
# in-place locomotion sampling (1/3 of walk resamples: vx=0, vy/wz kept with
# the dominant axis clamped >= 0.3 -- fixes the dead pure-strafe / pure-turn
# corners found on hardware) and per-env foot contact-compliance DR
# (arXiv:2504.13619; fixes standing sway on foam mats).
register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-G1-plus",
  env_cfg=unitree_g1_homie_env_cfg(
    torso_pitch=True, inplace_prob=1.0 / 3.0, floor="compliant"
  ),
  play_env_cfg=unitree_g1_homie_env_cfg(
    play=True, torso_pitch=True, inplace_prob=1.0 / 3.0, floor="compliant"
  ),
  rl_cfg=unitree_g1_homie_plus_himppo_runner_cfg(),
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
