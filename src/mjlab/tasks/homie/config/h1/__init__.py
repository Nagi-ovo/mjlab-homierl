from mjlab.tasks.homie.rl import HomieOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_h1_homie_env_cfg
from .rl_cfg import unitree_h1_homie_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-H1",
  env_cfg=unitree_h1_homie_env_cfg(),
  play_env_cfg=unitree_h1_homie_env_cfg(play=True),
  rl_cfg=unitree_h1_homie_ppo_runner_cfg(),
  runner_cls=HomieOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Homie-Unitree-H1-with_hands",
  env_cfg=unitree_h1_homie_env_cfg(hands=True),
  play_env_cfg=unitree_h1_homie_env_cfg(play=True, hands=True),
  rl_cfg=unitree_h1_homie_ppo_runner_cfg(),
  runner_cls=HomieOnPolicyRunner,
)
