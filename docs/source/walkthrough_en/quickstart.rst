.. _walkthrough-en-quickstart:

Quickstart: The Shortest Dev Loop (Train / Play / Modify Tasks)
==============================================================

This chapter only answers three questions:

- **How do I run a baseline quickly?**
- **How do I validate my MDP is not broken?**
- **Where do I iterate fastest (rewards / obs / randomization)?**

0) Run a working baseline first
-------------------------------

velocity (Unitree G1, flat):

.. code-block:: bash

   # Train: common overrides (tyro can override dataclass fields directly)
   uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

   # Play: load latest checkpoint from W&B (or pass --checkpoint-file)
   uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id

tracking (Unitree G1, motion imitation):

.. code-block:: bash

   # Tracking requires a motion registry (W&B artifact). train.py injects motion_file into the command cfg.
   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
     --registry-name your-org/motions/motion-name \
     --env.scene.num-envs 4096

   uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id

homie (Unitree H1, mixed velocity + squat + disturbances):

.. code-block:: bash

   # H1 is heavier and the task is more complex; start with fewer envs first.
   uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 2048

   uv run play Mjlab-Homie-Unitree-H1 --wandb-run-path your-org/mjlab/run-id

Homie also provides an optional “with hands” variant (mounts Robotiq 2F85, and adds policy-free random gripper motion):

.. code-block:: bash

   uv run train Mjlab-Homie-Unitree-H1-with_hands --env.scene.num-envs 2048
   uv run play Mjlab-Homie-Unitree-H1-with_hands --wandb-run-path your-org/mjlab/run-id

1) Use a dummy agent for a quick “MDP health check” (strongly recommended)
--------------------------------------------------------------------------

Before training, run the env for a few hundred steps with zero / random actions:

.. code-block:: bash

   uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero
   uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent random

Tracking dummy agents still need a motion registry (otherwise commands cannot load motions):

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
     --agent random \
     --registry-name your-org/motions/motion-name

Homie dummy agents do not need extra arguments:

.. code-block:: bash

   uv run play Mjlab-Homie-Unitree-H1 --agent zero
   uv run play Mjlab-Homie-Unitree-H1 --agent random

   # Same for the with_hands variant (gripper is policy-free, 0-dim action)
   uv run play Mjlab-Homie-Unitree-H1-with_hands --agent random

What should you look for?

- **Stability**: NaN/Inf, exploding values (enable NaN guard if needed, see :ref:`walkthrough-en-debugging-perf`).
- **Obs/action shapes**: mismatched dims usually fail early in ``ActionManager`` or when building gym spaces.
- **Reward signals**: check viewer / logger (metrics are available in ``extras``).

2) Where do I modify things fastest?
------------------------------------

These tasks follow the pattern: **base env cfg + robot-specific override**.

- base cfg (task definition):

  - velocity: ``src/mjlab/tasks/velocity/velocity_env_cfg.py::make_velocity_env_cfg``
  - tracking: ``src/mjlab/tasks/tracking/tracking_env_cfg.py::make_tracking_env_cfg``

- g1 overrides (fill-in overrides):

  - velocity: ``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``
  - tracking: ``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``

Homie uses the same structure (base cfg + H1 override):

- base cfg: ``src/mjlab/tasks/homie/homie_env_cfg.py::make_homie_env_cfg``
- h1 override: ``src/mjlab/tasks/homie/config/h1/env_cfgs.py::unitree_h1_homie_env_cfg``

If you want to modify Homie, start with :ref:`walkthrough-en-task-homie-h1`. The most common knobs are:

- **Action split / scaling**: policy controls legs only; upper body / gripper are policy-free actions.
- **Commands & env grouping**: who walks vs squats vs stands (``mdp.assign_homie_env_groups``).
- **Disturbances / randomization**: pushes, hand loads, friction randomization (``events``).

Suggested iteration order (fastest feedback first):

- **Reward weights**: tweak ``RewardTermCfg(weight=...)`` first.
- **Observation terms**: add/remove terms in ``observations["policy"].terms``.
- **Randomization**: edit ``events`` (startup/reset/interval) and domain randomization fields.
- **Command distributions**: adjust ranges and sampling modes in ``commands``.

3) How does CLI override configs?
--------------------------------

``train.py`` / ``play.py`` use tyro to parse dataclasses, so you can override fields directly from CLI:

.. code-block:: bash

   # Override num_envs, episode length, viewer resolution, and (example) a reward weight
   uv run train Mjlab-Velocity-Flat-Unitree-G1 \
     --env.scene.num-envs 2048 \
     --env.episode-length-s 15 \
     --env.viewer.width 1280 --env.viewer.height 720

.. note::

   Tyro overrides work best for dataclass fields. For “deep” keys inside dicts (e.g., a specific reward term),
   it is usually cleaner to edit the corresponding ``config/<robot>/env_cfgs.py`` in Python to keep CLI usage maintainable.
