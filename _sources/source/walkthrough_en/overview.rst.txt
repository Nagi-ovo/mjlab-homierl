.. _walkthrough-en-overview:

Architecture Overview: External Package on Top of mjlab
================================================================

The one sentence that matters most
-----------------------------------

**``mjlab-homierl`` is not the framework anymore; it is a task package that plugs into upstream ``mjlab``.**

- Upstream ``mjlab`` still owns ``ManagerBasedRlEnv``, managers, scene,
  simulation, CLI entrypoints, and viewer integrations.
- This repository owns HOMIE-specific env cfgs, H1 / 2F85 assets, the HIMPPO
  runner, and package-local tests and docs.

This distinction matters because many old paths from the vendored codebase no
longer exist here under ``src/mjlab/...``.

From task id to the training loop (end-to-end)
----------------------------------------------

1. **Task registration**

   - Path in this repo: ``src/mjlab_homierl/__init__.py``
   - API used: ``mjlab.tasks.registry.register_mjlab_task``
   - Result: upstream ``mjlab`` discovers
     ``Mjlab-Homie-Unitree-H1`` and ``Mjlab-Homie-Unitree-H1-with_hands`` via
     the ``mjlab.tasks`` entry-point group

2. **Train / play entrypoint**

   - CLI comes from upstream ``mjlab``: ``uv run train ...`` and
     ``uv run play ...``
   - The CLI loads the registered env cfg / rl cfg, constructs
     ``ManagerBasedRlEnv``, wraps it, and instantiates the configured runner

3. **Package-local task assembly**

   - Base HOMIE config: ``src/mjlab_homierl/homie_env_cfg.py``
   - H1 and with-hands overrides: ``src/mjlab_homierl/env_cfgs.py``
   - Task-specific MDP terms: ``src/mjlab_homierl/mdp/*``
   - Custom runner: ``src/mjlab_homierl/rl/runner.py``

4. **Runtime split**

   - Training uses HIMPPO with both actor and critic observations
   - Play can use a HOMIE-specific actor-only inference path when the play env
     strips the critic group

A readable control-flow sketch
------------------------------

.. code-block:: text

   uv run train/play -> upstream mjlab CLI
     |
     v
   task registry entry -> env cfg + rl cfg + runner class
     |
     v
   ManagerBasedRlEnv(cfg=...) + RslRlVecEnvWrapper
     |
     v
   HomieHimOnPolicyRunner
     |
     v
   policy(obs) -> action
     |
     v
   upstream env step:
     ActionManager -> Simulation -> Managers -> observations / rewards / resets
     |
     v
   train loop or viewer loop

Where to start reading code
---------------------------

- **Package entrypoint + task registration**: ``src/mjlab_homierl/__init__.py``
- **Base HOMIE task config**: ``src/mjlab_homierl/homie_env_cfg.py``
- **H1 task overrides and play behavior**: ``src/mjlab_homierl/env_cfgs.py``
- **Assets and hand attachment logic**:
  ``src/mjlab_homierl/robots/unitree_h1/h1_constants.py``
- **If you need framework internals**: inspect the installed upstream ``mjlab``
  package, especially ``mjlab.envs``, ``mjlab.managers``, ``mjlab.scene``, and
  ``mjlab.sim``
