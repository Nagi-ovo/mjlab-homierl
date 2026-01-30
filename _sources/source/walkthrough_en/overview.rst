.. _walkthrough-en-overview:

Architecture Overview: mjlab in One Diagram
==========================================

.. figure:: ../_static/content/mjlab_architecture_overview.svg
   :width: 100%
   :alt: mjlab architecture overview

The one sentence that matters most
---------------------------------

**mjlab is a manager-based RL environment:**

- ``Scene``: organizes **MuJoCo assets / sensors / terrains** into a vectorized scene (``num_envs``), and handles writing entity state into the simulator.
- ``Simulation``: runs MuJoCo + MuJoCo-Warp physics on GPU (step/forward/reset), and provides low-level features like CUDA graphs and NaN guards.
- ``Managers``: decomposes a task into composable modules: **actions / observations / rewards / terminations / commands / events / curriculum**.

Intuition: the “environment shell” is thin; most task logic lives in manager terms.

From task id to the training loop (end-to-end)
----------------------------------------------

1. **Register a task** (put ``env_cfg`` / ``rl_cfg`` into the registry)

   - Path: ``src/mjlab/tasks/<task>/config/<robot>/__init__.py``
   - API: ``src/mjlab/tasks/registry.py::register_mjlab_task``

2. **Training entrypoint** (load cfg from registry, build env + runner)

   - Path: ``src/mjlab/scripts/train.py``
   - Key chain: ``load_env_cfg()`` → ``ManagerBasedRlEnv(cfg=..., device=...)`` → ``RslRlVecEnvWrapper`` → ``rsl_rl.OnPolicyRunner``

3. **Environment step** (action → physics → terminations/rewards → events/commands → observations)

   - Path: ``src/mjlab/envs/manager_based_rl_env.py::ManagerBasedRlEnv.step``

4. **Where task logic lives** (MDP components)

   - Path: ``src/mjlab/envs/mdp/*`` (generic) + ``src/mjlab/tasks/<task>/mdp/*`` (task-specific)
   - Used by managers via ``RewardTermCfg(func=..., params=...)`` / ``ObservationTermCfg(func=..., ...)`` in cfg

A readable “data flow / control flow” sketch
--------------------------------------------

.. code-block:: text

   policy(obs) -> action
     |
     v
   ActionManager.process_action / apply_action
     |
     v
   for decimation steps:
     Scene.write_data_to_sim -> Simulation.step -> Scene.update
     |
     v
   TerminationManager.compute  -> reset mask
   RewardManager.compute(dt)   -> reward
   (if any reset) -> _reset_idx -> Event(reset) -> Managers.reset(...)
     |
     v
   CommandManager.compute(dt)
   EventManager.apply(interval, dt)
   ObservationManager.compute(update_history=True)
     |
     v
   return obs, reward, terminated, truncated, extras

Where to start reading code
---------------------------

- **Env lifecycle + manager loading order**: ``src/mjlab/envs/manager_based_rl_env.py``
- **Manager / term bases + “function vs class term” mechanics**: ``src/mjlab/managers/manager_base.py`` + ``src/mjlab/managers/*_manager.py``
- **SceneEntityCfg (late binding: names → ids)**: ``src/mjlab/managers/scene_entity_config.py``
- **Two G1 tasks**:

  - velocity: ``src/mjlab/tasks/velocity/velocity_env_cfg.py`` + ``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``
  - tracking: ``src/mjlab/tasks/tracking/tracking_env_cfg.py`` + ``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``
