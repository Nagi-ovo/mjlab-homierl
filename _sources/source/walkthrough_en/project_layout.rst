.. _walkthrough-en-project-layout:

Code Map: Where Should You Change Things?
========================================

This chapter is “directory-level navigation”. When you already know what you want to change (reward/obs/sim/robot/training scripts), jump to the right folder directly.

Key directories
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Directory
     - What you do here
   * - ``src/mjlab/envs/``
     - The manager-based environment core (``ManagerBasedRlEnv``), plus generic ``mdp`` components (actions/obs/rewards/terminations/events).
   * - ``src/mjlab/managers/``
     - Core abstractions: ``ManagerBase``, managers (action/obs/reward/termination/command/curriculum/event), and term cfgs.
   * - ``src/mjlab/scene/``
     - ``SceneCfg`` / ``Scene``: assemble terrain/entities/sensors into a MuJoCo ``MjSpec``, and manage env origins for vectorized envs.
   * - ``src/mjlab/sim/``
     - ``Simulation``: MuJoCo + MuJoCo-Warp GPU backend, CUDA graphs, model field expansion, NaN guard.
   * - ``src/mjlab/entity/``
     - Data interface for robots/objects (root/joint/body/site). Most task terms read data from here.
   * - ``src/mjlab/sensor/``
     - Sensors (built-in sensors, contact sensors, etc.). Many obs/rewards read from ``env.scene[sensor_name]``.
   * - ``src/mjlab/asset_zoo/``
     - Asset library (MJCF/XML). G1 is in ``asset_zoo/robots/unitree_g1``, H1 is in ``asset_zoo/robots/unitree_h1``.
   * - ``src/mjlab/tasks/``
     - Task packages. A task typically contains:
       ``<task>_env_cfg.py`` (base cfg), ``config/<robot>/`` (robot overrides + registry), ``mdp/`` (task terms), ``rl/`` (runner/export).
   * - ``src/mjlab/rl/``
     - RSL-RL integration: VecEnv wrapper, policy/algorithm cfg dataclasses.
   * - ``src/mjlab/scripts/``
     - CLI entrypoints: ``train``, ``play`` (implemented in ``train.py`` / ``play.py``), demos, utility scripts.
   * - ``docs/``
     - Sphinx documentation (this walkthrough lives here too).

What does a “standard task skeleton” look like?
----------------------------------------------

Using velocity as an example (applies to humanoids and quadrupeds):

- Base task cfg: ``src/mjlab/tasks/velocity/velocity_env_cfg.py``
- Robot override (e.g., g1/go1): ``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``
- Register task id: ``src/mjlab/tasks/velocity/config/g1/__init__.py``
- MDP terms: ``src/mjlab/tasks/velocity/mdp/*`` (plus reusable ``src/mjlab/envs/mdp/*``)

Similarly for tracking:

- Base task cfg: ``src/mjlab/tasks/tracking/tracking_env_cfg.py``
- g1 override + registry: ``src/mjlab/tasks/tracking/config/g1/*``
- MDP terms: ``src/mjlab/tasks/tracking/mdp/*``
