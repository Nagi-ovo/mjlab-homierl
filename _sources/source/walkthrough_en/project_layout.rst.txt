.. _walkthrough-en-project-layout:

Code Map: Where Should You Change Things?
============================================================

This chapter is package-level navigation for the refactored external task
layout. The key distinction is that this repository only contains HOMIE-specific
extensions; the general framework directories now live upstream in the installed
``mjlab`` package.

Key directories
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Directory
     - What you do here
   * - ``src/mjlab_homierl/__init__.py``
     - Entry-point task registration. This is where the two HOMIE task ids are
       registered against env cfg, rl cfg, and runner class.
   * - ``src/mjlab_homierl/homie_env_cfg.py``
     - Base HOMIE manager configuration: commands, observations, rewards,
       terminations, events, and curriculum shared by H1 variants.
   * - ``src/mjlab_homierl/env_cfgs.py``
     - H1-specific overrides: lower-body action split, upper-body disturbance
       action, contact sensors, play-time overrides, and ``with_hands`` logic.
   * - ``src/mjlab_homierl/mdp/``
     - HOMIE-specific commands, observations, rewards, terminations, and
       curriculum helpers.
   * - ``src/mjlab_homierl/rl_cfg.py``
     - Actor / critic / algorithm dataclasses aligned to modern ``mjlab`` RL
       config structure.
   * - ``src/mjlab_homierl/rl/``
     - Package-local HIMPPO implementation, custom runner, and ONNX export.
   * - ``src/mjlab_homierl/robots/``
     - H1 XML assets, Robotiq assets, and spec assembly helpers.
   * - ``src/mjlab/scripts/``
     - No longer part of this repository. ``train`` and ``play`` come from the
       installed upstream ``mjlab`` package.
   * - ``tests/``
     - Package-only regression tests for task registration, env cfgs, RL cfgs,
       and H1 asset assembly.
   * - ``docs/``
     - Sphinx documentation (this walkthrough lives here too).

What is upstream now?
---------------------

When you need framework internals, look in upstream ``mjlab`` for:

- ``mjlab.envs``: ``ManagerBasedRlEnv`` and generic env utilities
- ``mjlab.managers``: manager classes and term cfg machinery
- ``mjlab.scene`` / ``mjlab.sim``: scene assembly and MuJoCo runtime
- ``mjlab.rl``: vec-env wrapper and base runner integration
