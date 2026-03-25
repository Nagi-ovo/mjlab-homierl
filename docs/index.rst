Welcome to mjlab-homierl
========================

``mjlab-homierl`` is an external ``mjlab`` task package for the Unitree H1
HOMIE locomotion setup. This repository no longer vendors the full framework:
upstream ``mjlab`` provides the manager-based environment core, CLI, viewer, and
simulation stack, while this package contributes HOMIE-specific task configs,
assets, and the HIMPPO runner.

What lives in this repository
=============================

- Two registered tasks: ``Mjlab-Homie-Unitree-H1`` and
  ``Mjlab-Homie-Unitree-H1-with_hands``
- The external package under ``src/mjlab_homierl/``
- H1 and Robotiq 2F85 assets used by HOMIE
- A package-local HIMPPO runner and ONNX export path
- Focused regression tests for task registration, env configs, RL configs, and
  H1 asset assembly

Relationship to upstream mjlab
==============================

- ``uv run train ...`` and ``uv run play ...`` come from the installed
  ``mjlab`` package
- Tasks are discovered through the ``mjlab.tasks`` entry-point group
- Framework internals such as ``ManagerBasedRlEnv``, managers, scene, and
  simulation are upstream concerns, not part of this repo anymore

Install and day-to-day command examples live in the repository ``README.md`` at
the project root.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Developer Walkthrough (English)

   source/walkthrough_en/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Walkthrough（中文）

   source/walkthrough/index
