.. _developer-walkthrough-en:

Developer Walkthrough (Research & Development)
==============================================

The goal of this walkthrough is to help new contributors build a reliable mental model of the codebase in **1–2 hours**, and start modifying / creating manager-based RL tasks with confidence.

What you will learn here
-----------------------

- **Architecture overview**: how ``ManagerBasedRlEnv = Scene + Simulation + Managers`` fits together (data flow + control flow).
- **The manager-based API**: design philosophy, extension points, and how **rewards** / **terminations** are configured.
- **Task development workflow**: how to build a task in mjlab (dict-based cfg), similar to Isaac Lab’s manager-based tasks.
- **Task deep-dives (G1/H1)**: a guided tour of ``tasks/velocity``, ``tasks/tracking``, and ``tasks/homie`` (cfg / MDP / training entrypoints).

Recommended reading order
-------------------------

If this is your first time reading the code, follow the order below (overview → env lifecycle → managers/terms → task examples → build your own).

If you already know Isaac Lab well, you can jump to ``managers_and_terms`` and then compare the task chapters with the code directly.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   quickstart
   project_layout
   manager_based_env
   managers_and_terms
   rewards_and_terminations
   tasks_velocity_g1
   tasks_tracking_g1
   tasks_homie_h1
   how_to_add_g1_task
   debugging_perf
