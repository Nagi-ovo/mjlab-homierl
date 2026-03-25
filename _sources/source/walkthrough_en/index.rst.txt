.. _developer-walkthrough-en:

Developer Walkthrough (External HOMIE Package)
============================================================

This walkthrough explains the refactored ``mjlab-homierl`` layout: what still
lives upstream in ``mjlab``, what now lives in this repository, and where to
change HOMIE-specific behavior without confusing package-local code with
framework internals.

What you will learn here
-------------------------

- **Package boundaries**: which pieces are in upstream ``mjlab`` vs this repo.
- **Code navigation**: where HOMIE env cfg, assets, MDP terms, and HIMPPO live.
- **Task deep-dive**: how the H1 HOMIE task is assembled, including the
  ``with_hands`` variant and play-time overrides.

Recommended reading order
-------------------------

If this is your first time reading the refactored repository, follow the order
below. The pages are intentionally package-specific instead of trying to mirror
all upstream ``mjlab`` framework docs.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   project_layout
   tasks_homie_h1
