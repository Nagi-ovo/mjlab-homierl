.. _walkthrough-en-debugging-perf:

Debugging & Performance: Stay Stable and Fast
=============================================

This chapter is not an API restatement. It turns common engineering issues into a practical debugging order and a few performance heuristics.

Priority 1: make sure the MDP is correct
----------------------------------------

1) Dummy-agent sanity checks (zero/random)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/scripts/play.py``

.. code-block:: bash

   uv run play <task_id> --agent zero
   uv run play <task_id> --agent random

Before the policy learns anything, this helps you validate:

- action/observation spaces match what you expect
- resets are stable
- reward/termination terms do not produce NaN/Inf
- dependencies (e.g., contact sensors) are configured correctly

2) Visualize “command vs actual” (command debug vis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Velocity ``UniformVelocityCommand`` draws velocity arrows (blue/green = command, cyan/light-green = actual):

- Path: ``src/mjlab/tasks/velocity/mdp/velocity_command.py::_debug_vis_impl``

Tracking ``MotionCommand`` can visualize a ghost robot or frames:

- Path: ``src/mjlab/tasks/tracking/mdp/commands.py::_debug_vis_impl``

Priority 2: NaN/Inf and physics instability
-------------------------------------------

1) NaN guard (recommended during development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paths: ``src/mjlab/sim/sim.py`` (``NanGuard``) + ``src/mjlab/scripts/train.py`` (flag wiring)

Enable it during training:

.. code-block:: bash

   uv run train <task_id> --enable-nan-guard True

train.py sets ``cfg.env.sim.nan_guard.enabled = True``.

2) NaN as a termination (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/envs/mdp/terminations.py::nan_detection``

If you prefer “terminate and reset immediately when NaN happens”, add it to your task’s ``terminations`` dict.

Priority 3: the right way to do domain randomization
----------------------------------------------------

The core of domain randomization is not “randomize each reset”, but **supporting per-env model fields**.

1) Mark the event with domain_randomization=True
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paths: ``src/mjlab/envs/mdp/events.py::randomize_field`` + ``src/mjlab/managers/event_manager.py``

.. code-block:: python

   EventTermCfg(
     mode="startup",
     func=mdp.randomize_field,
     domain_randomization=True,
     params={"field": "geom_friction", ...},
   )

``EventManager`` collects ``params["field"]`` into ``domain_randomization_fields``.

2) Expand fields during load_managers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/envs/manager_based_rl_env.py::load_managers``

.. code-block:: python

   self.event_manager = EventManager(self.cfg.events, self)
   self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

3) CUDA graph pitfall: recapture after replacing arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/sim/sim.py`` (class notes + ``expand_model_fields``)

Simulation captures CUDA graphs for step/forward/reset, which are bound to the GPU memory addresses at capture time.
If you later replace model/data arrays, the graph may silently read old addresses.

Good news: ``Simulation.expand_model_fields`` automatically calls ``create_graph()`` to recapture.

Engineering advice:

- Decide all fields that need expansion at env init time (via EventManager collection) and avoid replacing arrays mid-run.

Performance heuristics: make training faster
--------------------------------------------

1) Knobs that matter most
^^^^^^^^^^^^^^^^^^^^^^^^^

- **num_envs**: larger is heavier on GPU but amortizes Python overhead (4096 is common for velocity/tracking)
- **decimation**: larger → lower control rate; physics still runs ``decimation`` steps per env step
- **term vectorization**: avoid Python ``for`` loops inside terms
- **contact sensor complexity**: match patterns, number of slots, reduce strategy all affect cost

2) Multi-GPU
^^^^^^^^^^^^

Path: ``src/mjlab/scripts/train.py`` (``--gpu-ids`` + torchrunx)

Multi-GPU uses torchrunx to launch multiple processes and aligns ``MUJOCO_EGL_DEVICE_ID`` with local_rank.

Observability: what is the “truth source” per step?
---------------------------------------------------

- **env.extras["log"]**: per-manager reset stats (episodic reward sums, termination counts, curriculum state, command metrics, ...)
  Path: ``src/mjlab/envs/manager_based_rl_env.py::_reset_idx``

- **RewardManager / CommandTerm metrics**: many tasks write metrics into ``env.extras["log"]["Metrics/..."]``
  Example: velocity metrics like angular_momentum_mean, air_time_mean, slip_velocity_mean (see ``src/mjlab/tasks/velocity/mdp/rewards.py``)
