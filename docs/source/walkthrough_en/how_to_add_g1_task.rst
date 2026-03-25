.. _walkthrough-en-how-to-add-g1-task:

How to Add a New G1 RL Task (From Zero to Trainable)
====================================================

This chapter gives you the shortest path:
**copy an existing task skeleton → replace MDP terms → register a task id → train / validate.**

In the end you will have a new task id that you can run directly:

.. code-block:: bash

   uv run train Mjlab-MyTask-Flat-Unitree-G1 --env.scene.num-envs 4096

Key design: tasks are “import-to-register”
------------------------------------------

Before resolving a task id, ``train.py`` / ``play.py`` will do:

.. code-block:: python

   import mjlab.tasks  # noqa: F401

And ``mjlab.tasks.__init__.py`` recursively imports subpackages (skipping blacklisted ones like ``.mdp``), which triggers each task’s registration code.

Paths: ``src/mjlab/tasks/__init__.py`` and ``src/mjlab/utils/lab_api/tasks/importer.py``

.. code-block:: python

   # file: src/mjlab/tasks/__init__.py
   from mjlab.utils.lab_api.tasks.importer import import_packages
   _BLACKLIST_PKGS = ["utils", ".mdp"]
   import_packages(__name__, _BLACKLIST_PKGS)

So: as long as your new task package lives under ``src/mjlab/tasks/<your_task>/`` and contains registration code, it will be discovered automatically.

Recommended folder skeleton
---------------------------

Using ``my_task`` as an example:

.. code-block:: text

   src/mjlab/tasks/my_task/
     __init__.py                  # can be empty (but a short docstring is nice)
     my_task_env_cfg.py           # base cfg (defines task MDP)
     mdp/
       __init__.py                # re-export envs.mdp + your own terms (like velocity/tracking)
       observations.py
       rewards.py
       terminations.py
       commands.py                # if you need CommandManager
       curriculums.py             # if you need CurriculumManager
     config/
       __init__.py
       g1/
         __init__.py              # register_mjlab_task (register task_id)
         env_cfgs.py              # G1 override (fill-in overrides)
         rl_cfg.py                # PPO/runner config
     rl/
       __init__.py
       runner.py                  # optional: custom runner (e.g., for exporting ONNX)

Step-by-step: from base cfg to runnable
--------------------------------------

Step 1: write the base cfg (task logic only, not robot-specific)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suggested path: ``src/mjlab/tasks/my_task/my_task_env_cfg.py``

Return a ``ManagerBasedRlEnvCfg`` and split MDP into dicts:

.. code-block:: python

   # file: src/mjlab/tasks/my_task/my_task_env_cfg.py
   from mjlab.envs import ManagerBasedRlEnvCfg
   from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
   from mjlab.managers.reward_manager import RewardTermCfg
   from mjlab.managers.termination_manager import TerminationTermCfg
   from mjlab.managers.scene_entity_config import SceneEntityCfg
   from mjlab.scene import SceneCfg
   from mjlab.sim import SimulationCfg, MujocoCfg
   from mjlab.terrains import TerrainImporterCfg
   from mjlab.tasks.my_task import mdp

   def make_my_task_env_cfg() -> ManagerBasedRlEnvCfg:
       observations = {
           "policy": ObservationGroupCfg(
               terms={
                   "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                   "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
               },
               concatenate_terms=True,
               enable_corruption=True,
           )
       }

       rewards = {
           "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
       }
       terminations = {
           "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
       }

       return ManagerBasedRlEnvCfg(
           scene=SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1),
           observations=observations,
           actions={...},     # reusing JointPositionActionCfg is easiest (see velocity/tracking)
           rewards=rewards,
           terminations=terminations,
           decimation=4,
           episode_length_s=10.0,
           sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
       )

**Guidelines:**

- keep the base cfg task-generic; leave robot differences (body/site/geom names, action scales, sensors) to overrides
- when referencing robot-specific pieces, use ``SceneEntityCfg("robot", joint_names=..., body_names=..., site_names=...)`` in params

Step 2: write MDP terms (reuse envs.mdp as much as possible)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like velocity/tracking, re-export in ``mdp/__init__.py``:

.. code-block:: python

   # file: src/mjlab/tasks/my_task/mdp/__init__.py
   from mjlab.envs.mdp import *  # generic: joint_pos_rel / time_out / randomize_field / ...
   from .rewards import *
   from .observations import *

Then in cfg you can simply do ``from mjlab.tasks.my_task import mdp`` and get both generic and custom terms.

Step 3: write the G1 override (fill placeholders)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suggested path: ``src/mjlab/tasks/my_task/config/g1/env_cfgs.py``

This layer typically:

- sets the entity: ``cfg.scene.entities = {"robot": get_g1_robot_cfg()}``
- adds sensors (e.g., contact sensors)
- sets ``JointPositionActionCfg.scale = G1_ACTION_SCALE``
- fills all ``SceneEntityCfg(..., body_names/site_names/geom_names=...)`` used by terms
- play mode: disable corruption, disable some randomization, set a very large episode_length

Step 4: register a task id (put cfg into the registry)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suggested path: ``src/mjlab/tasks/my_task/config/g1/__init__.py``

.. code-block:: python

   from mjlab.tasks.registry import register_mjlab_task
   from .env_cfgs import unitree_g1_my_task_env_cfg
   from .rl_cfg import unitree_g1_my_task_ppo_runner_cfg

   register_mjlab_task(
     task_id="Mjlab-MyTask-Flat-Unitree-G1",
     env_cfg=unitree_g1_my_task_env_cfg(),
     play_env_cfg=unitree_g1_my_task_env_cfg(play=True),
     rl_cfg=unitree_g1_my_task_ppo_runner_cfg(),
     runner_cls=None,  # or a custom runner
   )

Step 5: play (dummy agent) before train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run play Mjlab-MyTask-Flat-Unitree-G1 --agent random
   uv run train Mjlab-MyTask-Flat-Unitree-G1 --env.scene.num-envs 1024

Common pitfalls (roughly by frequency)
--------------------------------------

- **Shape mismatch**: obs terms must return ``(num_envs, ...)``; reward/termination terms must return ``(num_envs,)``.
- **Device mismatch**: don’t create CPU tensors inside terms by default; use ``device=env.device``.
- **SceneEntityCfg placeholders not filled**: base cfg may leave ``geom_names=()`` / ``site_names=()``; forgetting to fill in overrides leads to empty ids or unexpected behavior.
- **Too many Python loops**: it gets slow at 4096 envs; prefer vectorized torch ops.
- **Tracking motion_file not injected**: for tracking-like tasks, follow train.py flow (``--registry-name`` or ``--motion-file``).
