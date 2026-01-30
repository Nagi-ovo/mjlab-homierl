.. _walkthrough-en-managers-and-terms:

Key Class 2: Managers + Terms (IsaacLab-like Core Abstractions)
=============================================================

If you only remember one transferable idea, it is:

**A task = a set of manager dictionaries; each manager = multiple terms; each term = a callable defined by (func/class, params).**

This chapter breaks the mechanism down to the level where you can implement a new term (or a new term cfg) yourself.

Three layers: TermCfg → Term → Manager
-------------------------------------

1) TermCfg: a dataclass that describes “how to build / how to call”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/manager_base.py`` (``ManagerTermBaseCfg``) + ``src/mjlab/managers/*_manager.py`` (specific ``*TermCfg``)

- ``RewardTermCfg`` / ``TerminationTermCfg`` / ``CurriculumTermCfg`` / ``EventTermCfg``:
  inherit from ``ManagerTermBaseCfg``; the core fields are ``func`` and ``params`` (plus reward’s ``weight``, etc.).
- ``ObservationTermCfg``: additionally defines the processing pipeline (noise/clip/scale/delay/history).
- ``ActionTermCfg``: abstract; must implement ``build(env) -> ActionTerm``.
- ``CommandTermCfg``: abstract; must implement ``build(env) -> CommandTerm`` (used by ``CommandManager`` to build command terms).

.. code-block:: python

   # file: src/mjlab/managers/manager_base.py
   @dataclass
   class ManagerTermBaseCfg:
       func: Any
       params: dict[str, Any] = field(default_factory=lambda: {})

   # file: src/mjlab/managers/reward_manager.py
   @dataclass(kw_only=True)
   class RewardTermCfg(ManagerTermBaseCfg):
       func: Any
       weight: float

   # file: src/mjlab/managers/observation_manager.py
   @dataclass
   class ObservationTermCfg(ManagerTermBaseCfg):
       # Processing pipeline: compute → noise → clip → scale → delay → history.
       noise: NoiseCfg | NoiseModelCfg | None = None
       clip: tuple[float, float] | None = None
       scale: tuple[float, ...] | float | torch.Tensor | None = None
       delay_min_lag: int = 0
       delay_max_lag: int = 0
       history_length: int = 0

2) Term: function terms vs stateful class terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Managers call terms in a mostly uniform way: ``term(env, **params)``.

**Function terms**: best for pure computations with no internal state (e.g., reading entity/sensor values, simple rewards).

**Class terms**: best when you need caching / cross-step statistics / preprocessing (e.g., pre-resolving pattern→tensor mappings, tracking peak heights, maintaining EMAs).

You can see typical examples in velocity rewards:

``feet_swing_height`` and ``variable_posture`` are class terms (path: ``src/mjlab/tasks/velocity/mdp/rewards.py``).

3) Manager: assemble terms + own lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each manager does two things:

- Assemble cfg dict/list into callable terms in ``_prepare_terms`` (including resolution and buffer allocation)
- Schedule terms in ``compute`` / ``reset``

Core mechanism: SceneEntityCfg “late binding” + automatic class-term instantiation
-------------------------------------------------------------------------------

SceneEntityCfg: why it exists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you write in cfg:

.. code-block:: python

   SceneEntityCfg("robot", joint_names=(".*",))

you are effectively saying: “use **all joints** of the entity named ``robot`` in the scene”.

But at cfg construction time, there is no scene/model yet, so names cannot be resolved into indices.
mjlab therefore uses **late binding**: resolve names after env init, when the scene/model exist.

Path: ``src/mjlab/managers/scene_entity_config.py``

.. code-block:: python

   # file: src/mjlab/managers/scene_entity_config.py
   @dataclass
   class SceneEntityCfg:
       name: str
       joint_names: str | tuple[str, ...] | None = None
       joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))
       # ... body/geom/site/actuator similar ...

       def resolve(self, scene: Scene) -> None:
           entity = scene[self.name]
           # For each field: keep names <-> ids consistent, and optimize “select all” to slice(None)
           ...

ManagerBase._resolve_common_term_cfg: two key actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/manager_base.py``

.. code-block:: python

   # file: src/mjlab/managers/manager_base.py
   def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg):
       # 1) Automatically resolve SceneEntityCfg that appears in params
       for value in term_cfg.params.values():
           if isinstance(value, SceneEntityCfg):
               value.resolve(self._env.scene)

       # 2) If func is a class, auto-instantiate it into a callable object (inject cfg + env)
       if inspect.isclass(term_cfg.func):
           term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

This is why you can write in cfg:

.. code-block:: python

   RewardTermCfg(func=mdp.variable_posture, weight=1.0, params={...})

and it still works at runtime, even if ``variable_posture`` is a class (not a function).

Manager cheat sheet (high-signal notes)
---------------------------------------

ActionManager: slice the action vector and dispatch to each action term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/action_manager.py``

.. code-block:: python

   # file: src/mjlab/managers/action_manager.py
   def process_action(self, action: torch.Tensor) -> None:
       self._prev_prev_action[:] = self._prev_action
       self._prev_action[:] = self._action
       self._action[:] = action.to(self.device)
       idx = 0
       for term in self._terms.values():
           term_actions = action[:, idx : idx + term.action_dim]
           term.process_actions(term_actions)
           idx += term.action_dim

Key points:

- Action history (prev/prev_prev) is used by many rewards (action_rate/action_acc).
- Custom actions: implement ``ActionTermCfg.build(env)`` returning an ``ActionTerm`` subclass.

ObservationManager: the obs “pipeline” lives here (noise/clip/scale/delay/history)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/observation_manager.py``

.. code-block:: python

   # file: src/mjlab/managers/observation_manager.py
   def compute_group(self, group_name: str, update_history: bool = False):
       for term_name, term_cfg in obs_terms:
           obs = term_cfg.func(self._env, **term_cfg.params).clone()
           # noise -> clip -> scale
           # delay buffer
           # history buffer (flattenable)
       return torch.cat(...) or dict(...)

When writing obs terms:

- Return shape should be ``(num_envs, ...)`` (on the correct device).
- Avoid Python loops inside terms (they will slow down training at 4096 envs).

RewardManager: all rewards are weighted and multiplied by dt (NaN-safe)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/reward_manager.py``

.. code-block:: python

   # file: src/mjlab/managers/reward_manager.py
   value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
   value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

TerminationManager: terminated vs truncated (time_out)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/termination_manager.py``

``TerminationTermCfg(time_out=True)`` goes to the truncated branch; everything else goes to terminated.

CommandManager: command generators (resample + metrics + debug vis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/command_manager.py``

A command term inherits from ``CommandTerm`` and implements:

- ``_resample_command(env_ids)``
- ``_update_command()``
- ``_update_metrics()``

Two good templates:

- velocity: ``src/mjlab/tasks/velocity/mdp/velocity_command.py::UniformVelocityCommand``
- tracking: ``src/mjlab/tasks/tracking/mdp/commands.py`` (motion commands)

EventManager: startup/reset/interval + domain randomization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/event_manager.py``

When you write in cfg:

.. code-block:: python

   EventTermCfg(
     mode="startup",
     func=mdp.randomize_field,
     domain_randomization=True,
     params={"field": "geom_friction", ...},
   )

``EventManager`` collects ``field`` into ``domain_randomization_fields``, and the env calls
``Simulation.expand_model_fields`` to expand the field along env dimension (per-env randomization).

CurriculumManager: update “training schedule” before/during reset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/managers/curriculum_manager.py``; velocity curricula: ``src/mjlab/tasks/velocity/mdp/curriculums.py``

Typical uses:

- Expand command ranges over training steps
- Adjust terrain levels based on policy performance

Minimal templates for a new term
--------------------------------

**Function term (reward/obs/termination)**:

.. code-block:: python

   # file: src/mjlab/tasks/<your_task>/mdp/rewards.py
   def my_reward(env, foo: float, entity_cfg=SceneEntityCfg("robot")):
       entity = env.scene[entity_cfg.name]
       return ...

**Class term (needs caching/state)**:

.. code-block:: python

   class MyStatefulReward:
       def __init__(self, cfg, env):
           # Resolve patterns, allocate buffers, etc.
           self.buf = torch.zeros(env.num_envs, device=env.device)

       def reset(self, env_ids=None):
           if env_ids is None:
               env_ids = slice(None)
           self.buf[env_ids] = 0.0

       def __call__(self, env, **params):
           # Still called as term(env, **params)
           return ...

.. note::

   A manager will call ``reset`` on a term object during episode reset only if the term implements a ``reset`` method (see each manager’s ``_class_term_cfgs`` logic).
