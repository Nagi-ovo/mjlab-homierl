.. _walkthrough-en-manager-based-env:

Key Class 1: ManagerBasedRlEnv (Lifecycle and Data Flow)
=======================================================

This chapter answers one core question:
**where actions come from, how they enter physics, how obs/reward/done are produced, and how resets happen in the correct order.**

Config entrypoint: ManagerBasedRlEnvCfg
--------------------------------------

Path: ``src/mjlab/envs/manager_based_rl_env.py``

``ManagerBasedRlEnvCfg`` is not Isaac Lab’s nested ``@configclass`` style. It is:
**a top-level dataclass + multiple dicts (name → term cfg)**.

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   @dataclass(kw_only=True)
   class ManagerBasedRlEnvCfg:
       decimation: int                      # physics steps per env step
       scene: SceneCfg                      # terrain + entities + sensors
       observations: dict[str, ObservationGroupCfg]
       actions: dict[str, ActionTermCfg]
       events: dict[str, EventTermCfg] = {...}     # includes reset_scene_to_default by default
       rewards: dict[str, RewardTermCfg] = {}
       terminations: dict[str, TerminationTermCfg] = {}
       commands: dict[str, CommandTermCfg] | None = None
       curriculum: dict[str, CurriculumTermCfg] | None = None
       sim: SimulationCfg = SimulationCfg()
       viewer: ViewerConfig = ViewerConfig()
       episode_length_s: float = 0.0
       is_finite_horizon: bool = False

Three time scales to remember
-----------------------------

- **physics_dt**: MuJoCo timestep (``cfg.sim.mujoco.timestep``)
- **step_dt**: env control period (``physics_dt * decimation``)
- **episode_length**: derived from ``episode_length_s / step_dt`` (ceil)

Env construction: Scene + Simulation + Managers
----------------------------------------------

``ManagerBasedRlEnv.__init__`` is intentionally readable, building the system layer-by-layer:

1. ``Scene(cfg.scene, device)``: build MuJoCo ``MjSpec`` (terrain/entities/sensors), then compile to ``MjModel``.
2. ``Simulation(num_envs, cfg.sim, model, device)``: place MuJoCo model/data into MJWarp, ready for GPU step/forward/reset.
3. ``scene.initialize(mj_model, model, data)``: bind entities/sensors to simulation data.
4. ``load_managers()``: assemble cfg dicts into managers (**order matters**).

Manager loading order (why it matters)
--------------------------------------

Path: ``src/mjlab/envs/manager_based_rl_env.py``

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   def load_managers(self) -> None:
       # 1) EventManager must come first: it decides which model fields need per-env randomization
       self.event_manager = EventManager(self.cfg.events, self)
       self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

       # 2) CommandManager before ObservationManager: observations may reference commands
       self.command_manager = CommandManager(...) or NullCommandManager()

       # 3) Action + Observation: define action/obs spaces and buffers
       self.action_manager = ActionManager(self.cfg.actions, self)
       self.observation_manager = ObservationManager(self.cfg.observations, self)

       # 4) RL managers: termination/reward/curriculum
       self.termination_manager = TerminationManager(self.cfg.terminations, self)
       self.reward_manager = RewardManager(self.cfg.rewards, self)
       self.curriculum_manager = CurriculumManager(...) or NullCurriculumManager()

       self._configure_gym_env_spaces()
       if "startup" in self.event_manager.available_modes:
           self.event_manager.apply(mode="startup")

The subtle-but-important point: **domain randomization = modifying MuJoCo model fields**.
So the env calls ``sim.expand_model_fields(...)`` using ``event_manager.domain_randomization_fields`` to give each env its own parameters (e.g., friction).

step(): action → physics → done/reward → reset → obs
----------------------------------------------------

Path: ``src/mjlab/envs/manager_based_rl_env.py``

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   def step(self, action: torch.Tensor):
       self.action_manager.process_action(action.to(self.device))

       for _ in range(self.cfg.decimation):
           self.action_manager.apply_action()
           self.scene.write_data_to_sim()
           self.sim.step()
           self.scene.update(dt=self.physics_dt)

       self.episode_length_buf += 1
       self.common_step_counter += 1

       # done / reward
       self.reset_buf = self.termination_manager.compute()
       self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

       # reset (terminated or time_out)
       reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
       if len(reset_env_ids) > 0:
           self._reset_idx(reset_env_ids)
           self.scene.write_data_to_sim()
           self.sim.forward()

       # command + events
       self.command_manager.compute(dt=self.step_dt)
       if "interval" in self.event_manager.available_modes:
           self.event_manager.apply(mode="interval", dt=self.step_dt)

       # obs (note update_history=True)
       self.obs_buf = self.observation_manager.compute(update_history=True)
       return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

You can think of ``step`` as a strict pipeline: **actions go into physics first**, then termination/reward are computed, optional resets happen, and finally observations are computed.

Reset order is sensitive (why extras/log matter)
-----------------------------------------------

Path: ``src/mjlab/envs/manager_based_rl_env.py``

On reset, ``_reset_idx`` resets managers in a fixed order, and records per-manager stats into ``extras["log"]``:

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   def _reset_idx(self, env_ids):
       self.curriculum_manager.compute(env_ids=env_ids)
       self.sim.reset(env_ids)
       self.scene.reset(env_ids)
       if "reset" in self.event_manager.available_modes:
           self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=...)

       self.extras["log"] = {}
       self.extras["log"].update(self.observation_manager.reset(env_ids))
       self.extras["log"].update(self.action_manager.reset(env_ids))
       self.extras["log"].update(self.reward_manager.reset(env_ids))
       self.extras["log"].update(self.curriculum_manager.reset(env_ids))
       self.extras["log"].update(self.command_manager.reset(env_ids))
       self.extras["log"].update(self.event_manager.reset(env_ids))
       self.extras["log"].update(self.termination_manager.reset(env_ids))

The intuition behind this order:

- obs/action/reward modules maintain buffers or episodic accumulators, and should be cleared early.
- command/event may resample during reset, and event(reset) may directly modify simulator state.
- termination stats are reset last to avoid leaking the previous episode’s termination reason.

Finite horizon vs infinite horizon (terminated vs truncated)
------------------------------------------------------------

``TerminationManager`` splits termination terms into:

- ``time_out=True`` → **truncated**
- everything else → **terminated**

Path: ``src/mjlab/managers/termination_manager.py``

This matters for value bootstrapping (especially for PPO-like algorithms).
