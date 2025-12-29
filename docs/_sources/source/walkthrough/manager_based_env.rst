.. _walkthrough-manager-based-env:

重点类一：ManagerBasedRlEnv（环境生命周期与数据流）
=================================================

这一章解决一个核心问题：**一个动作 action 从哪里来、如何进入 physics、如何产出 obs/reward/done，并在 reset 时怎样“按正确顺序”重置各模块。**

配置入口：ManagerBasedRlEnvCfg
-----------------------------

路径：``src/mjlab/envs/manager_based_rl_env.py``

``ManagerBasedRlEnvCfg`` 不是 Isaac Lab 那种嵌套 ``@configclass``，而是“ **顶层 dataclass + 多个 dict（name -> term cfg）** ”：

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   @dataclass(kw_only=True)
   class ManagerBasedRlEnvCfg:
       decimation: int                      # 每个 env step 对应多少个 physics step
       scene: SceneCfg                      # 场景（terrain + entities + sensors）
       observations: dict[str, ObservationGroupCfg]
       actions: dict[str, ActionTermCfg]
       events: dict[str, EventTermCfg] = {...}     # 默认包含 reset_scene_to_default
       rewards: dict[str, RewardTermCfg] = {}
       terminations: dict[str, TerminationTermCfg] = {}
       commands: dict[str, CommandTermCfg] | None = None
       curriculum: dict[str, CurriculumTermCfg] | None = None
       sim: SimulationCfg = SimulationCfg()
       viewer: ViewerConfig = ViewerConfig()
       episode_length_s: float = 0.0
       is_finite_horizon: bool = False

你应该记住的 3 个时间尺度：

- **physics_dt**：MuJoCo 的 timestep（``cfg.sim.mujoco.timestep``）
- **step_dt**：环境控制周期（``physics_dt * decimation``）
- **episode_length**：由 ``episode_length_s / step_dt`` 推出来（向上取整）

环境构建：Scene + Simulation + Managers
---------------------------------------

``ManagerBasedRlEnv.__init__`` 的结构非常“可读”，按层次搭积木：

1. ``Scene(cfg.scene, device)``：构建 MuJoCo ``MjSpec``（terrain/entities/sensors），随后 compile 成 ``MjModel``。
2. ``Simulation(num_envs, cfg.sim, model, device)``：把 MuJoCo model/data 放到 MJWarp 里，准备 GPU step/forward/reset。
3. ``scene.initialize(mj_model, model, data)``：把 entity/sensor 绑定到仿真数据。
4. ``load_managers()``：把 cfg 里的 dict term_cfg 装配成各类 managers（ **顺序非常关键** ）。

Manager 加载顺序（为什么重要）
-----------------------------

路径：``src/mjlab/envs/manager_based_rl_env.py``

.. code-block:: python

   # file: src/mjlab/envs/manager_based_rl_env.py
   def load_managers(self) -> None:
       # 1) EventManager 必须最先：它决定 domain randomization 需要扩展哪些 model fields
       self.event_manager = EventManager(self.cfg.events, self)
       self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

       # 2) CommandManager 在 ObservationManager 之前：obs 可能引用 command
       self.command_manager = CommandManager(...) or NullCommandManager()

       # 3) Action + Observation：定义 action/obs 空间与缓存逻辑
       self.action_manager = ActionManager(self.cfg.actions, self)
       self.observation_manager = ObservationManager(self.cfg.observations, self)

       # 4) RL managers：termination/reward/curriculum
       self.termination_manager = TerminationManager(self.cfg.terminations, self)
       self.reward_manager = RewardManager(self.cfg.rewards, self)
       self.curriculum_manager = CurriculumManager(...) or NullCurriculumManager()

       self._configure_gym_env_spaces()
       if "startup" in self.event_manager.available_modes:
           self.event_manager.apply(mode="startup")

这里最容易忽略的一点是：**domain randomization = 改 model 字段**。因此 env 会用
``event_manager.domain_randomization_fields`` 调用 ``sim.expand_model_fields``，让每个 env 都能有自己的参数（例如不同的摩擦系数）。

step()：action → physics → done/reward → reset → obs
---------------------------------------------------

路径：``src/mjlab/envs/manager_based_rl_env.py``

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

       # reset（terminated 或 time_out）
       reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
       if len(reset_env_ids) > 0:
           self._reset_idx(reset_env_ids)
           self.scene.write_data_to_sim()
           self.sim.forward()

       # command + events
       self.command_manager.compute(dt=self.step_dt)
       if "interval" in self.event_manager.available_modes:
           self.event_manager.apply(mode="interval", dt=self.step_dt)

       # obs（注意 update_history=True）
       self.obs_buf = self.observation_manager.compute(update_history=True)
       return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

你可以把 ``step`` 看作一个严格顺序的 pipeline：**动作先进入 physics**，之后才计算 termination/reward，然后可能 reset，最后再算 obs。

reset 的顺序敏感（为什么 extras/log 很重要）
---------------------------------------------

路径：``src/mjlab/envs/manager_based_rl_env.py``

``_reset_idx`` 在 reset 时会按固定顺序 reset managers，并把每个 manager 的统计写进 ``extras["log"]``：

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

这套顺序的直觉是：

- obs/action/reward 这类模块通常维护 buffers 或 episodic accumulators，reset 时要先清。
- command/event 可能在 reset 时 resample，且 event(reset) 可能会改仿真状态。
- termination 最后重置统计，避免把上一回合的终止原因污染到下一回合。

finite horizon vs infinite horizon（terminated vs truncated）
------------------------------------------------------------

``TerminationManager`` 会把不同 termination term 分成两类：

- ``time_out=True`` → **truncated**
- 其他 → **terminated**

路径：``src/mjlab/managers/termination_manager.py``

这对 value bootstrapping 很关键（尤其是 PPO 这类算法）。


