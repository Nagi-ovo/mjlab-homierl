.. _walkthrough-managers-and-terms:

重点类二：Managers + Terms（isaaclab-like 的核心抽象）
====================================================

如果你只想记住一个“可迁移”的思想，那就是：

**任务 = 一组 manager dictionaries；每个 manager = 多个 terms；每个 term = (func/class, params) 的可调用对象。**

本章会把这套机制拆到“你可以自己实现一个新 term / 新 manager term cfg”的程度。

三层抽象：TermCfg → Term → Manager
---------------------------------

1) TermCfg：用 dataclass 描述“怎么构建/怎么调用”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/manager_term_config.py``

- ``RewardTermCfg`` / ``TerminationTermCfg`` / ``CurriculumTermCfg`` / ``EventTermCfg``：
  继承 ``ManagerTermBaseCfg``，核心字段是 ``func`` 和 ``params`` （以及 reward 的 ``weight`` 等）。
- ``ObservationTermCfg``：额外定义 noise/clip/scale/delay/history 的处理管线。
- ``ActionTermCfg``：是抽象类，要求实现 ``build(env) -> ActionTerm``。
- ``CommandTermCfg``：持有 ``class_type``，由 ``CommandManager`` 直接实例化 term。

.. code-block:: python

   # file: src/mjlab/managers/manager_term_config.py
   @dataclass
   class ManagerTermBaseCfg:
       func: Any
       params: dict[str, Any] = field(default_factory=lambda: {})

   @dataclass(kw_only=True)
   class RewardTermCfg(ManagerTermBaseCfg):
       func: Any
       weight: float

   @dataclass
   class ObservationTermCfg(ManagerTermBaseCfg):
       # Processing pipeline: compute → noise → clip → scale → delay → history.
       noise: NoiseCfg | NoiseModelCfg | None = None
       clip: tuple[float, float] | None = None
       scale: tuple[float, ...] | float | torch.Tensor | None = None
       delay_min_lag: int = 0
       delay_max_lag: int = 0
       history_length: int = 0

2) Term：可以是函数，也可以是“有状态的类”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

manager 调用 term 的方式基本统一：``term(env, **params)``。

**函数 term**：适合纯计算、无内部状态的项（例如从 entity/sensor 读数、简单 reward）。

**类 term**：适合需要缓存/跨步统计/预处理的项（例如把 pattern->tensor 映射提前解析、维护 peak heights、维护 EMA 等）。

> 你在 velocity 的 reward 里可以看到典型例子：

> ``feet_swing_height``、``variable_posture`` 都是类 term（路径：``src/mjlab/tasks/velocity/mdp/rewards.py``）。

3) Manager：负责装配 terms + 管生命周期
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个 manager 都做两件事：

- 在 ``_prepare_terms`` 中把 cfg dict/list 装配成可调用 terms（并做必要的 resolution / buffer 构建）
- 在 ``compute/reset`` 中调度这些 terms

核心机制：SceneEntityCfg 的“延迟绑定” + 类 term 自动实例化
------------------------------------------------------------

SceneEntityCfg：为什么需要它？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当你在 cfg 里写：

.. code-block:: python

   SceneEntityCfg("robot", joint_names=(".*",))

你其实是在说：“我想用 scene 里名为 robot 的 entity 的 **所有关节**。”

但在 **cfg 构建阶段** 还没有 scene/model，因此没法把名字解析成具体 indices。
所以 mjlab 采用 **延迟绑定**：把解析推迟到 env 初始化、scene 可用之后。

路径：``src/mjlab/managers/scene_entity_config.py``

.. code-block:: python

   # file: src/mjlab/managers/scene_entity_config.py
   @dataclass
   class SceneEntityCfg:
       name: str
       joint_names: str | tuple[str, ...] | None = None
       joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))
       # ... body/geom/site/actuator 同理 ...

       def resolve(self, scene: Scene) -> None:
           entity = scene[self.name]
           # 对每类字段：names <-> ids 一致化，并在“全选”时优化成 slice(None)
           ...

ManagerBase._resolve_common_term_cfg：两件关键事
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/manager_base.py``

.. code-block:: python

   # file: src/mjlab/managers/manager_base.py
   def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg):
       # 1) 自动 resolve params 里出现的 SceneEntityCfg
       for value in term_cfg.params.values():
           if isinstance(value, SceneEntityCfg):
               value.resolve(self._env.scene)

       # 2) 如果 func 是 class，则自动实例化成“可调用对象”（cfg + env 注入）
       if inspect.isclass(term_cfg.func):
           term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

这就是为什么你可以在 cfg 里直接写：

.. code-block:: python

   RewardTermCfg(func=mdp.variable_posture, weight=1.0, params={...})

即使 ``variable_posture`` 是一个类（而不是函数），也能在 runtime 正常工作。

各个 Manager 的“要点速记”
-------------------------

ActionManager：把 action 向量切片分发给每个 action term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/action_manager.py``

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

关键点：

- action history（prev/prev_prev）是很多 reward（action_rate/action_acc）的输入。
- 自定义 action：实现 ``ActionTermCfg.build(env)`` 返回一个 ``ActionTerm`` 子类即可。

ObservationManager：obs 的“流水线”在这里（noise/clip/scale/delay/history）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/observation_manager.py``

.. code-block:: python

   # file: src/mjlab/managers/observation_manager.py
   def compute_group(self, group_name: str, update_history: bool = False):
       for term_name, term_cfg in obs_terms:
           obs = term_cfg.func(self._env, **term_cfg.params).clone()
           # noise -> clip -> scale
           # delay buffer
           # history buffer（可展平）
       return torch.cat(...) or dict(...)

你在写 obs term 时要保证：

- 返回 shape 是 ``(num_envs, ...)`` （且在正确 device 上）。
- 尽量不要在 term 内部做 Python 循环（会拖慢 4096 env 的训练）。

RewardManager：所有 reward 都按 dt 乘权重（并对 NaN 做保护）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/reward_manager.py``

.. code-block:: python

   # file: src/mjlab/managers/reward_manager.py
   value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
   value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

TerminationManager：区分 terminated / truncated（time_out）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/termination_manager.py``

``TerminationTermCfg(time_out=True)`` 会进入 truncated 分支；其余进入 terminated。

CommandManager：命令生成器（resample + metrics + debug vis）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/command_manager.py``

命令 term 继承 ``CommandTerm``，实现：

- ``_resample_command(env_ids)``
- ``_update_command()``
- ``_update_metrics()``

velocity 的 ``UniformVelocityCommand`` 与 tracking 的 ``MotionCommand`` 都是很好的模板：

- velocity：``src/mjlab/tasks/velocity/mdp/velocity_command.py``
- tracking：``src/mjlab/tasks/tracking/mdp/commands.py``

EventManager：startup/reset/interval 三种时机 + domain randomization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/event_manager.py``

当你在 cfg 里写：

.. code-block:: python

   EventTermCfg(
     mode="startup",
     func=mdp.randomize_field,
     domain_randomization=True,
     params={"field": "geom_friction", ...},
   )

``EventManager`` 会把 ``field`` 收集进 ``domain_randomization_fields``，随后 env 会调用
``Simulation.expand_model_fields`` 让该字段按 env 维度展开（实现每个环境独立随机化）。

CurriculumManager：在 reset 前/时更新“训练日程”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/managers/curriculum_manager.py``、velocity 的 curriculum：``src/mjlab/tasks/velocity/mdp/curriculums.py``

典型用法：

- 随训练步数扩大 command ranges
- 按策略表现调整 terrain level

写一个新 term 的“最小模板”
-------------------------

**函数 term（reward/obs/termination）**：

.. code-block:: python

   # file: src/mjlab/tasks/<your_task>/mdp/rewards.py
   def my_reward(env, foo: float, entity_cfg=SceneEntityCfg("robot")):
       entity = env.scene[entity_cfg.name]
       return ...

**类 term（需要缓存/状态）**：

.. code-block:: python

   class MyStatefulReward:
       def __init__(self, cfg, env):
           # 这里可以把 pattern 解析成 tensor、分配 buffers 等
           self.buf = torch.zeros(env.num_envs, device=env.device)

       def reset(self, env_ids=None):
           if env_ids is None:
               env_ids = slice(None)
           self.buf[env_ids] = 0.0

       def __call__(self, env, **params):
           # 依然是 term(env, **params) 的调用方式
           return ...

.. note::

   只有当 term 对象提供 ``reset`` 方法时，manager 才会在 episode reset 时调用它（见各 manager 的 ``_class_term_cfgs`` 逻辑）。


