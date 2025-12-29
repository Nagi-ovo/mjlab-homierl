.. _walkthrough-overview:

架构总览：一张图看懂 mjlab
=========================

.. figure:: ../_static/content/mjlab_architecture_overview.svg
   :width: 100%
   :alt: mjlab architecture overview

这张图对应的“最重要一句话”
--------------------------

**mjlab 的核心是一个 manager-based RL 环境：**

- ``Scene``：把 **MuJoCo 资产/传感器/地形** 组织成可批量并行（num_envs）的场景，并负责把 entity 状态写入仿真。
- ``Simulation``：用 MuJoCo + MuJoCo-Warp 在 GPU 上跑 physics（step/forward/reset），并提供 CUDA graph / NaN guard 等底层能力。
- ``Managers``：把一个任务拆成 **actions / observations / rewards / terminations / commands / events / curriculum** 这些可组合模块。

你可以把它理解成：**环境本体很薄，绝大多数“任务逻辑”都在 managers 的 terms 里。**

从 task id 到训练循环：端到端链路
--------------------------------

1. **Task 注册** （把 env_cfg / rl_cfg 放进 registry）

   - 路径：``src/mjlab/tasks/<task>/config/<robot>/__init__.py``
   - API：``src/mjlab/tasks/registry.py::register_mjlab_task``

2. **训练入口** （从 registry 取 cfg，构建 env + runner）

   - 路径：``src/mjlab/scripts/train.py``
   - 关键调用：``load_env_cfg()`` → ``ManagerBasedRlEnv(cfg=..., device=...)`` → ``RslRlVecEnvWrapper`` → ``rsl_rl.OnPolicyRunner``

3. **环境 step** （action → physics → terminations/rewards → events/commands → observations）

   - 路径：``src/mjlab/envs/manager_based_rl_env.py::ManagerBasedRlEnv.step``

4. **任务逻辑的落点** （mdp 组件）

   - 路径：``src/mjlab/envs/mdp/*`` （通用） + ``src/mjlab/tasks/<task>/mdp/*`` （任务专用）
   - 在 cfg 里以 ``RewardTermCfg(func=..., params=...)`` / ``ObservationTermCfg(func=..., ...)`` 形式被 managers 调用

一眼看懂的“数据流/控制流”
------------------------

.. code-block:: text

   policy(obs) -> action
     |
     v
   ActionManager.process_action / apply_action
     |
     v
   for decimation steps:
     Scene.write_data_to_sim -> Simulation.step -> Scene.update
     |
     v
   TerminationManager.compute  -> reset mask
   RewardManager.compute(dt)   -> reward
   (if any reset) -> _reset_idx -> Event(reset) -> Managers.reset(...)
     |
     v
   CommandManager.compute(dt)
   EventManager.apply(interval, dt)
   ObservationManager.compute(update_history=True)
     |
     v
   return obs, reward, terminated, truncated, extras

建议从哪些文件开始读
--------------------

- **环境生命周期 + manager 加载顺序**：``src/mjlab/envs/manager_based_rl_env.py``
- **manager/term 的基类与“函数 vs 类 term”机制**：``src/mjlab/managers/manager_base.py``、``src/mjlab/managers/manager_term_config.py``
- **SceneEntityCfg（延迟绑定：名字 → ids）**：``src/mjlab/managers/scene_entity_config.py``
- **两个 g1 任务**：

  - velocity：``src/mjlab/tasks/velocity/velocity_env_cfg.py`` + ``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``
  - tracking：``src/mjlab/tasks/tracking/tracking_env_cfg.py`` + ``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``


