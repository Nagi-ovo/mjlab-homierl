.. _walkthrough-overview:

架构总览：外部包如何挂到 mjlab 上
============================================================

这一页最重要的一句话
--------------------------------

**``mjlab-homierl`` 现在不再是框架本体，而是一个挂在上游 ``mjlab`` 上的外部任务包。**

- 上游 ``mjlab`` 负责 ``ManagerBasedRlEnv``、manager、scene、simulation、
  CLI 入口和 viewer。
- 这个仓库负责 HOMIE 的 env cfg、H1 / 2F85 资产、HIMPPO runner，以及包内
  的测试和文档。

这也是为什么很多旧路径 ``src/mjlab/...`` 在当前仓库里已经不存在了。

从 task id 到训练循环：端到端链路
------------------------------------------------------------

1. **Task 注册**

   - 当前仓库路径：``src/mjlab_homierl/__init__.py``
   - 使用的 API：``mjlab.tasks.registry.register_mjlab_task``
   - 作用：通过 ``mjlab.tasks`` entry point 注册
     ``Mjlab-Homie-Unitree-H1`` 和 ``Mjlab-Homie-Unitree-H1-with_hands``

2. **train / play 入口**

   - CLI 仍然来自上游 ``mjlab``：``uv run train ...``、``uv run play ...``
   - CLI 会读取注册好的 env cfg / rl cfg，构建 ``ManagerBasedRlEnv``，再实例化
     对应 runner

3. **本包内的任务组装**

   - base HOMIE cfg：``src/mjlab_homierl/homie_env_cfg.py``
   - H1 与 with_hands override：``src/mjlab_homierl/env_cfgs.py``
   - 任务专用 MDP term：``src/mjlab_homierl/mdp/*``
   - 自定义 runner：``src/mjlab_homierl/rl/runner.py``

4. **运行时分工**

   - 训练时使用带 actor/critic 观测的 HIMPPO
   - play 时如果 env 去掉了 critic 组，就走 HOMIE 自己的 actor-only
     inference path

一眼看懂的控制流
------------------------------------------------------------

.. code-block:: text

   uv run train/play -> 上游 mjlab CLI
     |
     v
   task registry entry -> env cfg + rl cfg + runner class
     |
     v
   ManagerBasedRlEnv(cfg=...) + RslRlVecEnvWrapper
     |
     v
   HomieHimOnPolicyRunner
     |
     v
   policy(obs) -> action
     |
     v
   上游 env step:
     ActionManager -> Simulation -> Managers -> observations / rewards / resets
     |
     v
   train loop 或 viewer loop

建议从哪些文件开始读
--------------------

- **包入口与 task 注册**：``src/mjlab_homierl/__init__.py``
- **base HOMIE task cfg**：``src/mjlab_homierl/homie_env_cfg.py``
- **H1 task override 与 play 行为**：``src/mjlab_homierl/env_cfgs.py``
- **资产与手部挂载逻辑**：``src/mjlab_homierl/robots/unitree_h1/h1_constants.py``
- **如果你需要看框架内部**：去安装后的上游 ``mjlab`` 包里看
  ``mjlab.envs``、``mjlab.managers``、``mjlab.scene``、``mjlab.sim``
