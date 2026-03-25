.. _walkthrough-project-layout:

代码地图：你应该去哪里改东西？
============================================================

这一章是重构后外部包结构的“目录级导航”。关键点是：这个仓库现在只保留
HOMIE 相关扩展，通用框架目录已经回到上游 ``mjlab`` 包里。

核心目录一览
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 目录
     - 你会在这里做什么
   * - ``src/mjlab_homierl/__init__.py``
     - entry-point task 注册入口。两个 HOMIE task id 都在这里把 env cfg、
       rl cfg 和 runner 绑起来。
   * - ``src/mjlab_homierl/homie_env_cfg.py``
     - base HOMIE manager 配置：commands、observations、rewards、
       terminations、events、curriculum。
   * - ``src/mjlab_homierl/env_cfgs.py``
     - H1 专用 override：下肢动作拆分、上身扰动动作、接触传感器、play
       override，以及 ``with_hands`` 变体。
   * - ``src/mjlab_homierl/mdp/``
     - HOMIE 任务自己的 commands、observations、rewards、terminations、
       curriculum 辅助函数。
   * - ``src/mjlab_homierl/rl_cfg.py``
     - 与新版 ``mjlab`` 对齐的 actor / critic / algorithm dataclass。
   * - ``src/mjlab_homierl/rl/``
     - 包内 HIMPPO 实现、自定义 runner、ONNX 导出。
   * - ``src/mjlab_homierl/robots/``
     - H1 XML、Robotiq 资产以及 spec 组装逻辑。
   * - ``src/mjlab/scripts/``
     - 已不在本仓库内；``train`` / ``play`` 来自安装后的上游 ``mjlab``。
   * - ``tests/``
     - 只保留这个包自己的回归测试：task 注册、env cfg、rl cfg 和 H1
       资产组装。
   * - ``docs/``
     - Sphinx 文档（你正在读的这份 walkthrough 也在这里）。

哪些东西现在属于上游？
------------------------------------------------------------

如果你要看框架内部，请去上游 ``mjlab`` 包里找：

- ``mjlab.envs``：``ManagerBasedRlEnv`` 与通用 env 逻辑
- ``mjlab.managers``：各类 manager 与 term cfg 机制
- ``mjlab.scene`` / ``mjlab.sim``：scene 组装与 MuJoCo 运行时
- ``mjlab.rl``：vec-env wrapper 与基础 runner 接口
