.. _developer-walkthrough-zh:

Code Walkthrough（面向科研开发，上手 + 深入）
==========================================

这套文档的目标是：让第一次阅读本项目的同学在 **1-2 小时** 内建立可靠的心智模型，并能在现有 Unitree **G1** 任务（velocity / tracking）的基础上，快速开发自己的 RL 任务。

你会在这里学到什么
------------------

- **架构总览**：`ManagerBasedRlEnv = Scene + Simulation + Managers` 的数据流与控制流。
- **Manager-base API 设计**：为什么用“manager + term + cfg”的结构，它如何做到可组合、可扩展、可复用。
- **任务开发路径**：如何像写 Isaac Lab manager-based task 一样写 mjlab task（但用 dict cfg）。
- **g1/h1 任务解剖**：`tasks/velocity`、`tasks/tracking` 与 `tasks/homie` 的 cfg/MDP/训练入口拆解与改造建议。

阅读顺序建议
------------

第一次阅读：按顺序看（总览 → 环境生命周期 → managers/terms → 任务实例 → 自己动手）。

已经熟 Isaac Lab：直接跳到 `managers_and_terms` 与两个任务章节对照代码即可。

.. toctree::
   :maxdepth: 2
   :caption: 章节目录

   overview
   quickstart
   project_layout
   manager_based_env
   managers_and_terms
   tasks_velocity_g1
   tasks_tracking_g1
   tasks_homie_h1
   how_to_add_g1_task
   debugging_perf


