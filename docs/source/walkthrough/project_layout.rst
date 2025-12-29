.. _walkthrough-project-layout:

代码地图：你应该去哪里改东西？
============================

这一章是“目录级导航”。当你知道要改什么（reward/obs/sim/robot/训练脚本）时，直接跳到对应目录即可。

核心目录一览
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 目录
     - 你会在这里做什么
   * - ``src/mjlab/envs/``
     - manager-based 环境本体（ ``ManagerBasedRlEnv`` ），以及通用 ``mdp`` 组件（actions/obs/rewards/terminations/events）。
   * - ``src/mjlab/managers/``
     - 核心抽象：``ManagerBase``、各类 manager（action/obs/reward/termination/command/curriculum/event）与 term cfg。
   * - ``src/mjlab/scene/``
     - ``SceneCfg`` / ``Scene``：把 terrain/entities/sensors 拼成 MuJoCo ``MjSpec``，并管理并行环境的原点偏移。
   * - ``src/mjlab/sim/``
     - ``Simulation``：MuJoCo + MuJoCo-Warp GPU 后端、CUDA graph、model field 扩展、NaN guard。
   * - ``src/mjlab/entity/``
     - 机器人/物体等实体的数据接口（root/joint/body/site），是大部分任务 term 的“数据入口”。
   * - ``src/mjlab/sensor/``
     - 传感器（内建 sensor、contact sensor 等）。任务里的很多观测/奖励会通过 ``env.scene[sensor_name]`` 取值。
   * - ``src/mjlab/asset_zoo/``
     - 资产库（MJCF/XML）。G1 在 ``asset_zoo/robots/unitree_g1``，H1 在 ``asset_zoo/robots/unitree_h1``。
   * - ``src/mjlab/tasks/``
     - 任务包。每个任务通常包含：
       ``<task>_env_cfg.py``（base cfg）、``config/<robot>/``（robot override + 注册）、``mdp/``（任务专用 terms）、``rl/``（runner/export）。
   * - ``src/mjlab/rl/``
     - 与 RSL-RL 的接口：VecEnv wrapper、policy/algorithm cfg dataclass。
   * - ``src/mjlab/scripts/``
     - CLI 入口：``train.py``、``play.py``、demo、工具脚本等。
   * - ``docs/``
     - Sphinx 文档（你正在读的这份 walkthrough 也在这里）。

任务的“标准骨架”长什么样？
------------------------

以 velocity 为例（人形/四足都适用）：

- base task cfg：``src/mjlab/tasks/velocity/velocity_env_cfg.py``
- robot override（例如 g1/go1）：``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``
- 注册 task id：``src/mjlab/tasks/velocity/config/g1/__init__.py``
- MDP terms：``src/mjlab/tasks/velocity/mdp/*``（+ 复用 ``src/mjlab/envs/mdp/*``）

同样，tracking：

- base task cfg：``src/mjlab/tasks/tracking/tracking_env_cfg.py``
- g1 override + 注册：``src/mjlab/tasks/tracking/config/g1/*``
- MDP terms：``src/mjlab/tasks/tracking/mdp/*``


