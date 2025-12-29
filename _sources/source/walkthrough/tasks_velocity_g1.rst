.. _walkthrough-task-velocity-g1:

实例一：Velocity Tracking（Unitree G1）
=====================================

这一章用 ``Mjlab-Velocity-Flat-Unitree-G1`` / ``Mjlab-Velocity-Rough-Unitree-G1`` 作为“模板任务”，讲清楚：

- 任务 cfg 如何拆成 MDP managers
- g1 如何在 base cfg 上做覆盖（实体、传感器、entity 名称、权重、play mode）
- 你要写一个新的 locomotion 任务，最少改哪几处

任务骨架：make_velocity_env_cfg（base cfg）
-----------------------------------------

路径：``src/mjlab/tasks/velocity/velocity_env_cfg.py``

base cfg 做两件事：

1. 定义 velocity tracking 任务的 **MDP 结构** （obs/actions/commands/events/rewards/terminations/curriculum）
2. 留出“每个机器人都不同”的占位符（例如 foot site names、geom friction 的 geoms、viewer body name、部分 reward 权重）

例如 policy observations（注意：命令 command 也是 observation 的一部分）：

.. code-block:: python

   # file: src/mjlab/tasks/velocity/velocity_env_cfg.py
   policy_terms = {
     "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
     "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
     "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
     "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
     "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
     "actions": ObservationTermCfg(func=mdp.last_action),
     "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
   }

速度命令（CommandManager）默认用 ``UniformVelocityCommand``：

.. code-block:: python

   # file: src/mjlab/tasks/velocity/velocity_env_cfg.py
   commands = {
     "twist": UniformVelocityCommandCfg(
       entity_name="robot",
       resampling_time_range=(3.0, 8.0),
       heading_command=True,
       ranges=UniformVelocityCommandCfg.Ranges(
         lin_vel_x=(-1.0, 1.0),
         lin_vel_y=(-1.0, 1.0),
         ang_vel_z=(-0.5, 0.5),
         heading=(-pi, pi),
       ),
     )
   }

理解 velocity 任务最重要的 3 个 term
------------------------------------

1) Command term：UniformVelocityCommand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/tasks/velocity/mdp/velocity_command.py``

它提供三个关键特性：

- **resample**：每个 env 独立按时间窗重采样速度命令
- **heading 控制** ：一部分 env 用 heading error 转成 yaw rate（更像"面朝某方向走"）
- **standing env**：按比例采样“站立不动”的环境（让策略学会停止）

2) Rewards & Terminations：移动任务的“胡萝卜与大棒”
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 Velocity 任务中，奖励函数的设计遵循“生存 -> 任务完成 -> 运动质量”的层级。

*   **核心奖励（任务完成）** ：

    *   ``track_linear_velocity`` : 鼓励机器人的实际速度匹配 ``twist`` 命令。
    *   ``track_angular_velocity`` : 鼓励角速度匹配。
*   **正则化奖励（运动质量）** ：

    *   ``joint_pos_limits`` / ``joint_vel_limits`` : 惩罚关节接近死区或速度过快。
    *   ``action_rate_l2`` : 惩罚动作突变，产生平滑的关节运动。
    *   ``feet_air_time`` / ``feet_clearance`` : 引导产生自然的步态（如抬腿高度）。
*   **终止条件（安全边界）** ：

    *   ``fell_over`` (``bad_orientation``): 只要躯干倾斜超过 70 度，立即重置，防止学到“贴地爬行”。
    *   ``time_out``: 强制在固定步数后重置（truncated），确保持续探索。

3) Curriculum：terrain_levels_vel + commands_vel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/tasks/velocity/mdp/curriculums.py``

- ``terrain_levels_vel``：用“走到多远”决定升/降地形难度
- ``commands_vel``：按训练步数 stage 修改命令的 ranges（更大速度、更大 yaw rate）

g1 覆盖：unitree_g1_*_env_cfg（robot-specific）
----------------------------------------------

路径：``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``

这一层的作用是把 base cfg 变成“真的能训练 g1 的 cfg”，典型改动包括：

- 配置实体：``cfg.scene.entities = {"robot": get_g1_robot_cfg()}``
- 增加 contact sensors（foot-ground / self-collision）
- 补齐 base cfg 里留的占位符（site_names / geom_names / torso_link 等）
- 调整 reward 权重（让 g1 训练更稳定/更自然）

例如（contact sensors + action scale + play mode）：

.. code-block:: python

   # file: src/mjlab/tasks/velocity/config/g1/env_cfgs.py
   cfg.scene.entities = {"robot": get_g1_robot_cfg()}
   cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

   joint_pos_action = cfg.actions["joint_pos"]
   joint_pos_action.scale = G1_ACTION_SCALE

   if play:
       cfg.episode_length_s = int(1e9)
       cfg.observations["policy"].enable_corruption = False
       cfg.events.pop("push_robot", None)
       cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})

Task 注册：register_mjlab_task
-----------------------------

路径：``src/mjlab/tasks/velocity/config/g1/__init__.py``

.. code-block:: python

   # file: src/mjlab/tasks/velocity/config/g1/__init__.py
   register_mjlab_task(
     task_id="Mjlab-Velocity-Rough-Unitree-G1",
     env_cfg=unitree_g1_rough_env_cfg(),
     play_env_cfg=unitree_g1_rough_env_cfg(play=True),
     rl_cfg=unitree_g1_ppo_runner_cfg(),
     runner_cls=VelocityOnPolicyRunner,
   )

从这个点你就可以推出整个 CLI 链路：

- ``uv run train <task_id>`` → ``src/mjlab/scripts/train.py``  
- ``import mjlab.tasks`` 会触发动态 import（见 ``src/mjlab/tasks/__init__.py``）→ registry 被填充  
- ``load_env_cfg(task_id)`` 返回 env_cfg 的 deep copy（避免污染全局注册表）

你要做一个新 locomotion 任务：最少改哪几处？
--------------------------------------------

建议按“侵入性从小到大”的顺序改：

1. **改 reward 权重 / 增减 reward term** ：改 ``config/g1/env_cfgs.py`` 里对 ``cfg.rewards`` 的修改。
2. **改 command 分布** ：改 ``cfg.commands["twist"].ranges`` （也可以加 curriculum stage）。
3. **改 observation** ：在 ``velocity_env_cfg.py`` 的 policy/critic group 里加 term，或在 g1 override 里删/换 term。
4. **加新 event randomization** ：在 ``events`` 增加 ``EventTermCfg``（startup/reset/interval），必要时打开 ``domain_randomization=True``。

.. note::

   如果你要新增依赖 contact sensor 的 reward/obs，请先在 robot cfg 里把 sensor 加进 ``cfg.scene.sensors``，
   然后再在 term 里通过 ``env.scene[sensor_name]`` 读取（见 feet_* 与 self_collision_cost 的写法）。


