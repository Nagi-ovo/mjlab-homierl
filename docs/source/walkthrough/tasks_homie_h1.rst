.. _walkthrough-task-homie-h1:

实例三：Homie —— 混合运动与干扰（Unitree H1）
==========================================

``Mjlab-Homie-Unitree-H1`` 是一个综合性更强的任务，它结合了 **速度追踪 (Velocity)** 、 **蹲起 (Squat)** 以及 **上身随机干扰** 。

这个任务的核心设计思想是： **缩小策略的动作空间，将其集中在下肢控制上，而将上身作为“随时间变化的平滑扰动”。** 这能让策略在面对复杂的身体姿态变化时，依然保持下肢行走的鲁棒性。

任务骨架：make_homie_env_cfg
---------------------------

路径：``src/mjlab/tasks/homie/homie_env_cfg.py``

与之前的任务不同，Homie 默认支持两个命令生成器：

1. **twist** (``UniformVelocityCommand``)：控制前后、左右平移速度及转向速度。
2. **height** (``RelativeHeightCommand``)：控制骨盆相对于脚部的相对高度（实现蹲起动作）。

.. code-block:: python

   # file: src/mjlab/tasks/homie/homie_env_cfg.py
   commands = {
       "twist": UniformVelocityCommandCfg(...),
       "height": RelativeHeightCommandCfg(
           entity_name="robot",
           foot_site_names=(), # 由 robot override 填充
           ranges=RelativeHeightCommandCfg.Ranges(height=(0.6, 1.0)),
       ),
   }

核心特性：UpperBodyPoseAction (Policy-Free)
------------------------------------------

路径：``src/mjlab/tasks/homie/config/h1/env_cfgs.py``

这是 Homie 任务最独特的抽象。在 ``actions`` 字典中，除了策略控制的 ``joint_pos``，还有一个 ``upper_body_pose``：

*   **零动作维度** ：它在策略侧的维度为 0，不增加神经网络的输出负担。
*   **平滑差值** ：它内部维护一个目标姿态，每步通过 ``torch.lerp`` 向其靠近。
*   **定时重采样** ：通过 ``EventTermCfg`` 定期采样新的上身随机姿态。

.. code-block:: python

   # 上身动作配置（Policy-Free）
   cfg.actions["upper_body_pose"] = UpperBodyPoseActionCfg(
       entity_name="robot",
       joint_names=upper_body_joint_expr,
       interp_rate=0.05,
       target_range=(-0.6, 0.6),
       initial_ratio=0.0, # 初始无动作，随课程增加
   )

课程学习：让干扰逐渐增强
----------------------

路径：``src/mjlab/tasks/homie/mdp/curriculums.py``

为了防止训练初期扰动太大导致无法收敛，Homie 引入了 ``upper_body_action_curriculum``：

*   **表现挂钩** ：只有当 ``track_linear_velocity`` （速度追踪奖励）达到阈值（如 0.8）时，才增加上身动作的幅度。
*   **线性增长** ：幅度从 0 逐渐增加到 1.0。

.. code-block:: python

   # 课程配置
   cfg.curriculum["upper_body_action"] = CurriculumTermCfg(
       func=mdp.upper_body_action_curriculum,
       params={
           "action_name": "upper_body_pose",
           "reward_name": "track_linear_velocity",
           "success_threshold": 0.8,
           "increment": 0.05,
       },
   )

Rewards & Terminations：混合任务的平衡艺术
----------------------------------------

Homie 任务需要在“行走”和“蹲起”两个目标之间寻找平衡，同时忽略上身干扰。

1) 奖励设计：多任务解耦
^^^^^^^^^^^^^^^^^^^^^^

*   **按环境分组 (Env Grouping)** ：

    *   并不是所有环境都在做同一个任务。 ``track_linear_velocity`` 奖励只对 ``velocity`` 组的环境起强作用。
    *   ``track_height`` (``track_relative_height``) 奖励则只对 ``squat`` 和 ``standing`` 组的环境有效。
*   **对抗干扰的正则项** ：

    *   ``knee_deviation_reward`` : 在蹲起过程中，惩罚膝盖相对于脚尖的水平偏移，引导出合理的蹲姿。
    *   ``upright`` : 强力维持躯干直立，这是抵消上身随机摆动干扰的关键。

2) 终止条件：松耦合
^^^^^^^^^^^^^^^^^^

*   **放宽姿态限制** ：

    *   由于 H1 具有极大的蹲起幅度和上身扰动， ``fell_over`` 的角度阈值可能比 G1 默认的更宽松。
*   **自碰撞处理** ：

    *   对于大范围运动，Homie 会添加特殊的 ``self_collision_cost`` 奖励项而不是直接终止，给予策略"试错"的空间。

H1 机器人覆盖与 H1 Constants
---------------------------

路径：``src/mjlab/asset_zoo/robots/unitree_h1/h1_constants.py``

由于 H1 的电机规格与 G1 不同，Homie 任务深度使用了 ``h1_constants.py`` 中的参数：

*   **多组 Actuator** ：H1 分成了 ``HIP_KNEE``, ``ANKLE_TORSO``, ``ARM`` 三类驱动器组，每组有不同的 ``stiffness`` 和 ``damping``。
*   **自动 Action Scale** ：基于电机的 ``effort_limit / stiffness`` 自动计算每个关节的动作缩放比例。

.. code-block:: python

   # 自动计算 scale
   for a in H1_ARTICULATION.actuators:
       names = a.target_names_expr
       for n in names:
           H1_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness

总结：为什么参考 Homie？
----------------------

如果你想开发以下功能的任务，Homie 是最好的参考：

1.  **多任务混合** ：同时处理速度追踪与高度控制。
2.  **身体部分控制** ：策略只控制全身的一部分关节，另一部分关节按预设脚本或随机游走。
3.  **高级课程学习** ：不仅仅改环境参数（如摩擦力），还动态修改动作项的行为。

