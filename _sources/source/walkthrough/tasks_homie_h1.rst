.. _walkthrough-task-homie-h1:

实例三：Homie —— 混合运动与干扰（Unitree H1）
==========================================

Homie 是一个综合性更强的任务，它结合了 **速度追踪 (Velocity)**、**蹲起 (Squat)** 以及 **上身随机干扰**。

目前提供两个 task id：

- ``Mjlab-Homie-Unitree-H1``：标准版本。
- ``Mjlab-Homie-Unitree-H1-with_hands``：额外挂载 Robotiq 2F85 夹爪（包含 policy-free 的夹爪随机动作）。

这个任务的核心设计思想是：**缩小策略的动作空间，将其集中在下肢控制上，而将上身（以及可选的夹爪）作为“随时间变化的平滑扰动”。** 这能让策略在面对复杂的身体姿态变化时，依然保持下肢行走的鲁棒性。

任务骨架：make_homie_env_cfg（base cfg）
--------------------------------------

路径：``src/mjlab/tasks/homie/homie_env_cfg.py``

与之前的任务不同，Homie 默认支持两个命令生成器，并且都支持 **按 env group 进行 gating**（见下文的 Env Grouping）：

1. **twist** (``UniformVelocityCommand``)：控制前后、左右平移速度及转向速度。
2. **height** (``RelativeHeightCommand``)：控制骨盆相对于脚部的相对高度（实现蹲起动作）。

.. code-block:: python

   # file: src/mjlab/tasks/homie/homie_env_cfg.py
   commands = {
       "twist": UniformVelocityCommandCfg(
           ...,
           active_env_group="velocity",
           rel_standing_envs=1.0 / 6.0,
           avoid_consecutive_standing=True,
       ),
       "height": RelativeHeightCommandCfg(
           entity_name="robot",
           active_env_group="squat",
           # Smooth height-command transitions (avoid step changes at resampling).
           interp_rate=0.02,
           foot_site_names=(), # 由 robot override 填充
           ranges=RelativeHeightCommandCfg.Ranges(height=(0.6, 1.0)),
       ),
   }

Env Grouping：一套 vec env 并行训练三种“子任务”
----------------------------------------------

路径：``src/mjlab/tasks/homie/mdp/curriculums.py::assign_homie_env_groups``

Homie 通过 ``env_group`` 把一批并行环境拆成 3 组（mask），并在 ``commands`` / ``rewards`` / ``curriculum`` 里用这些组名做 gating：

- ``squat``：约 20%（``set_x < 1/5``），主要学习高度命令（蹲起）。
- ``standing``：约 13.3%（``1/5 <= set_x <= 1/3``），主要学习“稳站”与抗扰。
- ``velocity``：约 66.7%（``set_x > 1/3``），主要学习速度追踪（行走/跑步）。

这些组的作用要点：

- **命令 gating**：

  - ``twist`` 的 ``active_env_group="velocity"``：非 velocity 组的环境被强制 ``twist=0``（站立）。
  - ``height`` 的 ``active_env_group="squat"``：非 squat 组的环境会被设置为 ``inactive_height``（由 robot override 设定），避免 height 命令“干扰”走路环境。

- **奖励 gating**：很多 reward term 都带有 ``env_group``，用于只在某些组里激活（例如：站立稳定项、蹲起相关的几何约束等）。

H1 覆盖：unitree_h1_homie_env_cfg
--------------------------------

路径：``src/mjlab/tasks/homie/config/h1/env_cfgs.py::unitree_h1_homie_env_cfg``

Homie 的实现模式依然是 “**base cfg + robot-specific override**”。H1 override 主要做了这些事：

- **切平地 + 关闭地形课程**：把 terrain 切成 plane，同时移除 ``terrain_levels`` curriculum。
- **动作拆分**：策略只控制下肢（髋/膝/踝）；上身用 policy-free action 生成平滑扰动（见下一节）。
- **命令与脚部几何绑定**：为 height 命令填入 ``foot_site_names``，并设置 squat/standing 的高度范围与 ``inactive_height``：

  - ``height_cmd.ranges.height = (0.4, 0.98)``
  - ``height_cmd.inactive_height = 0.98``（非 squat 组保持稳定站立高度）

- **传感器与接触惩罚**：加入 ``self_collision`` 与 ``hip_knee_ground_contact`` 接触传感器，并把 ``hip_knee_contact`` 奖励项连上线。
- **脚底“平行”奖励落地**：为 ``feet_ground_parallel`` / ``feet_parallel`` 填入 H1 的脚底四角 sites（``*_foot_fi/fo/ri/ro``），并对右脚 sites 做了重排以匹配左右脚局部坐标系。
- **干扰与随机化**：用外力脉冲的方式 push 机器人，并在 reset 时对手部施加 0–5kg 的等效下压力（``hand_load``）。
- **可选夹爪版本**：``hands=True`` 时挂载 2F85，并加入 policy-free 的 ``gripper`` 动作 + interval 采样事件（见后文）。

核心特性：UpperBodyPoseAction（policy-free，上身 0 维 action）
------------------------------------------------------------

路径：``src/mjlab/tasks/homie/config/h1/env_cfgs.py``

这是 Homie 任务最独特的抽象。在 ``actions`` 字典中，除了策略控制的 ``joint_pos``，还有一个 ``upper_body_pose``（策略侧 action dim = 0）：

*   **零动作维度** ：它在策略侧的维度为 0，不增加神经网络的输出负担。
*   **平滑差值** ：它内部维护一个目标姿态，每步通过 ``torch.lerp`` 向其靠近。
*   **定时重采样** ：通过 ``EventTermCfg`` 定期采样新的上身随机姿态（H1 默认每 2s 重采样一次）。
*   **速度限幅（可选）** ：通过 ``max_speed_rad_s`` 对每步目标变化做 rate limit，避免上身“抽搐式”扰动。

.. code-block:: python

   # 上身动作配置（Policy-Free）
   cfg.actions["upper_body_pose"] = UpperBodyPoseActionCfg(
       entity_name="robot",
       joint_names=upper_body_joint_expr,
       interp_rate=0.05,
       max_speed_rad_s=1.0,
       target_range=(-0.6, 0.6),
       initial_ratio=0.0, # 训练时初始无动作，随课程增加（play 直接给 1.0）
       use_sampled_ratio=True,
   )

   # interval 事件：定期采样新的上身目标（目标范围更大，但会被 joint limit + ratio clamp）
   cfg.events["upper_body_random_targets"] = EventTermCfg(
       func=_sample_upper_body_targets_with_curriculum,
       mode="interval",
       interval_range_s=(2.0, 2.0),
       params={
           "action_name": "upper_body_pose",
           "target_range": (-3.0, 1.0),
           "start_step": step_threshold,
       },
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
           "max_ratio": 1.0,
           "start_step": step_threshold,
       },
   )

Rewards & Terminations：混合任务的平衡艺术
----------------------------------------

Homie 任务需要在“行走”和“蹲起”两个目标之间寻找平衡，同时忽略上身干扰。

1) 奖励设计：多任务解耦
^^^^^^^^^^^^^^^^^^^^^^

*   **按环境分组 (Env Grouping)** ：

    *   并不是所有环境都在做同一个任务。很多奖励通过 ``env_group=...`` 做 gating。
    *   例如：H1 override 会额外加入 ``track_*_standing``，专门让 standing 组更“稳”。
*   **对抗干扰的正则项** ：

    *   ``knee_deviation_reward`` : 在蹲起过程中，惩罚膝盖相对于脚尖的水平偏移，引导出合理的蹲姿。
    *   ``upright`` : 强力维持躯干直立，这是抵消上身随机摆动干扰的关键。
    *   ``feet_ground_parallel`` / ``feet_parallel`` : 约束脚底与地面/左右脚之间的相对姿态（需要机器人侧填入脚底 corner sites）。
    *   ``hip_knee_contact`` / ``self_collisions`` : 将“摔倒/自碰撞/膝触地”等问题更多放进 reward 里处理，而不是过早终止。

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

with_hands：带夹爪版本（policy-free）
-----------------------------------

路径：``src/mjlab/tasks/homie/config/h1/__init__.py`` 与 ``src/mjlab/tasks/homie/config/h1/env_cfgs.py``

如果你选择 ``Mjlab-Homie-Unitree-H1-with_hands``：

- 机器人侧会通过 ``get_h1_robot_cfg(hands=...)`` 挂载 2F85（默认配置见 ``_default_hands_cfg``）。
- 环境侧会添加一个 policy-free 的 ``gripper`` action（0 维），并用 interval event 定期采样夹爪目标（类似上身动作的思路）。

总结：为什么参考 Homie？
----------------------

如果你想开发以下功能的任务，Homie 是最好的参考：

1.  **多任务混合** ：同时处理速度追踪与高度控制。
2.  **身体部分控制** ：策略只控制全身的一部分关节，另一部分关节按预设脚本或随机游走。
3.  **高级课程学习** ：不仅仅改环境参数（如摩擦力），还动态修改动作项的行为。
