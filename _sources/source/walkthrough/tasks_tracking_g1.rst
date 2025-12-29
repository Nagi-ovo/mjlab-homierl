.. _walkthrough-task-tracking-g1:

实例二：Motion Tracking / Imitation（Unitree G1）
================================================

tracking 任务与 velocity 的最大不同点在于：**command 不再是“采样一个目标速度”，而是“加载一段 reference motion”，并在每个 step 推进时间索引。**

因此 tracking 的核心是 ``MotionCommand`` （一个 ``CommandTerm`` 子类），而 rewards/terminations 主要度量“reference vs robot”的误差。

任务骨架：make_tracking_env_cfg（base cfg）
-----------------------------------------

路径：``src/mjlab/tasks/tracking/tracking_env_cfg.py``

base cfg 同样定义完整 MDP，但把 motion 相关的关键信息留给 robot override（例如 motion_file、anchor_body_name、body_names）：

.. code-block:: python

   # file: src/mjlab/tasks/tracking/tracking_env_cfg.py
   commands = {
     "motion": MotionCommandCfg(
       entity_name="robot",
       resampling_time_range=(1e9, 1e9),
       motion_file="",        # override
       anchor_body_name="",   # override
       body_names=(),         # override
       sampling_mode="adaptive",
       debug_vis=True,
     )
   }

MotionCommand：tracking 的“发动机”
----------------------------------

路径：``src/mjlab/tasks/tracking/mdp/commands.py``

你可以把 ``MotionCommand`` 看成一个“reference motion 生成器”，它负责：

- 读取 motion npz（joint_pos/joint_vel/body_pos_w/body_quat_w/...）
- 选择当前 time_step（支持 start/uniform/adaptive sampling）
- 生成“相对 anchor 的 reference pose”（用于度量与观测）
- 在 reset 时把 robot 状态初始化到参考附近（并加随机扰动，提高鲁棒性）

Rewards & Terminations：高精度的影子模仿
--------------------------------------

在 Tracking 任务中，奖励和终止条件都围绕着 **“如何让机器人成为参考运动的影子”** 展开。

1) 奖励项：多维度的误差惩罚
^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **姿态误差 (Pose Reward)** ：

    *   ``joint_pos_tracking`` : 每一个关节角度都要贴合 reference。
    *   ``body_pos_tracking`` / ``body_quat_tracking`` : 手、脚、躯干在世界坐标系（相对 anchor）的位置和姿态。
*   **速度误差 (Velocity Reward)** ：

    *   ``joint_vel_tracking`` / ``body_lin_vel_tracking`` : 动态跟随的平滑度，防止只有姿态对、但运动生硬。

2) 终止条件：严格的精度控制
^^^^^^^^^^^^^^^^^^^^^^^^^^

与 Velocity 任务不同，Tracking 任务通常使用“自杀式”训练：

*   **轨迹偏离 (Tracking Error Termination)** ：

    *   如果躯干（torso）或脚部（feet）偏离参考位置超过阈值（如 0.3-0.5m），说明机器人已经由于失稳无法追随， **直接终止** 。
    *   这能有效防止策略在已经搞砸的情况下浪费算力。
*   **自碰撞 (Self Collision)** ：

    *   为了模仿人类或高难度动作，自碰撞通常是严格禁止的（终止触发）。

训练入口的关键逻辑：motion_file 如何注入？
----------------------------------------

路径：``src/mjlab/scripts/train.py``

train.py 通过检查 env_cfg 里是否存在 ``commands["motion"]`` 且类型为 ``MotionCommandCfg`` 来判断 tracking 任务，
并要求你提供 ``--registry-name`` （W&B artifact），然后把 artifact 下载路径写入 ``motion_cmd.motion_file``：

.. code-block:: python

   # file: src/mjlab/scripts/train.py
   is_tracking_task = (
     cfg.env.commands is not None
     and "motion" in cfg.env.commands
     and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
   )
   if is_tracking_task:
     artifact = wandb.Api().artifact(registry_name)
     motion_cmd = cfg.env.commands["motion"]
     motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")

这段逻辑的设计含义是： **motion 数据集管理与训练 run 解耦** （你不需要把 motion 文件硬编码在代码里）。

g1 覆盖：unitree_g1_flat_tracking_env_cfg
----------------------------------------

路径：``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``

这一层把 base cfg “落地到 g1”：

- 绑定 robot entity：``get_g1_robot_cfg()``
- 配置 self-collision sensor
- 设置 ``MotionCommandCfg.anchor_body_name`` 与 ``body_names`` （决定哪些 body 参与 tracking）
- 设置 termination 的 end-effector bodies
- 提供 ``has_state_estimation=False`` 分支：删除部分观测项，模拟状态估计不可用的场景

例如（has_state_estimation 分支）：

.. code-block:: python

   # file: src/mjlab/tasks/tracking/config/g1/env_cfgs.py
   if not has_state_estimation:
       new_policy_terms = {k: v for k, v in cfg.observations["policy"].terms.items()
                           if k not in ["motion_anchor_pos_b", "base_lin_vel"]}
       cfg.observations["policy"] = ObservationGroupCfg(
           terms=new_policy_terms, concatenate_terms=True, enable_corruption=True
       )

Task 注册
---------

路径：``src/mjlab/tasks/tracking/config/g1/__init__.py``

除了普通版本，还注册了 “No-State-Estimation” 版本（通过 cfg 变体实现）：

.. code-block:: python

   # file: src/mjlab/tasks/tracking/config/g1/__init__.py
   register_mjlab_task(
     task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
     env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
     play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
     rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
     runner_cls=MotionTrackingOnPolicyRunner,
   )

你要改 tracking：通常改哪里？
----------------------------

1. **body list / anchor** ：最关键（直接决定误差定义与训练难度）  
   路径：``src/mjlab/tasks/tracking/config/g1/env_cfgs.py`` （``motion_cmd.body_names`` / ``anchor_body_name``）
2. **sampling_mode** ：start/uniform/adaptive 的选择会显著影响学习曲线  
   路径：``MotionCommandCfg.sampling_mode`` （base cfg / play mode override）
3. **reward 标准差 std** ：控制 "exp(-err/std^2)" 的形状（学习信号强弱）  
   路径：``src/mjlab/tasks/tracking/tracking_env_cfg.py`` （rewards dict）
4. **termination 阈值** ：太严会学不起来；太松会姿态漂移  
   路径：``src/mjlab/tasks/tracking/tracking_env_cfg.py`` + g1 override 补 ``body_names``


