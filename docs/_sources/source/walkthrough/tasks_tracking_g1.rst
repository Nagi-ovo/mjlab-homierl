.. _walkthrough-task-tracking-g1:

实例二：Motion Tracking / Imitation（Unitree G1）
================================================

tracking 任务与 velocity 的最大不同点在于：**command 不再是“采样一个目标速度”，而是“加载一段 reference motion”，并在每个 step 推进时间索引。**

因此 tracking 的核心是 ``MotionCommand``（一个 ``CommandTerm`` 子类），而 rewards/terminations 主要度量“reference vs robot”的误差。

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

例如（读取 motion + 推进 time_steps）：

.. code-block:: python

   # file: src/mjlab/tasks/tracking/mdp/commands.py
   class MotionCommand(CommandTerm):
       def __init__(self, cfg: MotionCommandCfg, env):
           self.motion = MotionLoader(cfg.motion_file, body_indexes, device=env.device)
           self.time_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

       def _resample_command(self, env_ids):
           if self.cfg.sampling_mode == "start":
               self.time_steps[env_ids] = 0
           elif self.cfg.sampling_mode == "uniform":
               self.time_steps[env_ids] = torch.randint(0, self.motion.time_step_total, (len(env_ids),), device=self.device)
           else:
               self._adaptive_sampling(env_ids)
           # 写入 root/joint state 到 sim（把机器人拉回参考附近）
           self.robot.write_joint_state_to_sim(...)
           self.robot.write_root_state_to_sim(...)

       def _update_command(self):
           self.time_steps += 1
           if any(time_steps >= time_step_total): resample
           # 计算 relative reference（anchor 对齐）
           self.body_pos_relative_w = ...
           self.body_quat_relative_w = ...

训练入口的关键逻辑：motion_file 如何注入？
----------------------------------------

路径：``src/mjlab/scripts/train.py``

train.py 通过检查 env_cfg 里是否存在 ``commands["motion"]`` 且类型为 ``MotionCommandCfg`` 来判断 tracking 任务，
并要求你提供 ``--registry-name``（W&B artifact），然后把 artifact 下载路径写入 ``motion_cmd.motion_file``：

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

这段逻辑的设计含义是：**motion 数据集管理与训练 run 解耦**（你不需要把 motion 文件硬编码在代码里）。

g1 覆盖：unitree_g1_flat_tracking_env_cfg
----------------------------------------

路径：``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``

这一层把 base cfg “落地到 g1”：

- 绑定 robot entity：``get_g1_robot_cfg()``
- 配置 self-collision sensor
- 设置 ``MotionCommandCfg.anchor_body_name`` 与 ``body_names``（决定哪些 body 参与 tracking）
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

1. **body list / anchor**：最关键（直接决定误差定义与训练难度）  
   路径：``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``（``motion_cmd.body_names`` / ``anchor_body_name``）
2. **sampling_mode**：start/uniform/adaptive 的选择会显著影响学习曲线  
   路径：``MotionCommandCfg.sampling_mode``（base cfg / play mode override）
3. **reward 标准差 std**：控制 “exp(-err/std^2)” 的形状（学习信号强弱）  
   路径：``src/mjlab/tasks/tracking/tracking_env_cfg.py``（rewards dict）
4. **termination 阈值**：太严会学不起来；太松会姿态漂移  
   路径：``src/mjlab/tasks/tracking/tracking_env_cfg.py`` + g1 override 补 ``body_names``


