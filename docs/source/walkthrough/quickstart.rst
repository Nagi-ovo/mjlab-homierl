.. _walkthrough-quickstart:

快速上手：最短开发闭环（训练 / Play / 改任务）
===========================================

这一章只回答三个问题：

- **怎么跑起来？**
- **怎么验证你的 MDP 没写错？**
- **怎么最快迭代 reward/obs/随机化？**

0) 先把能跑的 baseline 跑通
--------------------------

velocity（Unitree G1，平地）：

.. code-block:: bash

   # 训练：常用参数覆盖（tyro 支持直接覆盖 dataclass 字段）
   uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

   # play：从 W&B 拉最新 checkpoint（或指定 --checkpoint-file）
   uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id

tracking（Unitree G1，motion imitation）：

.. code-block:: bash

   # tracking 训练必须给 registry motion（W&B artifact），train.py 会把 motion_file 注入到 command cfg
   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
     --registry-name your-org/motions/motion-name \
     --env.scene.num-envs 4096

   uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id

homie（Unitree H1，混合 velocity + squat + 扰动）：

.. code-block:: bash

   # 训练：H1 更重、更复杂，建议先从较少的 num-envs 开始
   uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 2048

   # play：从 W&B 拉 checkpoint（或指定 --checkpoint-file）
   uv run play Mjlab-Homie-Unitree-H1 --wandb-run-path your-org/mjlab/run-id

Homie 也提供了一个可选的“带夹爪”版本（额外挂载 Robotiq 2F85，并加入 policy-free 的夹爪随机动作）：

.. code-block:: bash

   uv run train Mjlab-Homie-Unitree-H1-with_hands --env.scene.num-envs 2048
   uv run play Mjlab-Homie-Unitree-H1-with_hands --wandb-run-path your-org/mjlab/run-id

1) 用 dummy agent 做“MDP 体检”（强烈建议）
----------------------------------------

在训练之前，先用零动作 / 随机动作把环境跑上几百步：

.. code-block:: bash

   uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero
   uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent random

tracking 的 dummy agent 仍然需要 motion registry（否则 command 无法加载 motion）：

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
     --agent random \
     --registry-name your-org/motions/motion-name

Homie 任务的 dummy agent 不需要额外参数（它没有 tracking 的 motion registry 依赖）：

.. code-block:: bash

   uv run play Mjlab-Homie-Unitree-H1 --agent zero
   uv run play Mjlab-Homie-Unitree-H1 --agent random

   # 带夹爪版本也一样（夹爪动作是 policy-free 的 0 维 action）
   uv run play Mjlab-Homie-Unitree-H1-with_hands --agent random

你在这一步要看什么？

- **能否稳定跑步**：是否有 NaN/Inf（必要时打开 NaN guard，见 :ref:`walkthrough-debugging-perf`）。
- **obs/action 维度是否正确**：维度错会直接在 ``ActionManager`` 或 gym space 构建时报错。
- **reward 是否“有信号”**：看 viewer / logger（extras 里会有 metrics）。

2) 最常见的修改点在哪里（velocity / tracking / homie）
-----------------------------------------------------

这两个任务都采用“ **base env cfg + robot-specific override** ”的模式：

- base cfg（任务定义）：  

  - velocity：``src/mjlab/tasks/velocity/velocity_env_cfg.py::make_velocity_env_cfg``  
  - tracking：``src/mjlab/tasks/tracking/tracking_env_cfg.py::make_tracking_env_cfg``

- g1 override（填空式覆盖）：  

  - velocity：``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``  
  - tracking：``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``

Homie 也遵循同样的结构（base cfg + H1 override）：

- base cfg（任务定义）：``src/mjlab/tasks/homie/homie_env_cfg.py::make_homie_env_cfg``  
- h1 override（填空式覆盖）：``src/mjlab/tasks/homie/config/h1/env_cfgs.py::unitree_h1_homie_env_cfg``  

如果你要改 Homie（强烈建议先看 :ref:`walkthrough-task-homie-h1`），最常改的是：

- **动作拆分与缩放**：策略只控下肢；上身/夹爪是 policy-free action（见 H1 override）。
- **命令与 env grouping**：哪些环境在走、哪些在蹲、哪些在站（``mdp.assign_homie_env_groups``）。
- **干扰/随机化**：push、手部负载、摩擦随机化等（``events``）。

建议你按下面顺序迭代（最快见效）：

- **reward 权重**：改 ``RewardTermCfg(weight=...)`` （优先）。  
  例如：velocity 的 ``cfg.rewards["angular_momentum"].weight`` / ``cfg.rewards["self_collisions"]``。
- **观测项**：在 ``observations["policy"].terms`` 增减 ``ObservationTermCfg``。  
  tracking 甚至提供了 ``has_state_estimation=False`` 的“去掉部分状态”版本。
- **随机化**：改 ``events`` （startup/reset/interval）。  
  domain randomization 用 ``EventTermCfg(domain_randomization=True, params={"field": ...})``。
- **命令分布**：改 ``commands`` 的 ranges（velocity 的 twist ranges / tracking 的 sampling_mode）。

3) 任务 cfg 是怎么被 “cli 覆盖” 的？
-----------------------------------

``train.py`` / ``play.py`` 使用 tyro 解析 dataclass，因此你可以直接从命令行覆盖任意字段：

.. code-block:: bash

   # 覆盖 num_envs、episode 长度、viewer 分辨率、以及某个 reward 权重（示例）
   uv run train Mjlab-Velocity-Flat-Unitree-G1 \
     --env.scene.num-envs 2048 \
     --env.episode-length-s 15 \
     --env.viewer.width 1280 --env.viewer.height 720

.. note::

   tyro 的覆盖只对 dataclass 字段直接生效；像 ``dict`` 内部的某个 key（例如 rewards 某项）
   通常更适合在 ``config/g1/env_cfgs.py`` 里用 Python 代码修改，避免 CLI 变得不可维护。

