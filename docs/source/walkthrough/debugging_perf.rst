.. _walkthrough-debugging-perf:

调试与性能：让你“稳”且“快”
==========================

这章不是 API 复述，而是把最常见的工程问题收敛成一套排障顺序与性能心法。

第一优先级：先确保 MDP 正确
---------------------------

1) dummy agents 体检（zero/random）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/scripts/play.py``

.. code-block:: bash

   uv run play <task_id> --agent zero
   uv run play <task_id> --agent random

这样做的意义是：在 policy 学会任何东西之前，你就能验证：

- action space / observation space 是否一致
- reset 是否稳定
- reward/termination 是否会产生 NaN/Inf
- contact sensor 等依赖是否配置齐全

2) 看清“命令 vs 实际”的差异（Command debug vis）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

velocity 的 ``UniformVelocityCommand`` 会画 velocity arrows（蓝/绿为命令，青/浅绿为实际）：

- 路径：``src/mjlab/tasks/velocity/mdp/velocity_command.py::_debug_vis_impl``

tracking 的 ``MotionCommand`` 支持 ghost robot 或 frames：

- 路径：``src/mjlab/tasks/tracking/mdp/commands.py::_debug_vis_impl``

第二优先级：NaN/Inf 与物理不稳定
-------------------------------

1) NaN Guard（推荐在开发期打开）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/sim/sim.py``（``NanGuard``） + ``src/mjlab/scripts/train.py``（开关）

训练时可以用 flag 打开：

.. code-block:: bash

   uv run train <task_id> --enable-nan-guard True

train.py 会把 ``cfg.env.sim.nan_guard.enabled = True``。

2) NaN 作为 termination（可选）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/envs/mdp/terminations.py::nan_detection``

如果你更倾向于“出现 NaN 立即终止并重置”，可以把它加入 task 的 ``terminations`` dict。

第三优先级：domain randomization 的正确姿势
------------------------------------------

domain randomization 的核心不是“每次 reset 随机一下”，而是 **需要支持 per-env 的 model field**。

1) 事件里声明 domain_randomization=True
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/envs/mdp/events.py::randomize_field`` + ``src/mjlab/managers/event_manager.py``

.. code-block:: python

   EventTermCfg(
     mode="startup",
     func=mdp.randomize_field,
     domain_randomization=True,
     params={"field": "geom_friction", ...},
   )

``EventManager`` 会收集 ``params["field"]``，形成 ``domain_randomization_fields``。

2) env 在 load_managers 时扩展字段
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/envs/manager_based_rl_env.py::load_managers``

.. code-block:: python

   self.event_manager = EventManager(self.cfg.events, self)
   self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

3) CUDA graph 的坑：替换数组后必须重新 capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径：``src/mjlab/sim/sim.py``（类注释 + ``expand_model_fields``）

Simulation 在 CUDA 上会 capture graph（step/forward/reset）。graph 绑定的是“当时的 GPU 内存地址”。
如果你后续替换了 model/data 数组，graph 会静默读取旧地址。

好消息是：``Simulation.expand_model_fields`` 内部会自动调用 ``create_graph()`` 重新 capture。

工程建议：

- **尽量把所有需要扩展的字段**在 env 初始化时一次性确定（通过 EventManager 收集），避免中途替换数组。

性能心法：让训练跑得快
----------------------

1) 训练快慢最敏感的几个旋钮
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **num_envs**：越大越吃 GPU，但也越能摊薄 Python 开销（4096 是 velocity/tracking 常见量级）
- **decimation**：越大，控制频率越低；但每步 physics 还是会跑 ``decimation`` 次
- **term 的向量化程度**：不要在 term 里写 Python for 循环
- **contact sensor 复杂度**：匹配模式、slots 数、reduce 策略都会影响开销

2) 多 GPU
^^^^^^^^^^

路径：``src/mjlab/scripts/train.py``（``--gpu-ids`` + torchrunx）

当选择多 GPU 时，会走 torchrunx 启动多进程，并设置 ``MUJOCO_EGL_DEVICE_ID`` 与 local_rank 对齐。

可观测性：你应该把什么看成“单步的 truth source”
------------------------------------------------

- **env.extras["log"]**：来自各 manager 的 reset 统计（reward episodic sums、termination counts、curriculum state、command metrics 等）  
  路径：``src/mjlab/envs/manager_based_rl_env.py::_reset_idx``

- **RewardManager/CommandTerm metrics**：很多任务会把 metrics 写入 ``env.extras["log"]["Metrics/..."]``  
  例子：velocity 的 angular_momentum_mean、air_time_mean、slip_velocity_mean 等（见 ``src/mjlab/tasks/velocity/mdp/rewards.py``）


