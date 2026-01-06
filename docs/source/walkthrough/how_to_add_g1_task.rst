.. _walkthrough-how-to-add-g1-task:

如何新增一个 G1 RL 任务（从 0 到可训练）
======================================

这一章给你一条最短路径：**复制一个现有任务骨架 → 替换 mdp terms → 注册 task id → 训练/验证。**

最终你会得到一个新的 task id，可以直接：

.. code-block:: bash

   uv run train Mjlab-MyTask-Flat-Unitree-G1 --env.scene.num-envs 4096

关键设计：tasks 是“导入即注册”的
------------------------------

``train.py`` / ``play.py`` 在解析 task id 前都会执行：

.. code-block:: python

   import mjlab.tasks  # noqa: F401

而 ``mjlab.tasks`` 的 ``__init__.py`` 会递归导入所有子包（跳过 ``.mdp`` 等黑名单），从而触发各任务的注册代码执行。

路径：``src/mjlab/tasks/__init__.py``、``src/mjlab/utils/lab_api/tasks/importer.py``

.. code-block:: python

   # file: src/mjlab/tasks/__init__.py
   from mjlab.utils.lab_api.tasks.importer import import_packages
   _BLACKLIST_PKGS = ["utils", ".mdp"]
   import_packages(__name__, _BLACKLIST_PKGS)

因此：**只要你的新任务包放在 ``src/mjlab/tasks/<your_task>/`` 且包含注册代码，就会被自动发现。**

目录骨架（推荐）
--------------

以 ``my_task`` 为例：

.. code-block:: text

   src/mjlab/tasks/my_task/
     __init__.py                  # 可为空（但建议有一句 docstring）
     my_task_env_cfg.py           # base cfg（定义任务 MDP）
     mdp/
       __init__.py                # re-export envs.mdp + 自己的 terms（仿照 velocity/tracking）
       observations.py
       rewards.py
       terminations.py
       commands.py                # 如果需要 CommandManager
       curriculums.py             # 如果需要 CurriculumManager
     config/
       __init__.py
       g1/
         __init__.py              # register_mjlab_task（注册 task_id）
         env_cfgs.py              # g1 override（填空式覆盖）
         rl_cfg.py                # PPO/runner 配置
     rl/
       __init__.py
       runner.py                  # 可选：自定义 runner（如 velocity/tracking 用于导出 onnx）

Step-by-step：从 base cfg 写到能跑
--------------------------------

Step 1：写 base cfg（只定义任务逻辑，不绑特定机器人）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径建议：``src/mjlab/tasks/my_task/my_task_env_cfg.py``

你要返回一个 ``ManagerBasedRlEnvCfg``，并把 MDP 拆成 dict：

.. code-block:: python

   # file: src/mjlab/tasks/my_task/my_task_env_cfg.py
   from mjlab.envs import ManagerBasedRlEnvCfg
   from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
   from mjlab.managers.reward_manager import RewardTermCfg
   from mjlab.managers.termination_manager import TerminationTermCfg
   from mjlab.managers.scene_entity_config import SceneEntityCfg
   from mjlab.scene import SceneCfg
   from mjlab.sim import SimulationCfg, MujocoCfg
   from mjlab.terrains import TerrainImporterCfg
   from mjlab.tasks.my_task import mdp

   def make_my_task_env_cfg() -> ManagerBasedRlEnvCfg:
       observations = {
           "policy": ObservationGroupCfg(
               terms={
                   "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                   "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
               },
               concatenate_terms=True,
               enable_corruption=True,
           )
       }

       rewards = {
           "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
       }
       terminations = {
           "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
       }

       return ManagerBasedRlEnvCfg(
           scene=SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1),
           observations=observations,
           actions={...},     # 复用 JointPositionActionCfg 最方便（见 velocity/tracking）
           rewards=rewards,
           terminations=terminations,
           decimation=4,
           episode_length_s=10.0,
           sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
       )

**建议原则**：

- base cfg 里尽量只写 “任务通用的结构”，把机器人差异（body/site/geom 名字、action scale、sensors）留给 override。
- 需要引用 robot 的局部组件时，用 ``SceneEntityCfg("robot", joint_names=..., body_names=..., site_names=...)`` 作为 params。

Step 2：写 mdp terms（尽量复用 envs.mdp）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

像 velocity/tracking 一样，在 ``mdp/__init__.py`` 里 re-export：

.. code-block:: python

   # file: src/mjlab/tasks/my_task/mdp/__init__.py
   from mjlab.envs.mdp import *  # 通用：joint_pos_rel / time_out / randomize_field / ...
   from .rewards import *
   from .observations import *

这样你在 cfg 里可以直接 ``from mjlab.tasks.my_task import mdp``，并同时拿到通用 + 自定义 term。

Step 3：写 g1 override（把“占位符”补齐）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径建议：``src/mjlab/tasks/my_task/config/g1/env_cfgs.py``

这一层通常要做：

- ``cfg.scene.entities = {"robot": get_g1_robot_cfg()}``
- 加 sensors（如 contact sensor）
- 设置 ``JointPositionActionCfg.scale = G1_ACTION_SCALE``
- 补齐所有用到的 ``SceneEntityCfg(..., body_names/site_names/geom_names=...)``
- play mode：关 corruption、关某些 randomization、把 episode_length 设大

Step 4：注册 task id（把 cfg 放进 registry）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

路径建议：``src/mjlab/tasks/my_task/config/g1/__init__.py``

.. code-block:: python

   from mjlab.tasks.registry import register_mjlab_task
   from .env_cfgs import unitree_g1_my_task_env_cfg
   from .rl_cfg import unitree_g1_my_task_ppo_runner_cfg

   register_mjlab_task(
     task_id="Mjlab-MyTask-Flat-Unitree-G1",
     env_cfg=unitree_g1_my_task_env_cfg(),
     play_env_cfg=unitree_g1_my_task_env_cfg(play=True),
     rl_cfg=unitree_g1_my_task_ppo_runner_cfg(),
     runner_cls=None,  # 或自定义 runner
   )

Step 5：先 play（dummy agent）再 train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run play Mjlab-MyTask-Flat-Unitree-G1 --agent random
   uv run train Mjlab-MyTask-Flat-Unitree-G1 --env.scene.num-envs 1024

常见踩坑清单（按出现频率排序）
----------------------------

- **shape 不一致**：obs term 必须返回 ``(num_envs, ...)``；reward/termination 必须返回 ``(num_envs,)``。
- **device 不一致**：不要在 term 里创建默认在 CPU 的 tensor；用 ``device=env.device``。
- **SceneEntityCfg 未补齐**：base cfg 里留了 ``geom_names=()`` / ``site_names=()``，robot override 忘记填会导致 ids 解析为空或不符合预期。
- **Python 循环太多**：4096 env 下会直接变慢；优先用 torch 向量化。
- **tracking motion_file 没注入**：tracking 类任务请按 train.py 的方式走（``--registry-name`` 或 ``--motion-file``）。
