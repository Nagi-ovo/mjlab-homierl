.. _walkthrough-rewards-and-terminations:

奖励函数与终止条件 (Rewards and Terminations)
===========================================

在强化学习任务中，奖励函数 (Reward Function) 决定了智能体“想做什么”，而终止条件 (Termination Conditions) 决定了任务“何时结束”。本章将以经典的 **CartPole（倒立摆）** 为例，讲解如何在 ``mjlab`` 中定义和配置它们。

奖励函数 (Reward Function)
-------------------------

奖励函数通过 ``RewardManager`` 管理。它是一组“项 (Terms)”的集合，每个项都有一个权重 (Weight)，最终奖励是所有项的加权之和。

**CartPole 奖励结构示例：**

1.  **保持直立 (Staying Upright)** ：当杆子（pole）的倾角越小，奖励越高。
2.  **位置惩罚 (Position Penalty)** ：当小车（cart）靠近轨道边缘时，给予负向奖励（惩罚）。

.. code-block:: python

   # 配置文件中的奖励配置
   rewards = {
       "stay_upright": RewardTermCfg(
           func=mdp.upright_reward, # 保持直立
           weight=1.0,
           params={"std": 0.2}
       ),
       "cart_pos_penalty": RewardTermCfg(
           func=mdp.cart_position_penalty, # 位置惩罚
           weight=-0.1,
           params={"threshold": 2.0}
       ),
   }

如何实现自定义奖励函数
^^^^^^^^^^^^^^^^^^^^

奖励函数通常是一个接收 ``env`` 和其他自定义参数的 Python 函数。

.. code-block:: python

   # file: src/mjlab/tasks/cartpole/mdp/rewards.py
   import torch

   def upright_reward(env, std: float):
       # 获取杆子的上方向向量（假设杆子是 robot entity 的一部分）
       # 这里演示如何从 scene 中提取数据
       pole_quat = env.scene["robot"].data.body_quat[:, pole_id]
       # ... 计算与垂直方向的夹角 ...
       # 返回 shape 为 (num_envs,) 的 tensor
       return torch.exp(-angle**2 / std**2)

终止条件 (Termination Conditions)
------------------------------

终止条件通过 ``TerminationManager`` 管理。当任何一个终止项返回 ``True`` 时，该环境就会被重置。

**CartPole 终止条件示例：**

1.  **杆子倒下 (Pole Falls Over)** ：当倾角超过一定阈值（如 15 度）。
2.  **出界 (Out of Bounds)** ：当小车水平位移超过轨道长度（如 2.4 米）。
3.  **超时 (Time Limit)** ：达到单局最大步数或时间。

.. code-block:: python

   # 配置文件中的终止配置
   terminations = {
       "pole_fell": TerminationTermCfg(
           func=mdp.pole_tilt_limit,
           params={"limit_angle": 15.0}
       ),
       "out_of_bounds": TerminationTermCfg(
           func=mdp.cart_out_of_bounds,
           params={"limit_dist": 2.4}
       ),
       "time_out": TerminationTermCfg(
           func=mdp.time_out, # 这是一个内置函数
           time_out=True      # 标记为超时（truncated），而非由于失败导致（terminated）
       ),
   }

如何配置终止管理器
^^^^^^^^^^^^^^^^

在 ``ManagerBasedRlEnvCfg`` 中，你只需要将上述字典赋值给 ``terminations`` 属性。管理器会自动在每步结束时调用这些函数，并汇总成一个布尔向量。

.. code-block:: python

   @dataclass
   class CartPoleEnvCfg(ManagerBasedRlEnvCfg):
       # ... 其他配置 ...
       rewards: dict[str, RewardTermCfg] = rewards
       terminations: dict[str, TerminationTermCfg] = terminations

**核心逻辑提示：**

*   **Terminated vs Truncated** : ``time_out=True`` 的项会将环境标记为 "truncated"（截断），这在处理 PPO 的价值函数引导时非常重要。而"杆子倒下"则是 "terminated"（真终止）。
*   **并行计算** : 所有的 reward 和 termination 函数都应该支持并行计算（输入 ``env``，返回长度为 ``num_envs`` 的 tensor）。

