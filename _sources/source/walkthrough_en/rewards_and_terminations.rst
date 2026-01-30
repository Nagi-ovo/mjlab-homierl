.. _walkthrough-en-rewards-and-terminations:

Rewards and Terminations
========================

In RL tasks, the reward function determines “what the agent wants”, while termination conditions define “when an episode ends”.
This chapter uses the classic **CartPole** example to show how to define and configure them in ``mjlab``.

Reward function
---------------

Rewards are managed by ``RewardManager``. A reward is a set of **terms**, each with a **weight**. The final reward is the weighted sum of all terms.

**Example CartPole reward structure:**

1. **Staying upright**: the smaller the pole tilt angle, the higher the reward.
2. **Position penalty**: give negative reward when the cart approaches the edge of the track.

.. code-block:: python

   # Reward configuration in env cfg
   rewards = {
       "stay_upright": RewardTermCfg(
           func=mdp.upright_reward,  # staying upright
           weight=1.0,
           params={"std": 0.2},
       ),
       "cart_pos_penalty": RewardTermCfg(
           func=mdp.cart_position_penalty,  # position penalty
           weight=-0.1,
           params={"threshold": 2.0},
       ),
   }

How to implement a custom reward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A reward term is usually a Python function that takes ``env`` plus custom parameters.

.. code-block:: python

   # file: src/mjlab/tasks/cartpole/mdp/rewards.py
   import torch

   def upright_reward(env, std: float):
       # Get the pole's up direction (assume the pole belongs to the "robot" entity)
       # This demonstrates how to read data from the scene.
       pole_quat = env.scene["robot"].data.body_quat[:, pole_id]
       # ... compute angle to vertical ...
       # Return a tensor of shape (num_envs,)
       return torch.exp(-angle**2 / std**2)

Termination conditions
----------------------

Terminations are managed by ``TerminationManager``. If any termination term returns ``True``, that env will be reset.

**Example CartPole terminations:**

1. **Pole falls over**: tilt angle exceeds a threshold (e.g., 15 degrees).
2. **Out of bounds**: cart position exceeds the track limit (e.g., 2.4 meters).
3. **Time limit**: reach the maximum episode steps/time.

.. code-block:: python

   # Termination configuration in env cfg
   terminations = {
       "pole_fell": TerminationTermCfg(
           func=mdp.pole_tilt_limit,
           params={"limit_angle": 15.0},
       ),
       "out_of_bounds": TerminationTermCfg(
           func=mdp.cart_out_of_bounds,
           params={"limit_dist": 2.4},
       ),
       "time_out": TerminationTermCfg(
           func=mdp.time_out,  # built-in
           time_out=True,      # mark as truncated, not failure terminated
       ),
   }

How to configure the termination manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``ManagerBasedRlEnvCfg``, you just assign the dict to ``terminations``. The manager will call the terms each step and aggregate them into a boolean vector.

.. code-block:: python

   @dataclass
   class CartPoleEnvCfg(ManagerBasedRlEnvCfg):
       # ... other fields ...
       rewards: dict[str, RewardTermCfg] = rewards
       terminations: dict[str, TerminationTermCfg] = terminations

**Key notes:**

- **Terminated vs truncated**: terms with ``time_out=True`` mark the env as "truncated". This is important for PPO-style value bootstrapping. "Pole fell" is a true "terminated".
- **Vectorization**: reward/termination functions should be fully vectorized (take ``env``, return a tensor of length ``num_envs``).
