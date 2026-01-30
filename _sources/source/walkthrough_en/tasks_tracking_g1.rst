.. _walkthrough-en-task-tracking-g1:

Example 2: Motion Tracking / Imitation (Unitree G1)
===================================================

The key difference from velocity is:
**the command is no longer “sample a target velocity”, but “load a reference motion” and advance a time index every step.**

So the core of tracking is ``MotionCommand`` (a ``CommandTerm`` subclass), and rewards/terminations mainly measure “reference vs robot” errors.

Task skeleton: make_tracking_env_cfg (base cfg)
----------------------------------------------

Path: ``src/mjlab/tasks/tracking/tracking_env_cfg.py``

The base cfg defines the full MDP, but leaves motion-specific details to robot overrides (e.g., motion_file, anchor_body_name, body_names):

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

MotionCommand: the “engine” of tracking
---------------------------------------

Path: ``src/mjlab/tasks/tracking/mdp/commands.py``

You can think of ``MotionCommand`` as a “reference motion generator” responsible for:

- loading the motion npz (joint_pos/joint_vel/body_pos_w/body_quat_w/…)
- selecting the current time_step (supports start/uniform/adaptive sampling)
- generating a reference pose relative to an anchor (used for metrics and observations)
- initializing the robot near the reference pose at reset (plus random perturbations for robustness)

Rewards & terminations: high-precision shadow imitation
-------------------------------------------------------

In tracking, rewards and terminations revolve around:
**“make the robot become a shadow of the reference motion.”**

1) Rewards: multi-dimensional error penalties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Pose errors**:

  - ``joint_pos_tracking``: track each joint angle
  - ``body_pos_tracking`` / ``body_quat_tracking``: track hands/feet/torso positions and orientations in world frame (relative to anchor)

- **Velocity errors**:

  - ``joint_vel_tracking`` / ``body_lin_vel_tracking``: encourage smooth motion, not just matching poses

2) Terminations: strict accuracy control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike velocity, tracking often uses “fail-fast” training:

- **Tracking error termination**:

  - if torso/feet deviate from reference beyond a threshold (e.g., 0.3–0.5m), the robot is likely unstable → **terminate immediately**
  - prevents wasting compute on hopeless trajectories

- **Self-collision**:

  - for human-like or difficult motions, self-collision is often strictly disallowed (terminate on contact)

Training entrypoint: how is motion_file injected?
-------------------------------------------------

Path: ``src/mjlab/scripts/train.py``

train.py detects tracking tasks by checking for ``commands["motion"]`` of type ``MotionCommandCfg``.
It then requires ``--registry-name`` (a W&B artifact), downloads it, and writes the path into ``motion_cmd.motion_file``:

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

The design intent: **motion dataset management is decoupled from the training run** (you don’t hardcode motion file paths in code).

G1 override: unitree_g1_flat_tracking_env_cfg
---------------------------------------------

Path: ``src/mjlab/tasks/tracking/config/g1/env_cfgs.py``

This layer “grounds” the base cfg to G1:

- bind the robot entity: ``get_g1_robot_cfg()``
- configure the self-collision sensor
- set ``MotionCommandCfg.anchor_body_name`` and ``body_names`` (decide which bodies participate in tracking)
- set termination end-effector bodies
- provide a ``has_state_estimation=False`` branch: remove some obs terms to simulate missing state estimation

Example (has_state_estimation branch):

.. code-block:: python

   # file: src/mjlab/tasks/tracking/config/g1/env_cfgs.py
   if not has_state_estimation:
       new_policy_terms = {k: v for k, v in cfg.observations["policy"].terms.items()
                           if k not in ["motion_anchor_pos_b", "base_lin_vel"]}
       cfg.observations["policy"] = ObservationGroupCfg(
           terms=new_policy_terms, concatenate_terms=True, enable_corruption=True
       )

Task registration
-----------------

Path: ``src/mjlab/tasks/tracking/config/g1/__init__.py``

Besides the default task, a “No-State-Estimation” variant is registered via cfg variants:

.. code-block:: python

   # file: src/mjlab/tasks/tracking/config/g1/__init__.py
   register_mjlab_task(
     task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
     env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
     play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
     rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
     runner_cls=MotionTrackingOnPolicyRunner,
   )

If you want to modify tracking: where do you usually change things?
-------------------------------------------------------------------

1. **Body list / anchor**: most important (directly defines errors and difficulty)
   Path: ``src/mjlab/tasks/tracking/config/g1/env_cfgs.py`` (``motion_cmd.body_names`` / ``anchor_body_name``)
2. **sampling_mode**: start/uniform/adaptive changes learning dynamics significantly
   Path: ``MotionCommandCfg.sampling_mode`` (base cfg / play overrides)
3. **Reward std**: controls the shape of ``exp(-err/std^2)`` (signal strength)
   Path: ``src/mjlab/tasks/tracking/tracking_env_cfg.py`` (rewards dict)
4. **Termination thresholds**: too strict → won’t learn; too loose → drift
   Path: ``src/mjlab/tasks/tracking/tracking_env_cfg.py`` + G1 override (fill ``body_names``)
