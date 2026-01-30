.. _walkthrough-en-task-velocity-g1:

Example 1: Velocity Tracking (Unitree G1)
=========================================

This chapter uses ``Mjlab-Velocity-Flat-Unitree-G1`` / ``Mjlab-Velocity-Rough-Unitree-G1`` as template tasks to explain:

- how an env cfg is decomposed into MDP managers
- how the G1 config overrides the base cfg (entities, sensors, entity names, weights, play mode)
- what you minimally need to change to create a new locomotion task

Task skeleton: make_velocity_env_cfg (base cfg)
----------------------------------------------

Path: ``src/mjlab/tasks/velocity/velocity_env_cfg.py``

The base cfg does two things:

1. defines the **MDP structure** (obs/actions/commands/events/rewards/terminations/curriculum)
2. leaves placeholders for robot-specific differences (e.g., foot site names, friction geoms, viewer body name, some reward weights)

For example, policy observations (note: commands are part of the observation):

.. code-block:: python

   # file: src/mjlab/tasks/velocity/velocity_env_cfg.py
   policy_terms = {
     "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
     "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
     "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
     "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
     "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
     "actions": ObservationTermCfg(func=mdp.last_action),
     "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
   }

Velocity commands (CommandManager) use ``UniformVelocityCommand`` by default:

.. code-block:: python

   # file: src/mjlab/tasks/velocity/velocity_env_cfg.py
   commands = {
     "twist": UniformVelocityCommandCfg(
       entity_name="robot",
       resampling_time_range=(3.0, 8.0),
       heading_command=True,
       ranges=UniformVelocityCommandCfg.Ranges(
         lin_vel_x=(-1.0, 1.0),
         lin_vel_y=(-1.0, 1.0),
         ang_vel_z=(-0.5, 0.5),
         heading=(-pi, pi),
       ),
     )
   }

The three most important terms to understand
--------------------------------------------

1) Command term: UniformVelocityCommand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/tasks/velocity/mdp/velocity_command.py``

Key features:

- **resampling**: sample velocity commands per env, on a time window
- **heading control**: some envs convert heading error into yaw rate (closer to “walk facing a direction”)
- **standing envs**: sample a fraction of “stand still” commands to teach stopping

2) Rewards & terminations: carrots and sticks for locomotion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Velocity rewards typically follow a layered design: survival → task completion → motion quality.

- **Core task rewards**:

  - ``track_linear_velocity``: match actual velocity to the ``twist`` command
  - ``track_angular_velocity``: match angular velocity

- **Regularizers (motion quality)**:

  - ``joint_pos_limits`` / ``joint_vel_limits``: penalize joints near limits or moving too fast
  - ``action_rate_l2``: penalize action discontinuities → smoother motion
  - ``feet_air_time`` / ``feet_clearance``: encourage natural gait patterns

- **Terminations (safety boundaries)**:

  - ``fell_over`` (``bad_orientation``): reset when torso tilt exceeds 70 degrees (prevents learning “crawl on the ground”)
  - ``time_out``: enforce reset after a fixed horizon (truncated) to keep exploration going

3) Curriculum: terrain_levels_vel + commands_vel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Path: ``src/mjlab/tasks/velocity/mdp/curriculums.py``

- ``terrain_levels_vel``: adjust terrain difficulty based on how far the robot walked
- ``commands_vel``: stage command ranges over training steps (higher speeds, larger yaw rate)

G1 override: unitree_g1_*_env_cfg (robot-specific)
-------------------------------------------------

Path: ``src/mjlab/tasks/velocity/config/g1/env_cfgs.py``

This layer turns the base cfg into an actually trainable G1 cfg. Typical changes:

- set the entity: ``cfg.scene.entities = {"robot": get_g1_robot_cfg()}``
- add contact sensors (foot-ground / self-collision)
- fill placeholders (site_names / geom_names / torso_link, etc.)
- tune reward weights for stable and natural training on G1

Example (contact sensors + action scale + play mode):

.. code-block:: python

   # file: src/mjlab/tasks/velocity/config/g1/env_cfgs.py
   cfg.scene.entities = {"robot": get_g1_robot_cfg()}
   cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

   joint_pos_action = cfg.actions["joint_pos"]
   joint_pos_action.scale = G1_ACTION_SCALE

   if play:
       cfg.episode_length_s = int(1e9)
       cfg.observations["policy"].enable_corruption = False
       cfg.events.pop("push_robot", None)
       cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})

Task registration: register_mjlab_task
--------------------------------------

Path: ``src/mjlab/tasks/velocity/config/g1/__init__.py``

.. code-block:: python

   # file: src/mjlab/tasks/velocity/config/g1/__init__.py
   register_mjlab_task(
     task_id="Mjlab-Velocity-Rough-Unitree-G1",
     env_cfg=unitree_g1_rough_env_cfg(),
     play_env_cfg=unitree_g1_rough_env_cfg(play=True),
     rl_cfg=unitree_g1_ppo_runner_cfg(),
     runner_cls=VelocityOnPolicyRunner,
   )

From here you can infer the full CLI chain:

- ``uv run train <task_id>`` → ``src/mjlab/scripts/train.py``
- ``import mjlab.tasks`` triggers dynamic imports (see ``src/mjlab/tasks/__init__.py``) and fills the registry
- ``load_env_cfg(task_id)`` returns a deep copy of the env cfg (avoids mutating the global registry)

If you want to build a new locomotion task: minimal changes
-----------------------------------------------------------

Recommended order (from least invasive to most):

1. **Tune reward weights / add/remove reward terms**: edit how ``cfg.rewards`` is modified in ``config/g1/env_cfgs.py``.
2. **Change command distributions**: adjust ``cfg.commands["twist"].ranges`` (optionally add curriculum stages).
3. **Change observations**: add terms in the base cfg’s policy/critic groups, or edit terms in the G1 override.
4. **Add new event randomization**: add ``EventTermCfg`` under ``events`` (startup/reset/interval), and set ``domain_randomization=True`` if needed.

.. note::

   If you add rewards/obs that depend on contact sensors, add the sensor in the robot cfg first (``cfg.scene.sensors``),
   then read it from terms via ``env.scene[sensor_name]`` (see feet_* and self_collision_cost implementations).
