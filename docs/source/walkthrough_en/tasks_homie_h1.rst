.. _walkthrough-en-task-homie-h1:

Example 3: Homie — Mixed Motion and Disturbances (Unitree G1 / H1)
====================================================================

Homie mixes **velocity tracking**, **squatting (height control)**,
**standing**, and **upper-body random disturbances**. The implementation
follows the OpenHomie reference (``HomieRL/legged_gym``): reward set, command
sampling, and curriculum.

Main task ids:

- ``Mjlab-Homie-Unitree-G1``: the OpenHomie robot (29 dof, 12 lower-body
  actions). The waist defaults to ``locked``: waist_roll/pitch are PD-held at
  the default pose and excluded from the upper-body disturbance, matching
  OpenHomie's 27-dof URDF (which welds those two joints) and real-robot
  deployment.
- ``Mjlab-Homie-Unitree-H1``: the H1 port (19 dof, 10 lower-body actions).
- ``Mjlab-Homie-Unitree-H1-with_hands``: H1 with Robotiq 2F85 grippers
  (policy-free random gripper motion and randomized hand payload).

G1 variants (identical observation/action interface; checkpoints load
interchangeably):

- ``Mjlab-Homie-Unitree-G1-free_waist``: all three waist joints join the
  upper-body disturbance (a strict superset of the original distribution).
- ``Mjlab-Homie-Unitree-G1-with_dex3`` / ``-with_inspire``: Unitree Dex3 /
  Inspire RH56 hand models mounted as inertial attachments with a randomized
  held-object payload. The base task's wrist-payload randomization covers
  these hand masses, so one checkpoint serves bare wrists and both hand
  models; these variants are primarily for play/eval with real hand geometry.
- ``Mjlab-Homie-Unitree-G1-mjlab_gains``: ablation with mjlab's
  first-principles actuator gains; sim-only.

HOMIE+ (interface fork — checkpoints do NOT interchange with the tasks above):

- ``Mjlab-Homie-Unitree-G1-plus``: the deployment fork — three deliberate,
  hardware-motivated extensions over parity: (1) commanded torso pitch (5th
  command dim, ``waist_pitch`` joint-angle target up to 0.45 rad, command
  driven / policy-free, slew-limited 1 rad/s; one-step obs 80 → 81, actor
  486, separate lineage ``g1_homie_plus_himppo``, pitch = 0 reproduces plain
  HOMIE); (2) in-place locomotion sampling (1/3 of walk resamples: vx = 0,
  sampled vy/wz kept with the dominant axis clamped ≥ 0.3 — the faithful
  sampler leaves pure strafe/turn at measure zero and probed policies stand
  through such commands); (3) per-env foot contact-compliance DR
  (``geom_solref`` near-rigid to soft foam, after arXiv:2504.13619 — fixes
  standing sway on EVA gym mats). The exported ONNX metadata declares the
  5-dim command contract so downstream plugins bootstrap without
  hardcoding.

The core idea: **reduce the policy action space to the lower body, and treat
the upper body (and optional grippers) as smooth, time-varying disturbances.**

Task registration
-----------------

Path: ``src/mjlab_homierl/__init__.py``

The task ids above are registered via ``mjlab.tasks.registry.register_mjlab_task``.
The play env is the same config with ``play=True`` (critic observations,
rewards, and curriculum stripped; upper-body motion at full range).

Three-mode command sampling
---------------------------

Path: ``src/mjlab_homierl/mdp/velocity_command.py``

Commands resample every 4 s. Each environment draws one of three mutually
exclusive modes (the OpenHomie scheme):

- **squat** (p = 1/3): zero twist, random relative-height target;
- **walk** (p = 1/2): random twist (x ∈ [-0.8, 1.2], y ∈ [-0.5, 0.5],
  yaw ∈ [-0.8, 0.8]), standing-height target;
- **stand** (p = 1/6): zero twist, standing-height target.

The ``twist`` command samples the mode and exposes it via its ``mode``
attribute; the ``height`` command (``RelativeHeightCommand``, pelvis height
above the lowest foot site) couples to it. Both commands must share the same
resampling interval, and ``twist`` must precede ``height`` in the commands
dict.

Height ranges scale with the robot: G1 stands at 0.78 m and squats down to
0.28 m; H1 stands at 0.98 m and squats to 0.4 m. A set of "standing only"
reward terms (hip/ankle deviation, feet_parallel, stand_still, ...) are gated
on the commanded height being near the standing height.

Reward set
----------

Paths: ``src/mjlab_homierl/homie_env_cfg.py`` (weights),
``src/mjlab_homierl/mdp/rewards.py`` (implementations)

Terms and weights follow OpenHomie's G1 config: split x/y velocity tracking
(1.5 / 1.0), yaw tracking (2.0, σ²=0.25), height tracking ``exp(-4|err|)``
(2.0), hip/ankle deviation (-0.2 / -0.5), knee-driven squatting (-0.75), the
full joint regularization suite (torques, power, velocity, acceleration, soft
limits), and the feet terms (air time, no-fly, clearance, slip, stumble,
contact forces, contact momentum, sole parallelism, feet parallel, lateral
distances).

Intentional deviations from OpenHomie:

- IsaacGym's torso-contact termination is replaced by contact penalties plus a
  torso-contact termination on G1, with additional self-collision and
  hip/knee ground-contact penalties.
- The fall-over termination threshold matches the original
  (``asin(0.8) ≈ 53°``).

Upper-body disturbance and curriculum
-------------------------------------

Paths: ``src/mjlab_homierl/mdp/actions.py``, ``mdp/curriculums.py``

``UpperBodyPoseAction`` contributes zero policy dimensions. Every 1 s (a
global interval event), upper-body goal poses are resampled for all
environments: amplitudes come from a truncated-exponential transform of the
curriculum ratio (heavily biased toward small motions early on), the direction
is a fair coin between each joint's lower/upper hard limit (so amplitudes are
proportional to joint range), and the target is reached by linear
interpolation over one interval.

Curriculum advancement matches OpenHomie: the check runs only when
``common_step_counter`` is a multiple of the max episode length; if the
episode-average raw forward-velocity tracking reward is ≥ 0.8, the global
ratio increases by 0.05.

Domain randomization
--------------------

All randomization uses mjlab-native ``dr.*`` events: PD gains ×[0.9, 1.1]
(per reset), link masses ×[0.8, 1.2], torso payload +[-1, +5] kg (see the env cfg
comment for the survey behind this deviation), CoM offset,
encoder bias, foot friction, a global horizontal push every 4 s
(Δv ≤ 0.5 m/s), and randomized joint poses / root velocities at reset.
OpenHomie's per-step torque injection has no mjlab equivalent and is
approximated by PD-gain randomization and encoder bias.

HIM-PPO
-------

Path: ``src/mjlab_homierl/rl/himppo/``

Hyperparameters and network sizes (actor/critic hidden dims 512-256-256,
estimator latent 32, prototype 64, sinkhorn contrastive loss) follow
OpenHomie. The left/right mirror maps for symmetry augmentation are derived
from joint names (``left_*``/``right_*`` pairing; joints whose names contain
``yaw``/``roll`` flip sign), so G1 and H1 share one implementation.

Termination steps feed the estimator the pre-reset critic observation via a
recorder term, matching OpenHomie's runner behavior.
