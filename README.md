# mjlab-homierl

[Screencast from 2026-03-25 19-53-04.webm](https://github.com/user-attachments/assets/a401ea12-95c9-4aec-a6dd-0c6c99aa47aa)

`mjlab-homierl` is an external `mjlab` task package reproducing the lower-body
locomotion RL portion of
[HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit](https://arxiv.org/abs/2502.13013)
on Unitree G1 (the robot used by the original OpenHomie release) and Unitree H1.
It follows the standard `anymal_c_velocity` style layout: the repository only
contains custom tasks, assets, and the HIM-PPO training stack, while the core
simulator/runtime comes from upstream `mjlab`.

Reward terms, weights, command sampling (1/3 squat, 1/2 walk, 1/6 stand every
4 s), the upper-body disturbance curriculum, and the HIM-PPO hyperparameters
follow the OpenHomie reference implementation (`HomieRL/legged_gym`).
Intentional deviations from OpenHomie are:

- Contact-based termination is replaced by contact penalties plus a
  torso-contact termination (G1), matching MuJoCo Warp's contact model.
- Domain randomization uses mjlab-native `dr.*` events (PD gains, link mass,
  payload, CoM offset, encoder bias, foot friction). OpenHomie's per-step
  torque injection has no mjlab equivalent and is approximated by PD-gain
  randomization and encoder bias.
- Per-joint action scales are derived from actuator effort/stiffness
  (`0.25 * effort / stiffness`), the mjlab convention, instead of a uniform
  0.25.
- The HIM estimator's next-step critic observation at termination steps is the
  post-reset observation (mjlab computes observations after resets).

## Structure

```text
src/mjlab_homierl/
  __init__.py              # entry-point task registration
  homie_env_cfg.py         # base HOMIE task config (OpenHomie rewards/commands)
  env_cfgs.py              # G1 / H1 / H1-with-hands configs
  rl_cfg.py                # HIM-PPO runner config
  mdp/                     # HOMIE commands, rewards, observations, actions
  rl/                      # HIM-PPO algorithm, runner, ONNX export
  robots/
    unitree_h1/            # H1 XML + constants (G1 comes from mjlab's asset zoo)
    robotiq_2f85/          # gripper XML assets
```

## Install

Clone the repository first:

```bash
git clone https://github.com/Nagi-ovo/mjlab-homierl.git
cd mjlab-homierl
```

For GPU training and playback:

```bash
uv sync --extra cu128
```

For CPU-only setups:

```bash
uv sync --extra cpu
```

For local docs builds:

```bash
uv sync --extra docs
```

This package depends on upstream `mjlab>=1.5.0,<1.6.0` and registers tasks
through the `mjlab.tasks` entry-point group.

## Registered Tasks

- `Mjlab-Homie-Unitree-G1` — trains with deployment-grade PD gains (the table
  HomieDeploy's real-robot low-level controller runs; see
  `robots/unitree_g1_deploy.py`) and the deployed pipeline's uniform 0.25
  action scale. Use this for sim2real. The effective per-joint
  stiffness/damping is embedded in the exported ONNX metadata, so deployment
  code can read the gains from the policy file. Waist defaults to `locked`:
  waist_roll/pitch are PD-held at the default pose and excluded from the
  upper-body disturbance, matching OpenHomie's 27-dof G1 (its URDF welds those
  two joints) and real-robot deployment, which software-holds them.

All G1 variants below share the base task's observation/action interface, so
checkpoints load interchangeably across them (and into the base task):

- `Mjlab-Homie-Unitree-G1-free_waist` — all three waist joints join the random
  upper-body disturbance; a strict superset of the original training
  distribution (the torso can be randomly pitched/rolled).
- `Mjlab-Homie-Unitree-G1-with_dex3` — Unitree Dex3 hands mounted as inertial
  attachments (~0.53 kg each) plus a randomized held-object payload.
- `Mjlab-Homie-Unitree-G1-with_inspire` — Inspire RH56 hands (RH56DFX spec
  mass, 0.54 kg each), same treatment. The base task randomizes a wrist
  payload covering these hand masses, so one checkpoint serves bare wrists
  and both hand models; these variants are primarily for play/eval with the
  real hand geometry.
- `Mjlab-Homie-Unitree-G1-mjlab_gains` — ablation variant with mjlab's
  first-principles actuator gains (armature × natural frequency); sim-only.
- `Mjlab-Homie-Unitree-G1-native` — frozen OpenHomie-parity preset. The diff
  between this cfg and the default task is the authoritative ledger of our
  deliberate deviations (torso payload DR narrowed to (-1, +5) kg, wrist
  payload DR widened to (0, 1.5) kg, hip/knee ground-contact penalty added —
  OpenHomie's `penalize_contacts_on` is dead code). Reference baseline; never
  iterated on.

HOMIE+ (interface fork — checkpoints do NOT interchange with the tasks above):

- `Mjlab-Homie-Unitree-G1-plus` — the deployment fork; three deliberate
  extensions over OpenHomie parity, each motivated by hardware findings:
  1. **Commanded torso pitch**: a 5th command dim carries a `waist_pitch`
     joint-angle target (rad, + = lean forward, sampled in walk/squat modes
     up to 0.45 rad). The joint is command-driven (policy-free, slew-limited
     at 1 rad/s); the policy keeps the 12-dim leg interface and learns to
     balance the lean — the missing DoF for pick-from-floor work. One-step
     observation grows 80 → 81 (actor input 486): separate training lineage
     (`g1_homie_plus_himppo`), checkpoints do not interchange with the base
     task. Commanding pitch = 0 reproduces plain HOMIE behavior.
  2. **In-place locomotion sampling**: 1/3 of walk-mode resamples set
     `vx = 0` and keep the sampled `vy`/`wz` (dominant axis clamped ≥ 0.3).
     The faithful sampler leaves pure strafe and pure rotation at measure
     zero — probed policies stand at 100% double support through such
     commands.
  3. **Foot contact-compliance DR**: per-env `geom_solref` on the foot
     spheres, near-rigid to soft foam (after arXiv:2504.13619) — fixes the
     standing sway observed on EVA gym mats.
  4. **Squat rework (v4)**: the height command is a slewed setpoint (per-env
     rate DR 0.25–0.75 m/s) instead of OpenHomie's instantaneous step — a
     position-type command step creates an unreachable error window whose
     tracking gradient rewards ballistic descent (the v3 policy crash-squatted
     onto its knees). Plus: hip/knee ground-contact penalty −5.0 (breaks the
     kneeling economics), the `stand_still` no-stepping shaping extended to
     all commanded heights (squat-mode shuffling was free), and the torso
     pitch law keyed on moving/stationary (stand + lean = the
     reach-over-a-table teleop pose).
  The exported ONNX metadata declares the 5-dim command
  (`one_step_obs_layout`, `pitch_command_joint`, `pitch_command_ranges`) so
  downstream plugins bootstrap without hardcoding. Play works the same as the
  base task:
  `uv run play Mjlab-Homie-Unitree-G1-plus --checkpoint-file ... --viewer viser`.

- `Mjlab-Homie-Unitree-H1` — trains with Unitree's official RL-stack PD gains
  (unitree_rl_gym `h1_config.py`; see `robots/unitree_h1_deploy.py`) and the
  uniform 0.25 action scale.
- `Mjlab-Homie-Unitree-H1-mjlab_gains` — H1 ablation variant with
  first-principles gains; sim-only.
- `Mjlab-Homie-Unitree-H1-with_hands` — deploy gains, with Robotiq 2F85
  grippers mounted.

## Usage

Pretrained locomotion checkpoints are available at
[Hugging Face](https://huggingface.co/Nagi-ovo/HOMIERL-loco) if you want to skip
training and go straight to playback.

List available environments:

```bash
uv run list-envs
```

Train:

```bash
uv run train Mjlab-Homie-Unitree-G1 --env.scene.num-envs 4096
uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 4096
```

The runner config sets `upload_model=False`: metrics are logged to W&B (when
the `wandb` logger is selected) but checkpoints and ONNX exports stay local.
Use `--agent.logger tensorboard` to skip W&B entirely.

Note: the HIM-PPO algorithm is single-GPU; the upstream `--gpu-ids` multi-GPU
path is not supported.

Play:

```bash
uv run play Mjlab-Homie-Unitree-G1 --checkpoint-file /path/to/model.pt --viewer viser
```

More explicit playback examples:

```bash
uv run play Mjlab-Homie-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --num-envs 30 \
  --viewer viser \
  --device cuda:0
```

Sanity-check the MDP before training:

```bash
uv run play Mjlab-Homie-Unitree-G1 --agent zero
uv run play Mjlab-Homie-Unitree-G1 --agent random
```

Play holds the upper body at its default pose by default (isolating the
lower-body gait). To preview deployment-like upper-body disturbances instead:

```bash
HOMIE_PLAY_UPPER_RATIO=1.0 uv run play Mjlab-Homie-Unitree-G1 --checkpoint-file ...
```

## Real-robot deployment (G1)

`src/mjlab_homierl/scripts/deploy_g1_homie.py` runs an exported policy ONNX on
the real robot over DDS, reading every convention (joint order, PD gains,
obs layout, command ranges, optional torso-pitch channel) from the ONNX
metadata. The real path needs only `numpy + onnxruntime + unitree_sdk2py` —
no mjlab/torch on the robot-side machine. One-time environment setup (builds
cyclonedds 0.10.2 and works around three upstream packaging defects — see the
script header):

```bash
bash scripts/setup_deploy_env.sh
```

Then (robot in debug/low-level mode, harnessed):

```bash
.venv-deploy/bin/python src/mjlab_homierl/scripts/deploy_g1_homie.py \
  --onnx <run_dir>/<run>.onnx --net <iface>
```

START moves to the default pose, A starts the policy, sticks drive vx/vy/wz,
dpad up/down slews the height command, X/B slews torso pitch (HOMIE+ models),
SELECT exits to damping. `--sim` (in the training venv:
`uv run --extra deploy ... --sim`) validates the deploy-side observation
builder bit-for-bit against the mjlab plant before any hardware session.

## Interactive sim teleop (keyboard, classic MuJoCo)

```bash
uv run python -m mjlab_homierl.scripts.teleop_sim_g1 --onnx <run>.onnx
```

Real-time keyboard teleop in a plain CPU MuJoCo window, driving the policy
through the same `runtime.py` pipeline as the real robot and the BiGym
plugin — physics independent of the training engine (sim2sim). WASD = vx/vy,
Q/E = yaw, arrows = height, R/F = torso pitch (HOMIE+), Space = stop,
Backspace = reset; Ctrl+drag shoves the robot. `--smoke` runs a 5 s headless
self-test. Note for anyone building a classic-MuJoCo harness from the raw
asset-zoo spec: it compiles with NO actuators, NO keyframe, and zero
armature (mjlab injects all three at Entity build time), and the default
Euler integrator is unstable at knee kp = 300 — this script re-injects all
four (see `build_model`).

## Notes

- The repository no longer vendors the `mjlab` framework itself.
- HIM-PPO remains package-local under `mjlab_homierl.rl.himppo`. Left/right
  mirror maps for its symmetry augmentation are derived from joint names, so
  any humanoid with `left_*`/`right_*` joint naming works.
- `Mjlab-Homie-Unitree-H1-with_hands` mounts the 2F85 grippers with hand
  collisions disabled by default. For HOMIE this keeps the hands as inertial /
  disturbance attachments instead of contact-rich manipulation tools.
- `uv run play ...` uses a HOMIE-specific actor-only inference path. The play
  env strips critic observations, rewards, and curriculum to reduce playback
  overhead.
- The custom inference helper is exposed as:

```bash
uv run infer-homie-lowerbody --help
```

## Development

Run the package regression tests:

```bash
uv run pytest tests -q
```

Build the docs locally:

```bash
uv run --extra docs sphinx-build docs docs/_build
```
