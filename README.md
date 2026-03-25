# mjlab-homierl

`mjlab-homierl` is an external `mjlab` task package for reproducing the lower-body
locomotion part of HOMIE on Unitree H1. It follows the standard `anymal_c_velocity`
style layout: the repository only contains custom tasks, assets, and the HIMPPO
training stack, while the core simulator/runtime comes from upstream `mjlab`.

## Structure

```text
src/mjlab_homierl/
  __init__.py              # entry-point task registration
  homie_env_cfg.py         # base HOMIE task config
  env_cfgs.py              # H1 / H1-with-hands overrides
  rl_cfg.py                # HIMPPO runner config
  mdp/                     # HOMIE-specific commands, rewards, observations
  rl/                      # custom HIMPPO algorithm, runner, ONNX export
  robots/
    unitree_h1/            # H1 XML + constants
    robotiq_2f85/          # gripper XML assets
```

## Install

```bash
uv sync --extra cu128
```

For CPU-only setups:

```bash
uv sync --extra cpu
```

This package depends on upstream `mjlab>=1.2.0,<1.3.0` and registers tasks through
the `mjlab.tasks` entry-point group.

## Registered Tasks

- `Mjlab-Homie-Unitree-H1`
- `Mjlab-Homie-Unitree-H1-with_hands`

## Usage

List available environments:

```bash
uv run list_envs
```

Train:

```bash
uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 4096
uv run train Mjlab-Homie-Unitree-H1-with_hands --env.scene.num-envs 4096
```

Play:

```bash
uv run play Mjlab-Homie-Unitree-H1 --checkpoint-file /path/to/model.pt --viewer viser
uv run play Mjlab-Homie-Unitree-H1-with_hands --checkpoint-file /path/to/model.pt --viewer viser
```

Sanity-check the MDP before training:

```bash
uv run play Mjlab-Homie-Unitree-H1 --agent zero
uv run play Mjlab-Homie-Unitree-H1 --agent random
```

## Notes

- The repository no longer vendors the `mjlab` framework itself.
- HIMPPO remains package-local under `mjlab_homierl.rl.himppo`.
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
