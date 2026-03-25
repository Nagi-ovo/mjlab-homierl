# mjlab-homierl

`mjlab-homierl` is an external `mjlab` task package for reproducing the lower-body
locomotion part of HOMIE on Unitree H1. It follows the standard `anymal_c_velocity`
style layout: the repository only contains custom tasks, assets, and the HIMPPO
training stack, while the core simulator/runtime comes from upstream `mjlab`.

This repository targets the lower-body locomotion RL portion of
[HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit](https://arxiv.org/abs/2502.13013).

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

This package depends on upstream `mjlab>=1.2.0,<1.3.0` and registers tasks through
the `mjlab.tasks` entry-point group.

## Registered Tasks

- `Mjlab-Homie-Unitree-H1`
- `Mjlab-Homie-Unitree-H1-with_hands`

## Usage

Pretrained locomotion checkpoints are available at
[Hugging Face](https://huggingface.co/Nagi-ovo/HOMIERL-loco) if you want to skip
training and go straight to playback.

List available environments:

```bash
uv run list_envs
```

Train:

```bash
uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 4096
uv run train Mjlab-Homie-Unitree-H1-with_hands --env.scene.num-envs 4096
```

Multi-GPU training is also supported by the upstream CLI:

```bash
uv run train Mjlab-Homie-Unitree-H1 \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096

uv run train Mjlab-Homie-Unitree-H1-with_hands \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096
```

Play:

```bash
uv run play Mjlab-Homie-Unitree-H1 --checkpoint-file /path/to/model.pt --viewer viser
uv run play Mjlab-Homie-Unitree-H1-with_hands --checkpoint-file /path/to/model.pt --viewer viser
```

More explicit playback examples:

```bash
uv run play Mjlab-Homie-Unitree-H1 \
  --checkpoint-file /path/to/model.pt \
  --num-envs 30 \
  --viewer viser

uv run play Mjlab-Homie-Unitree-H1-with_hands \
  --checkpoint-file /path/to/model.pt \
  --num-envs 30 \
  --viewer viser \
  --device cuda:0
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

## Development

Run the package regression tests:

```bash
uv run pytest tests -q
```

Build the docs locally:

```bash
uv run --extra docs sphinx-build docs docs/_build
```
