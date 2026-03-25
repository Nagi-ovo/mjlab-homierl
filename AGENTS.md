# Agent Guide — `mjlab-homierl`

## Scope

- This guide applies to the whole `mjlab-homierl` repository.
- The package is an external `mjlab` task extension. The source of truth is under
  `src/mjlab_homierl/**`, not `src/mjlab/**`.
- Default to small, local edits. Do not vendor or modify upstream `mjlab` unless
  the user explicitly asks for it.

## Project Shape

- `src/mjlab_homierl/__init__.py`
  - Registers task IDs at import time.
- `src/mjlab_homierl/homie_env_cfg.py`
  - Base HOMIE environment config factory.
- `src/mjlab_homierl/env_cfgs.py`
  - Unitree H1 and H1-with-hands task overrides.
  - This is the main place to change play-mode behavior, sensors, rewards, and
    robot/task wiring.
- `src/mjlab_homierl/rl_cfg.py`
  - HIMPPO runner config.
- `src/mjlab_homierl/mdp/`
  - HOMIE-specific observations, rewards, terminations, curriculum, and
    command sampling.
- `src/mjlab_homierl/rl/`
  - Custom runner, HIMPPO implementation, ONNX export.
- `src/mjlab_homierl/robots/`
  - H1 and Robotiq assets/constants.
- `src/mjlab_homierl/scripts/`
  - Standalone helpers such as the lower-body inference script.
- `tests/`
  - Package-local regression tests only.

## Registration And Entry Points

- This repo does not ship the `mjlab` framework itself.
- Task discovery happens through the `mjlab.tasks` entry-point declared in
  `pyproject.toml`, which points to `mjlab_homierl`.
- Importing `mjlab_homierl` runs task registration in
  `src/mjlab_homierl/__init__.py`.
- Registered task IDs:
  - `Mjlab-Homie-Unitree-H1`
  - `Mjlab-Homie-Unitree-H1-with_hands`

## Where To Change Behavior

- Change task structure or default task parameters:
  - `src/mjlab_homierl/env_cfgs.py`
- Change shared/base HOMIE config:
  - `src/mjlab_homierl/homie_env_cfg.py`
- Change HIMPPO algorithm or inference behavior:
  - `src/mjlab_homierl/rl/himppo/`
  - `src/mjlab_homierl/rl/runner.py`
- Change robot assets, mounting, or collision setup:
  - `src/mjlab_homierl/robots/unitree_h1/h1_constants.py`
  - `src/mjlab_homierl/robots/robotiq_2f85/`
- Change exported policy / ONNX behavior:
  - `src/mjlab_homierl/rl/exporter.py`

## Important Runtime Notes

- Use `uv` for install, tests, training, and play.
- Prefer GPU on Linux/NVIDIA:
  - `uv sync --extra cu128`
- CPU-only fallback:
  - `uv sync --extra cpu`
- Common commands:
  - `uv run list_envs`
  - `uv run train <TaskID> ...`
  - `uv run play <TaskID> --checkpoint-file /path/to/model.pt ...`
  - `uv run pytest tests/test_registration.py tests/test_env_cfgs.py tests/test_rl_cfg.py tests/test_h1_assets.py -q`

## HOMIE-Specific Notes

- `with_hands` keeps the hands for mass/kinematics, but default HOMIE task
  config disables hand collision geoms because locomotion does not need hand
  contact and leaving them enabled creates heavy collision overhead.
- Play mode is intentionally lighter than training:
  - rewards are stripped
  - curriculum is stripped
  - actor corruption is disabled
  - HOMIE uses a custom actor-only inference path in
    `src/mjlab_homierl/rl/runner.py`, so play envs do not need critic
    observations
- If you change play-mode observation layout, also verify
  `HomieHimOnPolicyRunner` still supports inference-only initialization.

## Shape-Sensitive Code

- HOMIE HIMPPO assumes one actor group and one critic group during training.
  The inference-only play path assumes exactly one actor group.
- If you change HOMIE observation packing or history length, review:
  - `src/mjlab_homierl/rl/himppo/actor_critic.py`
  - `src/mjlab_homierl/rl/himppo/algorithm.py`
  - `src/mjlab_homierl/rl/runner.py`
- The standalone helper
  `src/mjlab_homierl/scripts/infer_h1_with_hands_lowerbody_policy.py`
  hardcodes a separate checkpoint format (`obs_dim=97`, `act_dim=10`). If the
  target checkpoint format changes, update that script explicitly.

## Testing And Validation

- For config/registration edits, run:
  - `uv run pytest tests/test_registration.py tests/test_env_cfgs.py tests/test_rl_cfg.py -q`
- For robot/collision edits, also run:
  - `uv run pytest tests/test_h1_assets.py -q`
- For play-performance work, separate env cost from viewer cost:
  - headless `ManagerBasedRlEnv` / `RslRlVecEnvWrapper` stepping tells you if the
    environment is slow
  - `viser` `CAPPED` can still happen occasionally from browser/render jitter
    even if headless RT is above `1.0x`

## Style

- Match the repo's existing style:
  - 2-space indentation
  - explicit type hints
  - dataclasses for configs
- Prefer targeted edits over broad refactors.
- Keep new guidance aligned with the current external-package structure. If the
  package layout changes again, update this file in the same change.
