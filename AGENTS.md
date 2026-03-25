# Agent Guide (HOMIE) — `src/mjlab/**`

## Scope

- This guide applies to everything under `src/mjlab/**`.
- Default to editing only `src/mjlab/**` unless the user explicitly asks otherwise.
- Prefer reusing existing runnable entrypoints under `src/mjlab/scripts/` instead of writing new glue code.

## Project Shape (what lives where)

- `mjlab/envs/`: MuJoCo RL env core (`ManagerBasedRlEnv`, `ManagerBasedRlEnvCfg`) and shared MDP utilities.
- `mjlab/tasks/`: Task registry + task-specific configs/MDP terms.
- `mjlab/rl/`: RSL-RL wrappers, runner configs, exporters.
- `mjlab/scripts/`: CLI entrypoints (`train.py`, `play.py`, `list_envs.py`, `demo.py`).
- `mjlab/viewer/`: Viewer backends (`NativeMujocoViewer`, `ViserPlayViewer`).

## Task Registry (how tasks become runnable)

- Tasks are registered via `mjlab.tasks.registry.register_mjlab_task(...)`.
- Registration happens at import-time, typically from `mjlab/tasks/**/config/**/__init__.py`.
- `import mjlab.tasks` recursively imports task packages via `mjlab.utils.lab_api.tasks.importer.import_packages`,
  with a blacklist of package-name substrings: `["utils", ".mdp"]`.
  - Put registration code in `config/` packages (not in `mdp/`).
  - Keep import-time side effects limited to registration (avoid starting sims, allocating GPU memory, etc.).

## HOMIE / HOMIR task (this repo focus)

### Registered task IDs

- `Mjlab-Homie-Unitree-H1`
- `Mjlab-Homie-Unitree-H1-with_hands`

### Where to change behavior

- Base env config factory: `mjlab/tasks/homie/homie_env_cfg.py`
- Robot-specific overrides: `mjlab/tasks/homie/config/h1/env_cfgs.py`
- PPO runner config: `mjlab/tasks/homie/config/h1/rl_cfg.py`
- MDP terms (obs/reward/termination/curriculum): `mjlab/tasks/homie/mdp/`
- Custom runner: `mjlab/tasks/homie/rl/runner.py`
  - When logging to W&B, it exports an ONNX policy and attaches metadata on each save.

### Shape-sensitive code

- If you change HOMIE observation/action shapes, also update any dependent utilities.
  - Example: `mjlab/scripts/infer_h1_with_hands_lowerbody_policy.py` currently expects `obs_dim=97`, `act_dim=10`
    for the shipped `homie_rl.pt`.

## Common CLI workflows (from `src/mjlab/scripts/`)

- Use `uv run` from repo root for all CLI workflows.
- If `uv run` fails due to cache permissions, set `UV_CACHE_DIR=.uv-cache` (repo-local) or pass `--cache-dir .uv-cache`.
- List registered tasks: `uv run python -m mjlab.scripts.list_envs`
  - Filter: `uv run python -m mjlab.scripts.list_envs --keyword homie`
- Train (RSL-RL): `uv run python -m mjlab.scripts.train <TaskID> [flags...]`
  - Multi-GPU is handled internally via `torchrunx` when `CUDA_VISIBLE_DEVICES` exposes >1 GPU.
- Play: `uv run python -m mjlab.scripts.play <TaskID> [flags...]`
  - Trained policy: use `--wandb-run-path ...` or `--checkpoint-file ...`
  - Debug policies: `--agent zero` or `--agent random`
- Tracking demo (downloads a checkpoint + motion): `uv run python -m mjlab.scripts.demo`

### Tracking tasks (motion files)

- Some tracking tasks require a motion file.
  - Train: either set `--registry-name ...` (download artifact) or override `--env.commands.motion.motion-file ...`.
  - Play: use `--motion-file ...` for local, or `--wandb-run-path ...` so the motion artifact can be resolved.

## Rendering / Headless notes

- `train.py` sets `MUJOCO_GL=egl` and (multi-GPU) sets `MUJOCO_EGL_DEVICE_ID` from `LOCAL_RANK`.
- `play.py` uses `viewer="auto"`:
  - picks `native` when a display is available (`DISPLAY`/`WAYLAND_DISPLAY`)
  - otherwise defaults to `viser` (headless-friendly)
- For offscreen video, `--video` uses `render_mode="rgb_array"` internally.

## Code style (match existing code)

- Keep formatting consistent with this tree: 2-space indentation, explicit type hints, dataclasses for configs.
- Prefer small, local changes; don’t refactor unrelated modules.
- When adding a new task:
  1. Add `mjlab/tasks/<task>/config/<robot>/__init__.py` that calls `register_mjlab_task`.
  2. Provide `env_cfg(...)`, `play_env_cfg(...)`, and `rl_cfg(...)` factories.
  3. Verify it shows up in `mjlab.scripts.list_envs`.
