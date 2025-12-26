![Project banner](docs/static/mjlab-banner.jpg)

# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
  <a href="https://mujocolab.github.io/mjlab/nightly/"><img alt="benchmarks" src="https://img.shields.io/badge/nightly-blue" /></a>
</p>

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

> ⚠️ **BETA PREVIEW** mjlab is in active development. Expect **breaking
> changes** and **missing features** during the beta phase. There is **no stable
> release yet**. The PyPI package is only a snapshot — for the latest fixes and
> improvements, install from source or Git.

---

## Installation

**From source (recommended during beta):**

```bash
git clone https://github.com/Nagi-ovo/mjlab-homierl.git
cd mjlab-homierl
```

## Training HOMIE

### Train (w/o hands)

```bash
uv run train Mjlab-Homie-Unitree-H1 --env.scene.num-envs 4096
```

### Train (with robotiq hands)

You can also download the pre-trained w/ hands policy from [huggingface](https://huggingface.co/Nagi-ovo/HOMIERL-loco) to skip to play :)

```bash
uv run train Mjlab-Homie-Unitree-H1-with_hands --env.scene.num-envs 4096
```

### Play (w/o hands)

```bash
uv run play Mjlab-Homie-Unitree-H1 --checkpoint_file MODEL_PATH --num-envs 30 --viewer viser
```

### Play (with robotiq hands)

```bash
uv run play Mjlab-Homie-Unitree-H1-with_hands --checkpoint_file MODEL_PATH --num-envs 30 --viewer viser
```

---

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP **before** training.

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions.
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions.
```

> [!NOTE]
> When running motion-tracking tasks, add
> `--registry-name your-org/motions/motion-name` to the command.

---

## Documentation

- **[Installation Guide](docs/installation_guide.md)**
- **[Why mjlab?](docs/motivation.md)**
- **[Migration Guide](docs/migration_guide.md)**
- **[FAQ & Troubleshooting](docs/faq.md)**

---

## Development

Run tests:

```bash
make test          # Run all tests
make test-fast     # Skip slow integration tests
```

Format code:

```bash
uvx pre-commit install
make format
```

---

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

---

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
