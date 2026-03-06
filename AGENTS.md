# Repository Guidelines

## Project Structure & Module Organization
This repository implements ACT (Action Chunking with Transformers) for ALOHA simulation and real-robot transfer.
- Top-level training/eval scripts: `imitate_episodes.py`, `record_sim_episodes.py`, `visualize_episodes.py`.
- Environment and task logic: `sim_env.py`, `ee_sim_env.py`, `scripted_policy.py`, `constants.py`.
- Shared helpers: `utils.py`, `policy.py`.
- Model code: `detr/` (installable package with ACT/DETR model components).
- Simulation assets: `assets/` (MuJoCo XML and meshes).

Keep new modules near related entrypoints, and avoid mixing model code into environment scripts.

## Build, Test, and Development Commands
Use the Python/Conda flow documented in `README.md`.
- `conda env create -f conda_env.yaml` (or create `python=3.8.10` env manually): set up dependencies.
- `conda activate aloha`: activate the runtime environment.
- `cd detr && pip install -e . && cd ..`: install local model package in editable mode.
- `python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <dir> --num_episodes 50`: generate scripted demos.
- `python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <dir> --policy_class ACT ...`: train ACT.
- Add `--eval` to `imitate_episodes.py` command: run evaluation with best checkpoint.
- `python3 visualize_episodes.py --dataset_dir <dir> --episode_idx 0`: inspect collected episodes.

## Coding Style & Naming Conventions
- Follow Python conventions: 4-space indentation, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Keep scripts CLI-driven with `argparse` flags matching existing names (for example `--task_name`, `--ckpt_dir`).
- Prefer small utility functions in `utils.py` over duplicated logic across scripts.
- Keep imports explicit and grouped (stdlib, third-party, local).

## Testing Guidelines
There is no dedicated unit-test suite in this repo today. Validate changes with:
- Script-level smoke checks (data collection, training startup, eval startup).
- Behavioral verification via rollout metrics and rendered videos in checkpoint/output directories.

When adding tests, place them in a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Existing history uses short, imperative commit subjects (for example `Update README.md`, `remove clip ceiling`).
- Write focused commits with a single clear intent.
- Use concise subjects (<72 chars), present tense, imperative mood.
- In PRs, include: summary, affected tasks/scripts, exact reproduction commands, and before/after results.
- For policy/environment changes, attach rollout success metrics and sample video paths.

## Data & Configuration Notes
- Do not commit large datasets, checkpoints, or generated videos.
- Pass dataset/checkpoint locations via CLI args instead of hardcoding local absolute paths.
