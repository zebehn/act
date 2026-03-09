# Repository Guidelines

## Project Structure & Module Organization
This repository implements ACT-based imitation learning for the `transfer_cube` and `insertion` tasks, with an in-progress DPO-based post-training pipeline to improve ACT checkpoints. Core entry points live at the repository root: `imitate_episodes.py`, `record_sim_episodes.py`, `visualize_episodes.py`, `sim_env.py`, and helpers such as `utils.py`, `policy.py`, and `constants.py`. Model code is vendored under `detr/`. Automation, rollout evaluation, and post-training utilities live in `scripts/`. Simulator XML, meshes, and scene assets are in `assets/`. Experiment notes and performance reports are in `docs/` and `docs/reports/`. Local datasets and checkpoints typically live under `data/` and `logs/`, both of which are git-ignored.

## Build, Test, and Development Commands
- `conda env create -f conda_env.yaml && conda activate aloha` — create the baseline environment.
- `python -m pip install -e detr` — install the local ACT/DETR package in editable mode.
- `ACT_DATA_DIR="$PWD/data" PYTHONPATH="$PWD/detr:$PWD" python imitate_episodes.py ...` — train or evaluate ACT; add `--eval` for rollouts.
- `PYTHONPATH="$PWD/detr:$PWD" python scripts/regression_smoke_tests.py` — run core ACT imitation-learning smoke tests.
- `PYTHONPATH="$PWD/detr:$PWD" python scripts/posttrain_regression_smoke_tests.py` — validate the DPO/PPO post-training path.
- `bash scripts/posttrain_cli_smoke.sh` — run an end-to-end post-training CLI smoke check.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and keep changes consistent with the current script-oriented layout. Prefer `snake_case` for functions, variables, and filenames; use `UPPER_SNAKE_CASE` for constants and environment-variable defaults. Group imports as standard library, third-party, then local modules. Follow the existing style of explicit `argparse` flags, small helper functions, and direct control flow. No formatter is enforced in this repo, so match surrounding code closely.

## Testing Guidelines
There is no separate `tests/` package; regression coverage lives in `scripts/*smoke*.py` and shell wrappers. Add focused `test_*` helpers near the affected smoke script, and prefer temporary directories plus synthetic HDF5 fixtures over full datasets. For imitation-learning changes, start with `scripts/regression_smoke_tests.py`. For DPO or other post-training changes, run `scripts/posttrain_regression_smoke_tests.py`

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add ACT post-training RL pilot workflow`. Keep each commit scoped to one logical change. PRs should state whether they affect ACT training, rollout evaluation, or DPO/post-training, list the exact commands run, and note the device used (`mps`, `cuda`, or `cpu`). Link the relevant report when behavior changes, and include key metrics or screenshots. Never commit local datasets, checkpoints, videos, or generated outputs from `data/`, `logs/`, `outputs/`, or `wandb/`.
