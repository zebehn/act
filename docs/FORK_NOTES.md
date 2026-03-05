# Fork Notes (zebehn)

This fork extends the original ACT repository with Apple Silicon and experimentation workflow improvements.

## What Changed
- Added device-agnostic execution with explicit `--device` selection (`auto`, `mps`, `cuda`, `cpu`).
- Prioritized MPS on macOS for Apple Silicon workflows.
- Removed hard CUDA assumptions in training/eval code paths.
- Added checkpoint resume support with `--resume_ckpt` (including `auto` fallback to latest valid checkpoint).
- Improved small-dataset robustness for smoke tests and local validation.

## Why This Fork Exists
- Enable end-to-end ACT training and evaluation on MacBook Pro (Apple Silicon) without CUDA.
- Provide reproducible migration and experiment records.

## Reports
- [MPS Migration Development Log](reports/MPS_MIGRATION_DEVLOG.md)
- [ACT MPS Experiment Report](reports/EXPERIMENT_REPORT_MPS_ACT.md)

## Quick Start (MPS)
```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
conda run -n aloha python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps
```
