#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/act_posttrain_local_smoke_$(date +%Y%m%dT%H%M%S)}"
mkdir -p "${OUT_ROOT}"

ACT_DATA_DIR="${ROOT_DIR}/data" PYTHONPATH="${ROOT_DIR}/detr:${ROOT_DIR}" MPLCONFIGDIR=/tmp/matplotlib \
python "${ROOT_DIR}/scripts/collect_posttrain_rollouts.py" \
  --task_name sim_transfer_cube_scripted \
  --source_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --out_dir "${OUT_ROOT}/rollouts" \
  --num_seed_groups 2 \
  --rollouts_per_seed 2 \
  --device cpu \
  --temporal_agg \
  --max_timesteps 16

python "${ROOT_DIR}/scripts/build_posttrain_preferences.py" \
  --rollout_dir "${OUT_ROOT}/rollouts" \
  --out_file "${OUT_ROOT}/preferences.jsonl" \
  --window_length 32

python "${ROOT_DIR}/scripts/train_posttrain_dpo.py" \
  --task_name sim_transfer_cube_scripted \
  --pref_file "${OUT_ROOT}/preferences.jsonl" \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --out_dir "${OUT_ROOT}/dpo" \
  --device cpu \
  --max_steps 2 \
  --checkpoint_interval 1 \
  --temporal_agg

python "${ROOT_DIR}/scripts/train_posttrain_ppo.py" \
  --task_name sim_transfer_cube_scripted \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt "${OUT_ROOT}/dpo/policy_pref_last.ckpt" \
  --out_dir "${OUT_ROOT}/ppo" \
  --device cpu \
  --updates 1 \
  --num_rollouts 2 \
  --update_epochs 1 \
  --checkpoint_interval 1 \
  --temporal_agg

printf 'Local smoke outputs: %s\n' "${OUT_ROOT}"
