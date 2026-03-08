#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
export CKPT_DIR="${CKPT_DIR:-/tmp/act_mps_act_train}"
export TRAIN_LOG="${TRAIN_LOG:-${ROOT_DIR}/logs/train_transfer_cube_act_mps_with_rollout_best.log}"
export SWEEP_LOG="${SWEEP_LOG:-${ROOT_DIR}/logs/train_transfer_cube_act_mps_with_rollout_best_sweep.log}"
export SWEEP_SUMMARY_TSV="${SWEEP_SUMMARY_TSV:-${ROOT_DIR}/logs/train_transfer_cube_act_mps_with_rollout_best_sweep.tsv}"
export SWEEP_BEST_META="${SWEEP_BEST_META:-${ROOT_DIR}/logs/train_transfer_cube_act_mps_with_rollout_best_best.txt}"

exec bash "${ROOT_DIR}/scripts/train_act_mps_with_rollout_best.sh" "$@"
