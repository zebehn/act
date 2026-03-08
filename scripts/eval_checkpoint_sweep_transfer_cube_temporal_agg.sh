#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
export SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/tmp/act_mps_act_train}"
export RUN_ROOT="${RUN_ROOT:-/tmp/act_ckpt_sweep_$(date +%Y%m%dT%H%M%S)}"
export MASTER_LOG="${MASTER_LOG:-${ROOT_DIR}/logs/eval_checkpoint_sweep_transfer_cube_temporal_agg.log}"
export SUMMARY_TSV="${SUMMARY_TSV:-${ROOT_DIR}/logs/eval_checkpoint_sweep_transfer_cube_temporal_agg.tsv}"
export BEST_META="${BEST_META:-${ROOT_DIR}/logs/eval_checkpoint_sweep_transfer_cube_temporal_agg_best.txt}"

exec bash "${ROOT_DIR}/scripts/eval_checkpoint_sweep_temporal_agg.sh" "$@"
