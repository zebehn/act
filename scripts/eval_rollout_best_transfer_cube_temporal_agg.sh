#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
export SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/tmp/act_mps_act_train}"
export LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/eval_rollout_best_transfer_cube_temporal_agg.log}"

exec bash "${ROOT_DIR}/scripts/eval_rollout_best_temporal_agg.sh" "$@"
