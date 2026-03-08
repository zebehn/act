#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
export SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/tmp/act_mps_act_train}"
export SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
export DRY_RUN="${DRY_RUN:-0}"

exec bash "${ROOT_DIR}/scripts/eval_checkpoint_sweep_temporal_agg.sh" "$@"
