#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/rollout_best_common.sh"

TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
TASK_TAG="$(rollout_best_task_tag "${TASK_NAME}")"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-$(rollout_best_default_ckpt_dir "${TASK_TAG}")}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -n "${SOURCE_CKPT_NAME:-}" ]]; then
    SOURCE_CKPT_NAME="${SOURCE_CKPT_NAME}"
  elif [[ -d "${SOURCE_CKPT_DIR}" ]]; then
    SOURCE_CKPT_NAME="$(rollout_best_resolve_source_ckpt_name "${SOURCE_CKPT_DIR}" "" 2>/dev/null || printf 'policy_rollout_best.ckpt')"
  else
    SOURCE_CKPT_NAME='policy_rollout_best.ckpt'
  fi
else
  if ! SOURCE_CKPT_NAME="$(rollout_best_resolve_source_ckpt_name "${SOURCE_CKPT_DIR}" "${SOURCE_CKPT_NAME:-}" 2>/dev/null)"; then
    echo "Could not resolve a rollout-best checkpoint in ${SOURCE_CKPT_DIR}. Set SOURCE_CKPT_NAME explicitly if needed." >&2
    exit 1
  fi
fi
RUN_DIR="${RUN_DIR:-/tmp/act_eval_${TASK_TAG}_${SOURCE_CKPT_NAME%.ckpt}_$(date +%Y%m%dT%H%M%S)}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/eval_rollout_best_${TASK_TAG}_temporal_agg.log}"

mkdir -p "${RUN_DIR}" "${ROOT_DIR}/logs"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${SOURCE_CKPT_DIR}/${SOURCE_CKPT_NAME}" ]]; then
    echo "Checkpoint not found: ${SOURCE_CKPT_DIR}/${SOURCE_CKPT_NAME}" >&2
    exit 1
  fi

  if [[ ! -f "${SOURCE_CKPT_DIR}/dataset_stats.pkl" ]]; then
    echo "Missing dataset stats: ${SOURCE_CKPT_DIR}/dataset_stats.pkl" >&2
    exit 1
  fi

  ln -sfn "${SOURCE_CKPT_DIR}/${SOURCE_CKPT_NAME}" "${RUN_DIR}/policy_best.ckpt"
  ln -sfn "${SOURCE_CKPT_DIR}/dataset_stats.pkl" "${RUN_DIR}/dataset_stats.pkl"
fi

CMD=(
  python "${ROOT_DIR}/imitate_episodes.py"
  --eval
  --task_name "${TASK_NAME}"
  --ckpt_dir "${RUN_DIR}"
  --policy_class ACT
  --kl_weight 10
  --chunk_size 100
  --hidden_dim 512
  --batch_size 8
  --dim_feedforward 3200
  --num_epochs 2000
  --lr 1e-5
  --seed 0
  --device mps
  --temporal_agg
  --eval_ckpt_names policy_best.ckpt
)

if [[ "${SAVE_VIDEOS}" != "1" ]]; then
  CMD+=(--no_save_episode)
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval script start"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "TASK_NAME=${TASK_NAME}"
  echo "SOURCE_CKPT_DIR=${SOURCE_CKPT_DIR}"
  echo "SOURCE_CKPT_NAME=${SOURCE_CKPT_NAME}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "SAVE_VIDEOS=${SAVE_VIDEOS}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "COMMAND=${CMD[*]}"
} | tee "${LOG_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DRY_RUN_DONE ${SOURCE_CKPT_NAME}" | tee -a "${LOG_PATH}"
  printf 'Run dir: %s\n' "${RUN_DIR}"
  printf 'Checkpoint: %s\n' "${SOURCE_CKPT_DIR}/${SOURCE_CKPT_NAME}"
  printf 'Log file: %s\n' "${LOG_PATH}"
  exit 0
fi

ACT_DATA_DIR="${ROOT_DIR}/data" \
PYTHONPATH="${ROOT_DIR}/detr:${ROOT_DIR}" \
MPLCONFIGDIR=/tmp/matplotlib \
"${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo
printf 'Run dir: %s\n' "${RUN_DIR}"
printf 'Result file: %s\n' "${RUN_DIR}/result_policy_best.txt"
printf 'Log file: %s\n' "${LOG_PATH}"
