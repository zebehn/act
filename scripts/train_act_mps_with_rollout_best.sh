#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/rollout_best_common.sh"

TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
TASK_TAG="$(rollout_best_task_tag "${TASK_NAME}")"
CKPT_DIR="${CKPT_DIR:-$(rollout_best_default_ckpt_dir "${TASK_TAG}")}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT_DIR}/logs/train_${TASK_TAG}_act_mps_with_rollout_best.log}"
SWEEP_LOG="${SWEEP_LOG:-${ROOT_DIR}/logs/train_${TASK_TAG}_act_mps_with_rollout_best_sweep.log}"
SWEEP_SUMMARY_TSV="${SWEEP_SUMMARY_TSV:-${ROOT_DIR}/logs/train_${TASK_TAG}_act_mps_with_rollout_best_sweep.tsv}"
SWEEP_BEST_META="${SWEEP_BEST_META:-${ROOT_DIR}/logs/train_${TASK_TAG}_act_mps_with_rollout_best_best.txt}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_AUTO="${RESUME_AUTO:-0}"
SWEEP_SAVE_VIDEOS="${SWEEP_SAVE_VIDEOS:-${SAVE_VIDEOS:-0}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${ROOT_DIR}/logs"

TRAIN_CMD=(
  python "${ROOT_DIR}/imitate_episodes.py"
  --task_name "${TASK_NAME}"
  --ckpt_dir "${CKPT_DIR}"
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
)

if [[ -n "${RESUME_CKPT}" ]]; then
  TRAIN_CMD+=(--resume_ckpt "${RESUME_CKPT}")
elif [[ "${RESUME_AUTO}" == "1" ]]; then
  TRAIN_CMD+=(--resume_ckpt auto)
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] train+rollout-best workflow start"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "TASK_NAME=${TASK_NAME}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "RESUME_CKPT=${RESUME_CKPT}"
  echo "RESUME_AUTO=${RESUME_AUTO}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "TRAIN_COMMAND=${TRAIN_CMD[*]}"
} | tee "${TRAIN_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  TASK_NAME="${TASK_NAME}" \
  SOURCE_CKPT_DIR="${CKPT_DIR}" \
  SAVE_VIDEOS="${SWEEP_SAVE_VIDEOS}" \
  DRY_RUN=1 \
  MASTER_LOG="${SWEEP_LOG}" \
  SUMMARY_TSV="${SWEEP_SUMMARY_TSV}" \
  BEST_META="${SWEEP_BEST_META}" \
  bash "${SCRIPT_DIR}/eval_checkpoint_sweep_temporal_agg.sh"
  exit 0
fi

ACT_DATA_DIR="${ROOT_DIR}/data" \
PYTHONPATH="${ROOT_DIR}/detr:${ROOT_DIR}" \
MPLCONFIGDIR=/tmp/matplotlib \
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${TRAIN_LOG}"

TASK_NAME="${TASK_NAME}" \
SOURCE_CKPT_DIR="${CKPT_DIR}" \
SAVE_VIDEOS="${SWEEP_SAVE_VIDEOS}" \
DRY_RUN=0 \
MASTER_LOG="${SWEEP_LOG}" \
SUMMARY_TSV="${SWEEP_SUMMARY_TSV}" \
BEST_META="${SWEEP_BEST_META}" \
  bash "${SCRIPT_DIR}/eval_checkpoint_sweep_temporal_agg.sh"
