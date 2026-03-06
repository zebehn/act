#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/sim_transfer_cube_scripted"
LOG_DIR="${ROOT_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/act_train_auto.log"
MONITOR_LOG="${LOG_DIR}/download_monitor.log"
TARGET_EPISODES=50

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] monitor start" >> "${MONITOR_LOG}"

count_episodes() {
  find "${DATA_DIR}" -maxdepth 1 -name 'episode_*.hdf5' | wc -l | tr -d ' '
}

while true; do
  COUNT="$(count_episodes)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] episodes=${COUNT}" >> "${MONITOR_LOG}"

  if [[ "${COUNT}" -ge "${TARGET_EPISODES}" ]]; then
    if [[ -f "${LOG_DIR}/act_train_started.flag" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] training already started; monitor exit" >> "${MONITOR_LOG}"
      exit 0
    fi

    touch "${LOG_DIR}/act_train_started.flag"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching ACT training on MPS" >> "${MONITOR_LOG}"

    ACT_DATA_DIR="${ROOT_DIR}/data" \
    PYTHONPATH="${ROOT_DIR}/detr" \
    MPLCONFIGDIR=/tmp/matplotlib \
    conda run -n aloha python "${ROOT_DIR}/imitate_episodes.py" \
      --task_name sim_transfer_cube_scripted \
      --ckpt_dir /tmp/act_mps_act_train \
      --policy_class ACT \
      --kl_weight 10 \
      --chunk_size 100 \
      --hidden_dim 512 \
      --batch_size 8 \
      --dim_feedforward 3200 \
      --num_epochs 2000 \
      --lr 1e-5 \
      --seed 0 \
      --device mps >> "${TRAIN_LOG}" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] training process finished" >> "${MONITOR_LOG}"
    exit 0
  fi

  sleep 60
done
