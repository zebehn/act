#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/sim_transfer_cube_scripted"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/download_transfer_cube_all.log"
FOLDER_URL="https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj"
TARGET_COUNT=50

mkdir -p "${TARGET_DIR}" "${LOG_DIR}"

count_eps() {
  find "${TARGET_DIR}" -maxdepth 1 -name 'episode_*.hdf5' | wc -l | tr -d ' '
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] start download loop" >> "${LOG_FILE}"

while true; do
  current="$(count_eps)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] current episodes=${current}" >> "${LOG_FILE}"
  if [[ "${current}" -ge "${TARGET_COUNT}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] target reached (${current})" >> "${LOG_FILE}"
    exit 0
  fi

  python -m gdown --folder --remaining-ok "${FOLDER_URL}" -O "${TARGET_DIR}" >> "${LOG_FILE}" 2>&1 || true
  sleep 15
done
