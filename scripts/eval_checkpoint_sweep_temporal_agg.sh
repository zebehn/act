#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/rollout_best_common.sh"

TASK_NAME="${TASK_NAME:-sim_transfer_cube_scripted}"
TASK_TAG="$(rollout_best_task_tag "${TASK_NAME}")"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-$(rollout_best_default_ckpt_dir "${TASK_TAG}")}"
RUN_ROOT="${RUN_ROOT:-/tmp/act_${TASK_TAG}_ckpt_sweep_$(date +%Y%m%dT%H%M%S)}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_LOG="${MASTER_LOG:-${ROOT_DIR}/logs/eval_checkpoint_sweep_${TASK_TAG}_temporal_agg.log}"
SUMMARY_TSV="${SUMMARY_TSV:-${ROOT_DIR}/logs/eval_checkpoint_sweep_${TASK_TAG}_temporal_agg.tsv}"
BEST_META="${BEST_META:-${ROOT_DIR}/logs/eval_checkpoint_sweep_${TASK_TAG}_temporal_agg_best.txt}"

mkdir -p "${RUN_ROOT}" "${ROOT_DIR}/logs"

collect_default_ckpts() {
  local tmpfile
  tmpfile="$(mktemp)"
  find "${SOURCE_CKPT_DIR}" -maxdepth 1 -name 'policy_epoch_*_seed_*.ckpt' | sort -V | tail -n 5 > "${tmpfile}" || true
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    basename "${line}"
  done < "${tmpfile}"
  rm -f "${tmpfile}"
  for extra in policy_last.ckpt policy_best.ckpt policy_rollout_best.ckpt; do
    if [[ -f "${SOURCE_CKPT_DIR}/${extra}" || -L "${SOURCE_CKPT_DIR}/${extra}" ]]; then
      echo "${extra}"
    fi
  done
}

dry_run_default_ckpts() {
  for epoch in 1500 1600 1700 1800 1900; do
    echo "policy_epoch_${epoch}_seed_0.ckpt"
  done
  echo 'policy_last.ckpt'
  echo 'policy_best.ckpt'
  echo 'policy_rollout_best.ckpt'
}

unique_append() {
  local item="$1"
  for existing in "${CKPTS[@]:-}"; do
    [[ "${existing}" == "${item}" ]] && return 0
  done
  CKPTS+=("${item}")
}

CKPTS=()
if [[ "$#" -gt 0 ]]; then
  for arg in "$@"; do
    unique_append "${arg}"
  done
else
  while IFS= read -r ck; do
    [[ -n "${ck}" ]] || continue
    unique_append "${ck}"
  done < <(collect_default_ckpts)
fi

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    while IFS= read -r ck; do
      [[ -n "${ck}" ]] || continue
      unique_append "${ck}"
    done < <(dry_run_default_ckpts)
  else
    echo "No checkpoints found to evaluate in ${SOURCE_CKPT_DIR}" >&2
    exit 1
  fi
fi

printf 'checkpoint\tsuccess_rate\taverage_return\trun_dir\n' > "${SUMMARY_TSV}"
: > "${MASTER_LOG}"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] checkpoint sweep start"
  echo "TASK_NAME=${TASK_NAME}"
  echo "SOURCE_CKPT_DIR=${SOURCE_CKPT_DIR}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SAVE_VIDEOS=${SAVE_VIDEOS}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "CANDIDATES=${CKPTS[*]}"
} | tee -a "${MASTER_LOG}"

for CKPT in "${CKPTS[@]}"; do
  if [[ "${DRY_RUN}" != "1" && ! -f "${SOURCE_CKPT_DIR}/${CKPT}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP missing ${CKPT}" | tee -a "${MASTER_LOG}"
    continue
  fi

  RUN_DIR="${RUN_ROOT}/${CKPT%.ckpt}"
  mkdir -p "${RUN_DIR}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${SOURCE_CKPT_DIR}/dataset_stats.pkl" ]]; then
      echo "Missing dataset stats: ${SOURCE_CKPT_DIR}/dataset_stats.pkl" >&2
      exit 1
    fi

    ln -sfn "${SOURCE_CKPT_DIR}/${CKPT}" "${RUN_DIR}/policy_best.ckpt"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${CKPT}"
    echo "RUN_DIR=${RUN_DIR}"
    echo "COMMAND=${CMD[*]}"
  } | tee -a "${MASTER_LOG}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DRY_RUN_DONE ${CKPT}" | tee -a "${MASTER_LOG}"
    continue
  fi

  ACT_DATA_DIR="${ROOT_DIR}/data" \
  PYTHONPATH="${ROOT_DIR}/detr:${ROOT_DIR}" \
  MPLCONFIGDIR=/tmp/matplotlib \
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"

  python - <<'PY' "${RUN_DIR}/result_policy_best.txt" "${CKPT}" "${RUN_DIR}" >> "${SUMMARY_TSV}"
import pathlib
import re
import sys

result_path = pathlib.Path(sys.argv[1])
ckpt = sys.argv[2]
run_dir = sys.argv[3]
text = result_path.read_text()
ms = re.search(r'Success rate:\s*([0-9.]+)', text)
ma = re.search(r'Average return:\s*([0-9.]+)', text)
if not ms or not ma:
    raise SystemExit(f'Could not parse metrics from {result_path}')
print(f"{ckpt}\t{ms.group(1)}\t{ma.group(1)}\t{run_dir}")
PY

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${CKPT}" | tee -a "${MASTER_LOG}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] checkpoint sweep dry-run complete" | tee -a "${MASTER_LOG}"
  exit 0
fi

python - <<'PY' "${SUMMARY_TSV}" "${SOURCE_CKPT_DIR}" "${BEST_META}" | tee -a "${MASTER_LOG}"
import csv
import os
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
source_ckpt_dir = pathlib.Path(sys.argv[2])
best_meta = pathlib.Path(sys.argv[3])
rows = list(csv.DictReader(summary_path.open(), delimiter='\t'))
if not rows:
    raise SystemExit('No completed checkpoint evaluations found')
rows.sort(key=lambda r: (float(r['success_rate']), float(r['average_return'])), reverse=True)
best = rows[0]
link_path = source_ckpt_dir / 'policy_rollout_best.ckpt'
target_path = source_ckpt_dir / best['checkpoint']
if link_path.exists() or link_path.is_symlink():
    link_path.unlink()
os.symlink(target_path, link_path)
best_meta.write_text(
    f"best_checkpoint={best['checkpoint']}\n"
    f"task_name={os.environ.get('TASK_NAME', '')}\n"
    f"success_rate={best['success_rate']}\n"
    f"average_return={best['average_return']}\n"
    f"run_dir={best['run_dir']}\n"
    f"symlink={link_path}\n"
)
print('RANKED_RESULTS')
for row in rows:
    print(f"{row['checkpoint']}: success_rate={row['success_rate']} average_return={row['average_return']}")
print(f"ROLLOUT_BEST={best['checkpoint']} success_rate={best['success_rate']} average_return={best['average_return']}")
print(f"ROLLOUT_BEST_LINK={link_path} -> {target_path}")
PY
