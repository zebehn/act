#!/usr/bin/env bash

rollout_best_task_tag() {
  local task_name="$1"
  printf '%s' "${task_name#sim_}" | sed 's/_scripted$//' | tr -c 'A-Za-z0-9._-' '_'
}

rollout_best_default_ckpt_dir() {
  local task_tag="$1"
  printf '/tmp/act_%s_mps_act_train\n' "${task_tag}"
}

rollout_best_resolve_source_ckpt_name() {
  local source_ckpt_dir="$1"
  local explicit_name="${2:-}"
  local latest_rollout_best

  if [[ -n "${explicit_name}" ]]; then
    printf '%s\n' "${explicit_name}"
    return 0
  fi

  if [[ -e "${source_ckpt_dir}/policy_rollout_best.ckpt" || -L "${source_ckpt_dir}/policy_rollout_best.ckpt" ]]; then
    printf 'policy_rollout_best.ckpt\n'
    return 0
  fi

  latest_rollout_best="$(find "${source_ckpt_dir}" -maxdepth 1 -name 'policy_rollout_best_epoch_*_seed_*.ckpt' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_rollout_best}" ]]; then
    printf '%s\n' "${latest_rollout_best}"
    return 0
  fi

  for candidate in policy_best.ckpt policy_last.ckpt; do
    if [[ -e "${source_ckpt_dir}/${candidate}" || -L "${source_ckpt_dir}/${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}
