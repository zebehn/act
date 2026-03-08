# ACT Post-Training RL Artifact Schema

## Rollout artifact schema
Each collected rollout is stored as two files under the rollout directory:
- `<rollout_id>.json`
- `<rollout_id>.npz`

### Required metadata keys (`.json`)
- `schema_version`
- `rollout_id`
- `task_name`
- `source_checkpoint`
- `source_label`
- `seed`
- `candidate_index`
- `temporal_agg`
- `deterministic`
- `initial_object_pose`
- `num_steps`
- `episode_return`
- `highest_reward`
- `success`
- `env_max_reward`

### Stored arrays (`.npz`)
- `actions`
- `actions_norm`
- `actions_env`
- `rewards`
- `qpos`

### Collection root metadata
The rollout directory also contains:
- `collection_meta.json`
- `manifest.jsonl`

`collection_meta.json` records the task name, source checkpoint, dataset stats path, rollout counts, device, source label, and temporal aggregation setting.

## Preference schema
Preference output consists of:
- `preferences.jsonl`
- `preferences.jsonl.meta.json`

Each JSONL record stores:
- `pair_id`
- `task_name`
- `seed`
- `chosen_rollout_id`
- `rejected_rollout_id`
- `chosen_success`
- `rejected_success`
- `chosen_return`
- `rejected_return`
- `window_start`
- `window_length`
- `ranking_rule`

The current ranking rule is:
- `success_first_return_tiebreak`

## DPO outputs
Expected files in a DPO output directory:
- `config.json`
- `metrics.jsonl`
- `policy_pref_best.ckpt`
- `policy_pref_last.ckpt`
- `policy_pref_step_<N>.ckpt`
- `dpo_state.pt`
- `dataset_stats.pkl` (symlink)

## PPO-like outputs
Expected files in a PPO output directory:
- `config.json`
- `metrics.jsonl`
- `policy_ppo_best.ckpt`
- `policy_ppo_last.ckpt`
- `policy_ppo_update_<N>.ckpt`
- `ppo_state.pt`
- `dataset_stats.pkl` (symlink)

## Evaluation outputs
The post-training checkpoint evaluator writes:
- rollout artifacts under `--out_dir`
- `result_summary.json`

Recommended summary fields:
- `success_rate`
- `average_return`
- `rollout_count`
- `max_reward`
- `reward_thresholds`

Header-only fields in `preferences.jsonl.meta.json` include:
- `task_name`
- `camera_names`
- `dataset_stats_path`
- `source_checkpoint`
- `rollout_source_label`
- `pair_count`
- `window_length`
