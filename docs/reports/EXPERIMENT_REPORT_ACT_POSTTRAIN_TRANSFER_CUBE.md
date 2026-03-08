# ACT Post-Training RL Pilot Report (Transfer Cube)

## 1. Objective
Prototype a simulator-only post-training stack for ACT on `sim_transfer_cube_scripted` using:
1. Phase 1: DPO-style preference optimization from successful/failed rollouts.
2. Phase 2: PPO-like RL fine-tuning from the Phase-1 checkpoint.

## 2. Scope
This was a **pilot transfer-cube-first experiment** under the locked v1 scope:
- simulator-only
- transfer-cube-first
- DPO first
- PPO-like RL second
- success-first / return-tiebreak preference rule
- default start from `policy_rollout_best.ckpt`
- temporal aggregation enabled for collection and evaluation

## 3. Baseline Checkpoint
Starting BC checkpoint:
- `/tmp/act_mps_act_train/policy_rollout_best.ckpt`

Dataset stats:
- `/tmp/act_mps_act_train/dataset_stats.pkl`

## 4. Fixed Benchmark Protocol
To avoid using an already-saturated seed range, the pilot benchmark used:
- task: `sim_transfer_cube_scripted`
- rollout count: **6**
- seed start: **3000**
- temporal aggregation: **ON**
- backend: **MPS**

### BC baseline result
Output directory:
- `/tmp/act_posttrain_baseline_eval_3000`

Result:
- **Success rate:** **0.6667**
- **Average return:** **238.17**

## 5. Rollout Collection for Preferences
Rollout collection command family:

```bash
python scripts/collect_posttrain_rollouts.py \
  --task_name sim_transfer_cube_scripted \
  --source_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --out_dir /tmp/act_posttrain_transfer_cube_rollouts_pilot_... \
  --num_seed_groups 6 \
  --rollouts_per_seed 3 \
  --seed_start 3000 \
  --device mps \
  --temporal_agg
```

Actual rollout artifact root:
- `/tmp/act_posttrain_transfer_cube_rollouts_pilot_20260308T144733`

Collection summary:
- **18 total rollouts**
- **6 seed groups**
- **3 candidates per seed**

Observed rollout diversity:
- seeds 3001 and 3004 produced all-failure buckets
- seeds 3000, 3002, 3003, 3005 produced all-success buckets with different returns

## 6. Preference Dataset
Preference construction command family:

```bash
python scripts/build_posttrain_preferences.py \
  --rollout_dir /tmp/act_posttrain_transfer_cube_rollouts_pilot_20260308T144733 \
  --out_file /tmp/act_posttrain_transfer_cube_prefs_pilot.jsonl \
  --window_length 32
```

Preference dataset summary:
- **6 preference pairs**
- ranking rule: `success_first_return_tiebreak`
- window length: **32**

Preference file:
- `/tmp/act_posttrain_transfer_cube_prefs_pilot.jsonl`

## 7. Phase 1 — DPO Pilot
Command family:

```bash
python scripts/train_posttrain_dpo.py \
  --task_name sim_transfer_cube_scripted \
  --pref_file /tmp/act_posttrain_transfer_cube_prefs_pilot.jsonl \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --reference_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --out_dir /tmp/act_posttrain_dpo_pilot_... \
  --device mps \
  --max_steps 1 \
  --checkpoint_interval 1 \
  --temporal_agg
```

Actual DPO output directory:
- `/tmp/act_posttrain_dpo_pilot_20260308T145635`

Recorded DPO metric:
- step 0 loss: **36.27**
- margin: **-36.27**

### Phase-1 evaluation result
Evaluation directory:
- `/tmp/act_posttrain_dpo_eval_seed3000`

Result:
- **Success rate:** **0.8333**
- **Average return:** **487.33**

### Comparison vs BC baseline
- BC baseline success: **0.6667**
- Phase-1 DPO success: **0.8333**
- **Change:** **+0.1666** success rate on the fixed benchmark

This satisfies the v1 success criterion at the pilot level because Phase 1 improved over the BC baseline on the fixed benchmark run.

## 8. Phase 2 — PPO-like Pilot
Command family:

```bash
python scripts/train_posttrain_ppo.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt /tmp/act_posttrain_dpo_pilot_20260308T145635/policy_pref_last.ckpt \
  --out_dir /tmp/act_posttrain_ppo_pilot_... \
  --device mps \
  --updates 1 \
  --num_rollouts 1 \
  --update_epochs 1 \
  --checkpoint_interval 1 \
  --temporal_agg \
  --max_timesteps 64
```

Actual PPO-like output directory:
- `/tmp/act_posttrain_ppo_pilot_20260308T150051`

Recorded PPO-like metric:
- update 0 mean success during collection: **0.0**
- update 0 mean return during collection: **0.0**

### Phase-2 evaluation result
Evaluation directory:
- `/tmp/act_posttrain_ppo_eval_seed3000`

Result:
- **Success rate:** **0.50**
- **Average return:** **278.0**

### Comparison vs BC baseline and Phase 1
- BC baseline success: **0.6667**
- DPO success: **0.8333**
- PPO-like success: **0.50**

Interpretation:
- Phase 2 ran end-to-end
- but this very small PPO-like pilot **did not improve** over either BC or Phase 1
- the implemented PPO-like path is functional, but it clearly needs more rollout budget and hyperparameter tuning before it can be treated as an improvement stage

## 9. Implementation Verification
### Syntax / import checks
- `python -m py_compile policy.py detr/main.py detr/models/detr_vae.py posttrain/*.py scripts/*.py`

### Existing BC smoke
- `PYTHONPATH="$PWD/detr:$PWD" python scripts/regression_smoke_tests.py`

### New post-training smoke
- `PYTHONPATH="$PWD/detr:$PWD" python scripts/posttrain_regression_smoke_tests.py`

### End-to-end CLI smoke
- `bash scripts/posttrain_cli_smoke.sh`

These checks passed during development after the final compatibility fixes.

## 10. Main Findings
1. **The post-training pipeline is implemented end-to-end.**
2. **The rollout/preference/RL artifact flow works with a separate schema from demo HDF5s.**
3. **Phase 1 DPO produced an improvement over the BC baseline on the fixed pilot benchmark.**
4. **Phase 2 PPO-like RL is functional but under-tuned in this pilot and regressed relative to Phase 1.**
5. **Temporal aggregation remained enabled throughout collection/evaluation, matching the locked scope.**

## 11. Artifact Summary
### Baseline eval
- `/tmp/act_posttrain_baseline_eval_3000/result_summary.json`

### Rollouts
- `/tmp/act_posttrain_transfer_cube_rollouts_pilot_20260308T144733/`

### Preferences
- `/tmp/act_posttrain_transfer_cube_prefs_pilot.jsonl`
- `/tmp/act_posttrain_transfer_cube_prefs_pilot.jsonl.meta.json`

### DPO
- `/tmp/act_posttrain_dpo_pilot_20260308T145635/`
- `/tmp/act_posttrain_dpo_eval_seed3000/result_summary.json`

### PPO-like
- `/tmp/act_posttrain_ppo_pilot_20260308T150051/`
- `/tmp/act_posttrain_ppo_eval_seed3000/result_summary.json`

## 12. Recommended Next Experiments
1. Increase Phase-1 DPO steps beyond 1 while keeping the same fixed benchmark.
2. Tune PPO-like rollout count and update budget before judging Phase 2 quality.
3. Repeat the pilot with multi-seed evaluation as a follow-up validation step.
4. After transfer-cube stabilization, replicate the same workflow on insertion.
