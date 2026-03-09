# ACT Post-Training RL Usage Guide

## Environment
Recommended environment variables:

```bash
export ACT_DATA_DIR="$(pwd)/data"
export PYTHONPATH="$(pwd)/detr:$(pwd)"
export MPLCONFIGDIR=/tmp/matplotlib
```

## 1. Collect rollouts from a BC checkpoint
Example: collect 24 matched seed groups with 4 rollouts each for transfer cube. `auto` sampling falls back to mean-centered Gaussian exploration for legacy BC checkpoints whose learned action std was never trained.

```bash
python scripts/collect_posttrain_rollouts.py \
  --task_name sim_transfer_cube_scripted \
  --source_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --out_dir /tmp/act_posttrain_transfer_cube_rollouts \
  --num_seed_groups 24 \
  --rollouts_per_seed 4 \
  --seed_start 3000 \
  --device mps \
  --temporal_agg \
  --sampling_strategy auto \
  --exploration_std 0.05
```

## 2. Build preferences from rollout artifacts
```bash
python scripts/build_posttrain_preferences.py \
  --rollout_dir /tmp/act_posttrain_transfer_cube_rollouts \
  --out_file /tmp/act_posttrain_transfer_cube_prefs.jsonl \
  --window_length 32
```

## 3. Phase 1 — DPO-style preference optimization
```bash
python scripts/train_posttrain_dpo.py \
  --task_name sim_transfer_cube_scripted \
  --pref_file /tmp/act_posttrain_transfer_cube_prefs.jsonl \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --reference_ckpt /tmp/act_mps_act_train/policy_rollout_best.ckpt \
  --out_dir /tmp/act_posttrain_dpo \
  --device mps \
  --max_steps 3 \
  --checkpoint_interval 1 \
  --temporal_agg
```

## 4. Phase 2 — PPO-like RL fine-tuning
```bash
python scripts/train_posttrain_ppo.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_stats /tmp/act_mps_act_train/dataset_stats.pkl \
  --init_ckpt /tmp/act_posttrain_dpo/policy_pref_last.ckpt \
  --out_dir /tmp/act_posttrain_ppo \
  --device mps \
  --updates 1 \
  --num_rollouts 1 \
  --update_epochs 1 \
  --checkpoint_interval 1 \
  --temporal_agg \
  --max_timesteps 64
```

## 5. Evaluate a checkpoint on a fixed benchmark protocol
```bash
python scripts/evaluate_posttrain_checkpoint.py \
  --task_name sim_transfer_cube_scripted \
  --source_ckpt /tmp/act_posttrain_dpo/policy_pref_last.ckpt \
  --dataset_stats /tmp/act_posttrain_dpo/dataset_stats.pkl \
  --out_dir /tmp/act_posttrain_dpo_eval_seed3000 \
  --num_rollouts 6 \
  --seed_start 3000 \
  --device mps \
  --temporal_agg
```

## 6. Optional: use the generic rollout-best sweep on post-training outputs
```bash
TASK_NAME=sim_transfer_cube_scripted \
SOURCE_CKPT_DIR=/tmp/act_posttrain_dpo \
SAVE_VIDEOS=0 \
bash scripts/eval_posttrain_sweep.sh policy_pref_last.ckpt policy_pref_best.ckpt
```

## Recommended v1 evaluation protocol
For the pilot transfer-cube workflow:
- task: `sim_transfer_cube_scripted`
- temporal aggregation: ON
- rollout count: fixed per experiment report
- compare:
  - BC rollout-best baseline
  - Phase-1 DPO checkpoint
  - Phase-2 PPO-like checkpoint
