# LIBERO-Object Task 0 Usage

## Runtime
This integration is designed for a separate environment, e.g. `libero_act`.

Required pieces:
- this repo on the current branch
- official LIBERO repo cloned locally
- LIBERO benchmark dependencies (`robosuite`, `bddl`, etc.)
- LIBERO Object dataset downloaded

Environment variables:
```bash
export LIBERO_REPO_PATH=/tmp/libero_official_69231
export PYTHONPATH="$LIBERO_REPO_PATH:$PWD/detr:$PWD"
export MPLCONFIGDIR=/tmp/matplotlib
```

LIBERO dataset/config paths are resolved from `~/.libero/config.yaml` when available.

## Task Name
Use:
```bash
--task_name libero_object_task0
```

This maps to:
- suite: `libero_object`
- task id: `0`
- demo file: `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`

## BC Training
Example 10-epoch pilot:
```bash
python -u imitate_episodes.py \
  --task_name libero_object_task0 \
  --ckpt_dir /tmp/libero_object_task0_bc10 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 4 \
  --dim_feedforward 3200 \
  --num_epochs 10 \
  --lr 1e-5 \
  --seed 0 \
  --device mps
```

## Closed-Loop Evaluation
```bash
python -u imitate_episodes.py \
  --eval \
  --task_name libero_object_task0 \
  --ckpt_dir /tmp/libero_object_task0_bc10 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 4 \
  --dim_feedforward 3200 \
  --num_epochs 10 \
  --lr 1e-5 \
  --seed 0 \
  --device mps \
  --temporal_agg \
  --eval_ckpt_names policy_best.ckpt \
  --no_save_episode
```

## Rollout Probe for DPO Readiness
```bash
python -u scripts/collect_posttrain_rollouts.py \
  --task_name libero_object_task0 \
  --source_ckpt /tmp/libero_object_task0_bc10/policy_best.ckpt \
  --dataset_stats /tmp/libero_object_task0_bc10/dataset_stats.pkl \
  --out_dir /tmp/libero_object_task0_rollout_probe \
  --num_seed_groups 2 \
  --rollouts_per_seed 2 \
  --seed_start 3000 \
  --device mps \
  --temporal_agg \
  --sampling_strategy auto \
  --exploration_std 0.05 \
  --deterministic_candidates 1
```

If rollout success remains zero across all candidates, DPO preference optimization is not yet informative for this task/checkpoint.
