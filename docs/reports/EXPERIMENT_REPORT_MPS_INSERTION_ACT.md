# ACT MPS Insertion Experiment Report

## 1. Objective
Train and evaluate the ACT policy on Apple Silicon using the PyTorch MPS backend for the `sim_insertion_scripted` task, then identify the checkpoint with the best closed-loop simulator rollout performance.

Date completed: **March 8, 2026** (Asia/Seoul)  
Repository: `act`

## 2. Environment and Setup
- Runtime backend target: `mps` (`--device mps`)
- Python environment: `conda` env `aloha`
- Dataset root via env var: `ACT_DATA_DIR="$(pwd)/data"`
- Model import path: `PYTHONPATH="$(pwd)/detr:$PWD"`
- Primary checkpoint directory: `/tmp/act_insertion_mps_act_train`

Observed runtime notes:
- MPS was used successfully for both training and evaluation (`Using device: mps` in logs).
- One non-blocking PyTorch warning persisted during training on Apple Silicon:
  - `aten::sgn.out` falls back to CPU during backward on MPS.
- Temporal aggregation evaluation on MPS required a small dtype fix so the temporal-agg weights are converted to `float32` before moving to MPS.
- The DETR compatibility argument parser in `detr/main.py` also needed to accept the eval-only flags `--eval_ckpt_names` and `--no_save_episode` so rollout checkpoint sweeps could run end-to-end.

## 3. Dataset
- Task: `sim_insertion_scripted`
- Dataset directory: `data/sim_insertion_scripted`
- Local availability confirmed: `episode_0.hdf5` ... `episode_49.hdf5` (50 episodes total, contiguous)
- Dataset family notes:
  - This is one of the two main simulator task families in the repo (`transfer_cube`, `insertion`).
  - It uses the insertion simulator environment and 50-episode scripted demonstration set.

## 4. Training Configuration
Command equivalent used by the generalized rollout-best workflow:

```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr:$PWD" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py \
  --task_name sim_insertion_scripted \
  --ckpt_dir /tmp/act_insertion_mps_act_train \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --device mps
```

Workflow wrapper used:

```bash
TASK_NAME=sim_insertion_scripted \
CKPT_DIR=/tmp/act_insertion_mps_act_train \
SAVE_VIDEOS=0 \
bash scripts/train_act_mps_with_rollout_best.sh
```

## 5. Execution Timeline and Incidents
1. Training started successfully on MPS and began writing periodic checkpoints every 100 epochs.
2. An interrupted turn terminated the active insertion training process before completion.
3. The run was resumed safely from the latest valid periodic checkpoint:
   - `/tmp/act_insertion_mps_act_train/policy_epoch_400_seed_0.ckpt`
   - resumed with `--resume_ckpt auto`
4. Training then completed successfully to epoch 2000.
5. The first automatic insertion checkpoint sweep failed because `detr/main.py` reparsed `sys.argv` and rejected the new eval-only flags (`--eval_ckpt_names`, `--no_save_episode`).
6. Parser compatibility was patched, and the insertion checkpoint sweep was rerun successfully.
7. A final clean rollout evaluation of the rollout-best checkpoint was run with temporal aggregation and rollout videos enabled.

## 6. Final Training Outcome
- Total target epochs: 2000
- Completed: yes
- Best validation loss: **0.045595**
- Best validation epoch: **1939**

Produced checkpoints:
- `/tmp/act_insertion_mps_act_train/policy_best.ckpt`
- `/tmp/act_insertion_mps_act_train/policy_last.ckpt`
- `/tmp/act_insertion_mps_act_train/policy_epoch_1939_seed_0.ckpt`
- Periodic epoch checkpoints (`policy_epoch_0_seed_0.ckpt`, `policy_epoch_100_seed_0.ckpt`, ..., `policy_epoch_1900_seed_0.ckpt`)

Produced plots:
- `/tmp/act_insertion_mps_act_train/train_val_loss_seed_0.png`
- `/tmp/act_insertion_mps_act_train/train_val_l1_seed_0.png`
- `/tmp/act_insertion_mps_act_train/train_val_kl_seed_0.png`

Training log:
- `logs/train_insertion_act_mps_with_rollout_best.log`

## 7. Rollout Checkpoint Sweep (Temporal Aggregation, No Videos)
To identify the best **closed-loop** checkpoint rather than relying only on offline validation loss, a checkpoint sweep was run with `--temporal_agg` and `--no_save_episode`.

Sweep command family:

```bash
TASK_NAME=sim_insertion_scripted \
SOURCE_CKPT_DIR=/tmp/act_insertion_mps_act_train \
SAVE_VIDEOS=0 \
bash scripts/eval_checkpoint_sweep_temporal_agg.sh
```

Sweep outputs:
- Log: `logs/eval_checkpoint_sweep_insertion_temporal_agg.log`
- TSV summary: `logs/eval_checkpoint_sweep_insertion_temporal_agg.tsv`
- Best-meta file: `logs/eval_checkpoint_sweep_insertion_temporal_agg_best.txt`
- Run root: `/tmp/act_insertion_ckpt_sweep_20260307T232656`

### Sweep results
| Checkpoint | Success rate | Average return |
|---|---:|---:|
| `policy_epoch_1600_seed_0.ckpt` | **0.58** | **418.8** |
| `policy_epoch_1700_seed_0.ckpt` | 0.26 | 364.46 |
| `policy_epoch_1800_seed_0.ckpt` | 0.36 | 352.42 |
| `policy_epoch_1900_seed_0.ckpt` | 0.28 | 386.42 |
| `policy_epoch_1939_seed_0.ckpt` | 0.40 | 386.82 |
| `policy_last.ckpt` | 0.16 | 369.44 |
| `policy_best.ckpt` | 0.40 | 386.82 |

### Sweep conclusion
The rollout-best insertion checkpoint was:
- **`policy_epoch_1600_seed_0.ckpt`**

This was preserved as:
- `/tmp/act_insertion_mps_act_train/policy_rollout_best.ckpt`

This result is significant because it shows the same pattern observed on transfer cube:
- the checkpoint with the lowest offline validation loss was **not** the checkpoint with the best simulator rollout performance.

## 8. Clean Final Evaluation of the Rollout-Best Checkpoint (Temporal Aggregation, Videos Enabled)
A clean final evaluation was run on the selected rollout-best insertion checkpoint with temporal aggregation enabled and rollout videos saved.

Command family:

```bash
TASK_NAME=sim_insertion_scripted \
SOURCE_CKPT_DIR=/tmp/act_insertion_mps_act_train \
SAVE_VIDEOS=1 \
bash scripts/eval_rollout_best_temporal_agg.sh
```

Run directory:
- `/tmp/act_eval_insertion_policy_rollout_best_20260308T083602`

Artifacts:
- Metrics file: `/tmp/act_eval_insertion_policy_rollout_best_20260308T083602/result_policy_best.txt`
- Rollout videos: `/tmp/act_eval_insertion_policy_rollout_best_20260308T083602/video0.mp4` ... `video49.mp4`
- Log: `logs/eval_rollout_best_insertion_temporal_agg.log`

### Final rollout-best insertion result
- **Success rate:** **0.58** (29/50)
- **Average return:** **418.8**
- Reward>=4: **29/50 (58.0%)**

Reward breakdown:
- Reward >= 0: **50/50 (100.0%)**
- Reward >= 1: **50/50 (100.0%)**
- Reward >= 2: **50/50 (100.0%)**
- Reward >= 3: **48/50 (96.0%)**
- Reward >= 4: **29/50 (58.0%)**

## 9. Temporal Aggregation A/B Comparison on the Same Rollout-Best Checkpoint
To measure the impact of action ensembling directly, the same rollout-best insertion checkpoint (`policy_epoch_1600_seed_0.ckpt`) was evaluated both with and without temporal aggregation.

### Without temporal aggregation
Clean result on the same rollout-best checkpoint, videos disabled:
- **Success rate:** **0.22**
- **Average return:** **407.06**
- Reward >= 4: **11/50 (22.0%)**

Artifacts:
- Log: `logs/eval_rollout_best_insertion_no_temporal_agg.log`
- Result dir: `/tmp/act_eval_insertion_policy_rollout_best_no_temporal_agg_20260308T085941`

### With temporal aggregation
- **Success rate:** **0.58**
- **Average return:** **418.8**
- Reward >= 4: **29/50 (58.0%)**

### A/B conclusion
Temporal aggregation materially improved insertion rollout success on the best checkpoint:
- **0.22 → 0.58** success rate
- **+36 percentage points**

Average return moved only modestly:
- **407.06 → 418.8**

This indicates that temporal aggregation helped the policy complete the final insertion step much more reliably, even when the underlying average return metric changed less dramatically.

## 10. Interpretation
### Main findings
1. **MPS training/evaluation path is functional end-to-end for insertion.**
2. **Offline validation loss was not sufficient for checkpoint selection.**
   - `policy_best.ckpt` (selected by validation loss at epoch 1939) achieved only **0.40** rollout success in the temporal-agg sweep.
   - `policy_epoch_1600_seed_0.ckpt` achieved the best insertion rollout success at **0.58**.
3. **Temporal aggregation is especially important for insertion.**
   - The same rollout-best checkpoint dropped from **0.58** success to **0.22** without temporal aggregation.
4. **The insertion task result is broadly consistent with the original README’s expectation.**
   - The README states insertion success should be around **50%**.
   - The final clean rollout-best insertion result was **58%**.

### Practical recommendation
For this run, the preferred insertion checkpoint is:
- **`/tmp/act_insertion_mps_act_train/policy_rollout_best.ckpt`**

For future insertion experiments, checkpoint selection should continue to use the rollout-best workflow rather than relying only on `policy_best.ckpt` from validation loss.

## 11. Artifact Summary
### Training
- `/tmp/act_insertion_mps_act_train/`
- `logs/train_insertion_act_mps_with_rollout_best.log`

### Sweep
- `logs/eval_checkpoint_sweep_insertion_temporal_agg.log`
- `logs/eval_checkpoint_sweep_insertion_temporal_agg.tsv`
- `logs/eval_checkpoint_sweep_insertion_temporal_agg_best.txt`
- `/tmp/act_insertion_ckpt_sweep_20260307T232656/`

### Final clean temporal-agg evaluation
- `logs/eval_rollout_best_insertion_temporal_agg.log`
- `/tmp/act_eval_insertion_policy_rollout_best_20260308T083602/result_policy_best.txt`
- `/tmp/act_eval_insertion_policy_rollout_best_20260308T083602/video0.mp4` ... `video49.mp4`

### Non-temporal comparison
- `logs/eval_rollout_best_insertion_no_temporal_agg.log`
- `/tmp/act_eval_insertion_policy_rollout_best_no_temporal_agg_20260308T085941/result_policy_best.txt`
