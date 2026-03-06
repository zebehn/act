# ACT MPS Experiment Report

## 1. Objective
Train and evaluate the ACT policy on Apple Silicon using PyTorch MPS backend, then verify rollout performance on `sim_transfer_cube_scripted`.

Date completed: **March 5, 2026** (Asia/Seoul)  
Repository: `act`

## 2. Environment and Setup
- Runtime backend target: `mps` (`--device mps`)
- Python env used: `conda` env `aloha`
- Dataset root via env var: `ACT_DATA_DIR="$(pwd)/data"`
- Model import path: `PYTHONPATH="$(pwd)/detr"`
- Output directory: `/tmp/act_mps_act_train`

Key dependency fixes during evaluation:
- Installed `scipy` in `aloha` env (required by `dm_control` import chain during sim evaluation).

## 3. Dataset
- Task: `sim_transfer_cube_scripted`
- Local availability confirmed: `episode_0.hdf5` ... `episode_49.hdf5` (50 episodes total, contiguous)
- This matches README’s standard sim setup size.

## 4. Training Configuration
Command (ACT training):

```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
conda run --no-capture-output -n aloha python -u imitate_episodes.py \
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
  --device mps
```

## 5. Execution Timeline and Incidents
1. Training started and confirmed `Using device: mps`.
2. First long run stopped at epoch ~900 due checkpoint write failure (`PytorchStreamWriter failed writing file`).
3. Resume support was implemented:
   - Added `--resume_ckpt` option (`auto` or explicit path).
   - Auto-skip invalid/corrupted checkpoints.
4. Resumed successfully from `policy_epoch_800_seed_0.ckpt` and completed to epoch 2000.
5. A post-run `NameError` (stray code after `main()`) occurred after completion; fixed immediately. This did **not** invalidate trained outputs.

## 6. Final Training Outcome
- Total target epochs: 2000
- Completed: yes
- Best validation loss: **0.044867**
- Best epoch: **1921**

Produced checkpoints:
- `/tmp/act_mps_act_train/policy_best.ckpt`
- `/tmp/act_mps_act_train/policy_last.ckpt`
- `/tmp/act_mps_act_train/policy_epoch_1921_seed_0.ckpt`
- Periodic epoch checkpoints (`policy_epoch_*.ckpt`)

Plots:
- `/tmp/act_mps_act_train/train_val_loss_seed_0.png`
- `/tmp/act_mps_act_train/train_val_l1_seed_0.png`
- `/tmp/act_mps_act_train/train_val_kl_seed_0.png`

## 7. Evaluation (50 Rollouts)
Command:

```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
conda run --no-capture-output -n aloha python -u imitate_episodes.py \
  --eval \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps
```

Results (`policy_best.ckpt`):
- Success rate: **0.88** (44/50)
- Average return: **518.04**
- Reward>=4: **44/50 (88.0%)**

Artifacts:
- Metrics file: `/tmp/act_mps_act_train/result_policy_best.txt`
- Rollout videos: `/tmp/act_mps_act_train/video0.mp4` ... `/tmp/act_mps_act_train/video49.mp4`

## 8. Interpretation
- MPS training/evaluation path is functional end-to-end.
- Performance is close to README reference level for transfer cube (~90%).
- Main operational risk observed was checkpoint write robustness; resume support mitigated this successfully.
