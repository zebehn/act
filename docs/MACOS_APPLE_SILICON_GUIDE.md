# macOS Apple Silicon Guide (MPS)

This guide documents a full local workflow for ACT on macOS Apple Silicon (M1/M2/M3), including installation, dataset download, training, and evaluation with rollout videos.

## 1. Prerequisites
- macOS on Apple Silicon
- Xcode Command Line Tools
- Conda (Miniconda or Anaconda)
- ~60GB free disk (dataset + checkpoints + videos)

Install CLI tools if needed:

```bash
xcode-select --install
```

## 2. Clone and Environment Setup
```bash
git clone https://github.com/zebehn/act.git
cd act

conda create -n aloha python=3.10 -y
conda activate aloha
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install pyquaternion pyyaml rospkg pexpect
python -m pip install mujoco==2.3.7 dm_control==1.0.14
python -m pip install opencv-python matplotlib einops packaging h5py ipython scipy gdown
python -m pip install -e detr
```

Verify MPS:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Expected: `True True` for MPS built/available.

## 3. Download Sim Dataset (50 episodes)
Use the public ACT demo folder:

```bash
mkdir -p data
python -m gdown --folder --remaining-ok \
  "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj" \
  -O data/sim_transfer_cube_scripted
```

Check count:

```bash
find data/sim_transfer_cube_scripted -maxdepth 1 -name 'episode_*.hdf5' | wc -l
```

Expected for full set: `50` (`episode_0.hdf5` ... `episode_49.hdf5`).

## 4. Train ACT on MPS
```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py \
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

You should see:

```text
Using device: mps
```

## 5. Resume Training (if interrupted)
```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps \
  --resume_ckpt auto
```

## 6. Evaluate and Generate Rollout Videos
```bash
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py \
  --eval \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT \
  --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps --temporal_agg
```

Outputs:
- Metrics: `/tmp/act_mps_act_train/result_policy_best.txt`
- Videos: `/tmp/act_mps_act_train/video0.mp4` ... `/tmp/act_mps_act_train/video49.mp4`

### Recommended: preserve a rollout-best checkpoint
In this fork, the checkpoint with the lowest offline validation loss is not always the best checkpoint for simulator rollout success. The helper scripts can automatically sweep checkpoints with temporal aggregation and preserve the best rollout checkpoint:

```bash
SAVE_VIDEOS=0 bash scripts/train_transfer_cube_act_mps_with_rollout_best.sh
```

This workflow creates a convenience symlink when available:
- `/tmp/act_mps_act_train/policy_rollout_best.ckpt`

Related reports:
- `docs/reports/EXPERIMENT_REPORT_MPS_ACT.md`
- `docs/reports/EXPERIMENT_REPORT_MPS_INSERTION_ACT.md`

## 7. Troubleshooting
- `MPS backend requested but not available`: install Apple Silicon PyTorch build and verify MPS availability.
- `ModuleNotFoundError: scipy` during eval: `python -m pip install scipy`.
- Checkpoint write failures: free disk space and resume with `--resume_ckpt auto`.
- Slow first matplotlib import: set `MPLCONFIGDIR=/tmp/matplotlib`.
