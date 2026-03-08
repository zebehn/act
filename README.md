# ACT: Action Chunking with Transformers

## Fork-Specific Information (zebehn/act)
This fork adds Apple Silicon support and experiment documentation on top of the original ACT repository.

- Device support: `--device auto|mps|cuda|cpu` (`auto` prefers MPS on macOS).
- Resume support: `--resume_ckpt auto` to continue from latest valid checkpoint.
- Full macOS setup + run guide: [docs/MACOS_APPLE_SILICON_GUIDE.md](docs/MACOS_APPLE_SILICON_GUIDE.md)
- Experiment reports: [transfer cube ACT on MPS](docs/reports/EXPERIMENT_REPORT_MPS_ACT.md), [insertion ACT on MPS](docs/reports/EXPERIMENT_REPORT_MPS_INSERTION_ACT.md), [post-training RL pilot on transfer cube](docs/reports/EXPERIMENT_REPORT_ACT_POSTTRAIN_TRANSFER_CUBE.md)

### Quickstart (Apple Silicon)
```bash
# 1) Setup env
conda create -n aloha python=3.10 -y
conda activate aloha
python -m pip install --upgrade pip
python -m pip install torch torchvision pyquaternion pyyaml rospkg pexpect \
  mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython scipy gdown
python -m pip install -e detr

# 2) Download 50-episode sim dataset
mkdir -p data
python -m gdown --folder --remaining-ok \
  "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj" \
  -O data/sim_transfer_cube_scripted

# 3) Train on MPS
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps

# 4) Evaluate + save rollout videos (temporal aggregation recommended)
ACT_DATA_DIR="$(pwd)/data" PYTHONPATH="$(pwd)/detr" MPLCONFIGDIR=/tmp/matplotlib \
python imitate_episodes.py --eval --task_name sim_transfer_cube_scripted --ckpt_dir /tmp/act_mps_act_train \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --seed 0 --device mps --temporal_agg
```

### Rollout-best checkpoint workflow (recommended)
In this fork, `policy_best.ckpt` is selected by **offline validation loss**, but the best simulator rollout checkpoint can be different. The rollout-best helper workflow now works for both `sim_transfer_cube_scripted` and `sim_insertion_scripted` while keeping the existing transfer-cube entry points available as shortcuts.

Generic helpers:
- `scripts/train_act_mps_with_rollout_best.sh` -- train ACT for `TASK_NAME`, then sweep checkpoints with temporal aggregation and preserve the rollout-best checkpoint.
- `scripts/eval_checkpoint_sweep_temporal_agg.sh` -- compare multiple saved checkpoints for `TASK_NAME` by rollout success/return.
- `scripts/eval_rollout_best_temporal_agg.sh` -- evaluate the current rollout-best checkpoint for `TASK_NAME`.

Transfer-cube compatibility wrappers:
- `scripts/train_transfer_cube_act_mps_with_rollout_best.sh`
- `scripts/eval_checkpoint_sweep_transfer_cube_temporal_agg.sh`
- `scripts/eval_rollout_best_transfer_cube_temporal_agg.sh`

Key environment variables:
- `TASK_NAME` -- simulator task to run, e.g. `sim_transfer_cube_scripted` or `sim_insertion_scripted`
- `CKPT_DIR` / `SOURCE_CKPT_DIR` -- checkpoint directory (recommended to keep separate per task)
- `SAVE_VIDEOS=0` -- skip rollout video generation during sweeps/eval for faster turnaround
- `DRY_RUN=1` -- print the generated commands without running training/evaluation

Useful eval flags in `imitate_episodes.py`:
- `--eval_ckpt_names ...` -- evaluate one or more specific checkpoint filenames inside `--ckpt_dir`
- `--no_save_episode` -- skip rollout video generation during sweeps for faster turnaround
- `--temporal_agg` -- enable temporal ensembling during evaluation

Examples:
```bash
# Transfer cube: existing shortcut entry points still work
SAVE_VIDEOS=0 bash scripts/train_transfer_cube_act_mps_with_rollout_best.sh
SAVE_VIDEOS=0 bash scripts/eval_checkpoint_sweep_transfer_cube_temporal_agg.sh
SAVE_VIDEOS=0 bash scripts/eval_rollout_best_transfer_cube_temporal_agg.sh

# Insertion: use the generic task-parameterized workflow
TASK_NAME=sim_insertion_scripted CKPT_DIR=/tmp/act_insertion_mps_act_train SAVE_VIDEOS=0 bash scripts/train_act_mps_with_rollout_best.sh

TASK_NAME=sim_insertion_scripted SOURCE_CKPT_DIR=/tmp/act_insertion_mps_act_train SAVE_VIDEOS=0 bash scripts/eval_checkpoint_sweep_temporal_agg.sh

TASK_NAME=sim_insertion_scripted SOURCE_CKPT_DIR=/tmp/act_insertion_mps_act_train SAVE_VIDEOS=0 bash scripts/eval_rollout_best_temporal_agg.sh
```

The sweep helper maintains a convenience symlink when available:
- `<ckpt_dir>/policy_rollout_best.ckpt`

If `SOURCE_CKPT_NAME` is not set, the rollout-best eval helper prefers `<ckpt_dir>/policy_rollout_best.ckpt` automatically and falls back to other common checkpoint names when needed.

For benchmark-style simulator rollouts, prefer evaluating the rollout-best checkpoint with `--temporal_agg` rather than assuming the minimum-validation-loss checkpoint is best in closed-loop control.

### Post-training RL extension (simulator-only, ACT-only)
This fork now includes a documented pilot post-training pipeline for ACT:
- Phase 1: DPO-style preference optimization from simulator rollouts
- Phase 2: PPO-like RL fine-tuning

Primary docs:
- `docs/posttraining-act-rl.md`
- `docs/posttraining-act-rl-usage.md`
- `docs/posttraining-act-rl-troubleshooting.md`
- `docs/posttraining-act-rl-reproducibility.md`
- `docs/posttraining-act-rl-schema.md`
- `docs/reports/EXPERIMENT_REPORT_ACT_POSTTRAIN_TRANSFER_CUBE.md`

Main scripts:
- `scripts/collect_posttrain_rollouts.py`
- `scripts/build_posttrain_preferences.py`
- `scripts/train_posttrain_dpo.py`
- `scripts/train_posttrain_ppo.py`
- `scripts/evaluate_posttrain_checkpoint.py`
- `scripts/eval_posttrain_sweep.sh`

Smoke checks:
- `scripts/posttrain_regression_smoke_tests.py`
- `scripts/posttrain_cli_smoke.sh`
- `scripts/posttrain_local_smoke.sh`

---
## Original README (ACT)
### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --device auto \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.
`--device auto` prefers MPS on Apple Silicon, then CUDA, then CPU. You can force `--device mps`.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.
