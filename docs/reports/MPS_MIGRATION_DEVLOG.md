# MPS Migration Development Log

## Objective
Convert hard CUDA dependencies to backend-agnostic device handling, with MPS-first behavior for Apple Silicon MacBook Pro environments.

Date: 2026-03-04  
Repository: `act`

## Baseline Audit
Command used:

```bash
rg -n "cuda|\\.cuda\\(|device\\s*=\\s*['\\\"]cuda|torch\\.device\\(['\\\"]cuda|to\\(['\\\"]cuda|set_device|pytorch-cuda"
```

High-impact findings:
- `imitate_episodes.py`: direct `.cuda()` calls in training and evaluation paths.
- `detr/main.py`: model construction forced with `model.cuda()`.
- `detr/util/misc.py`: distributed helpers hardcoded tensors on CUDA.
- `conda_env.yaml`: `pytorch-cuda=11.8` dependency incompatible with Apple Silicon setups.

## Design Decisions
1. Introduce a single device-resolution utility instead of scattered checks.
2. Default policy: `auto` should prefer `mps` on macOS, then `cuda`, then `cpu`.
3. Keep optional CUDA compatibility where useful (e.g., Linux training nodes), but remove CUDA hard requirements.
4. Keep CLI ergonomics by adding `--device {auto,mps,cuda,cpu}`.

## Implementation Notes
### 1) New shared device utility
File: `device_utils.py`
- Added `resolve_device(preferred="auto")`.
- Added `dataloader_pin_memory(device)` to use pinning only for CUDA.

### 2) Training/eval script conversion
File: `imitate_episodes.py`
- Added startup device resolution and runtime print (`Using device: ...`).
- Added config propagation of resolved device.
- Replaced all `.cuda()` tensor/model moves with `.to(device)`.
- Added `map_location=device` when loading checkpoints.
- Updated `forward_pass` to move each batch tensor to selected device.
- Added CLI flag: `--device`.

### 3) Model builder conversion
File: `detr/main.py`
- Added `--device` parser argument.
- Replaced `model.cuda()` with:
  - `device = resolve_device(args.device)`
  - `model.to(device)`

### 4) Data pipeline tweaks
File: `utils.py`
- Updated `load_data(...)` signature to accept `device`.
- Made `pin_memory` conditional on CUDA only.

### 5) Distributed helper hardening
File: `detr/util/misc.py`
- Replaced CUDA-hardcoded temporary tensors with runtime device fallback (`cuda` if available else `cpu`).
- Updated distributed init to:
  - use `nccl` only when CUDA exists,
  - fallback to `gloo` otherwise.

### 6) Environment specification update
File: `conda_env.yaml`
- Removed `nvidia` channel.
- Removed `pytorch-cuda=11.8`.

### 7) User docs update
File: `README.md`
- Added `--device auto` in ACT training example.
- Added note that `auto` prefers MPS on Apple Silicon.

## Validation Log
### Static checks
Command:

```bash
python -m compileall device_utils.py imitate_episodes.py utils.py detr/main.py detr/util/misc.py
```

Result:
- All files compiled successfully.

### Runtime probe
Command:

```bash
python - <<'PY'
from device_utils import resolve_device
print(resolve_device('auto'))
PY
```

Observed output in current environment:
- `cpu`

Interpretation:
- The current Python runtime does not report available MPS (likely non-MPS PyTorch build or non-macOS runtime context). The code now handles this safely by fallback.

## Risks and Follow-Ups
1. MPS operator coverage: some PyTorch ops can still fall back or error on specific versions. If encountered, force `--device cpu` for those runs.
2. Full end-to-end simulation training was not executed in this pass due runtime/dependency constraints; only static compile validation was completed.
3. If this repo is used across heterogeneous machines, keep `--device auto` as default and set explicit `--device` in automation scripts for reproducibility.

## Basic MPS Training Smoke Test (2026-03-04)
Objective: verify that training can run on MPS with a real downloaded dataset.

### Plan Executed
1. Download at least one public episode from the ACT Google Drive dataset.
2. Create a minimal task config to point training at that local data.
3. Run one-epoch training on MPS and confirm checkpoint/plots are produced.

### Commands and Outcomes
1) Install downloader:

```bash
python -m pip install gdown
```

2) Download from shared folder (interrupted after getting first usable episode to keep runtime manageable):

```bash
python -m gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link" -O data
```

Resulting local files (used for smoke test):
- `data/sim_insertion_human/episode_0.hdf5`
- `data/sim_insertion_human/episode_0_qpos.png`
- `data/sim_insertion_human/episode_0_video.mp4`

3) Environment/launch command:

```bash
ACT_DATA_DIR="$(pwd)/data" \
MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH="$(pwd)/detr" \
conda run -n aloha python imitate_episodes.py \
  --task_name sim_insertion_human_smoke \
  --ckpt_dir /tmp/act_mps_smoke \
  --policy_class CNNMLP \
  --batch_size 1 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-5 \
  --device mps
```

Observed output highlights:
- `Using device: mps`
- Completed 1/1 epoch
- `Best ckpt, val loss 0.784387 @ epoch0`
- Artifacts written to `/tmp/act_mps_smoke` (checkpoints + training plots)

### Additional Code Changes for Smoke Robustness
- `constants.py`: added `ACT_DATA_DIR` env-var support and a `sim_insertion_human_smoke` task config.
- `utils.py`: made train/val split robust for tiny datasets (fallback to reuse one episode for validation if needed).
- `imitate_episodes.py`: deferred `sim_env` import to evaluation path, allowing training-only runs without `dm_control` installed.
