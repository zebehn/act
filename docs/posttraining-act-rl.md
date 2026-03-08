# ACT Post-Training with Preference Learning and RL

## Overview
This codebase now includes a simulator-only post-training stack for ACT that extends the existing behavior-cloning workflow with two additional optimization phases:

1. **Phase 1 — DPO-style preference optimization**
   - Build chosen/rejected preference pairs from closed-loop simulator rollouts.
   - Fine-tune the ACT policy against a frozen reference checkpoint.
2. **Phase 2 — PPO-like RL fine-tuning**
   - Start from the Phase-1 checkpoint.
   - Collect on-policy simulator rollouts and optimize a clipped policy/value objective.

## Locked v1 Scope
- Simulator-only
- Pilot task: `sim_transfer_cube_scripted`
- ACT-only (no CNNMLP post-training path)
- Preference ranking rule: **success first, return tie-breaker**
- Default starting checkpoint: `policy_rollout_best.ckpt`
- `policy_best.ckpt` supported as an ablation start point
- Temporal aggregation configurable, default ON for rollout collection/evaluation
- New rollout/preference/RL artifact schema, separate from demo HDF5s
- Comprehensive documentation required

## Architectural Rationale
The original BC path remains the source of truth for imitation-learning training and evaluation. Post-training is implemented as an additive stack rather than a replacement path.

### Existing BC/Eval surfaces
- `imitate_episodes.py` — BC train/eval orchestration
- `policy.py` — ACT/CNNMLP adapters
- `detr/models/detr_vae.py` — ACT model core
- `utils.py` — demo dataset loading and normalization
- `sim_env.py` — simulator reward and success logic

### Post-training extensions
- `posttrain/common.py`
  - build/load ACT policies and checkpoints
  - common config and serialization helpers
- `posttrain/schema.py`
  - rollout + preference artifact schema helpers
- `posttrain/rollouts.py`
  - simulator rollout collection
  - temporal aggregation distribution logic
  - trajectory log-prob replay for DPO
- `posttrain/preferences.py`
  - deterministic chosen/rejected pair construction
- `posttrain/dpo.py`
  - Phase-1 trainer
- `posttrain/ppo.py`
  - Phase-2 PPO-like trainer
- `posttrain/eval.py`
  - fixed-protocol checkpoint evaluation helper

## Policy/Model Changes
### `detr/models/detr_vae.py`
Added post-training signals to the ACT model:
- `value_head`
- `action_log_std`
- optional `return_aux=True` path returning:
  - hidden states
  - state value
  - broadcast action log-std

The default BC path still returns the original ACT outputs when `return_aux=False`.

### `policy.py`
`ACTPolicy` remains BC-compatible but now also exposes:
- `get_action_distribution(...)`
- `sample_action(...)`
- `evaluate_action(...)`

These methods are used by the post-training stack while the original `__call__` behavior remains intact for BC.

### Checkpoint compatibility
Older BC checkpoints do not contain the new `value_head` and `action_log_std` parameters. Compatibility is preserved by allowing those specific keys to be missing when loading legacy checkpoints into the updated ACT policy.

## Temporal Aggregation
The post-training stack reuses the repo’s temporal aggregation idea during rollout collection/evaluation. Instead of using only the current query prediction, it can aggregate overlapping action predictions across timesteps using exponentially decayed weights.

The rollout collector stores whether temporal aggregation was enabled in artifact metadata so collection/evaluation settings stay traceable.

## DPO Design
### Preference unit
For v1, the trainer uses **trajectory-level ranking with fixed window extraction**:
- rollouts are paired by shared seed / initial state bucket
- success is the primary preference key
- return breaks ties inside the same success class
- the trainer uses a deterministic fixed window from each chosen/rejected trajectory pair to keep optimization tractable

### DPO objective
The trainer replays chosen and rejected trajectories to compute trajectory-window log-probs under:
- the trainable policy
- the frozen reference policy

Then applies a DPO-style objective to increase preference-consistent likelihood ratios.

## PPO-like RL Design
The PPO-like trainer:
- collects on-policy simulator rollouts
- stores observations, actions, old log-probs, rewards, dones, and values in memory
- computes GAE-style returns/advantages
- applies a clipped policy/value/entropy objective

This v1 implementation is intentionally lightweight and optimized for pilot experiments and smoke coverage rather than full-scale throughput.

## Artifact Layout
Recommended root:
- `posttrain_artifacts/<task>/...` or experiment-specific temp/output directories

Main artifact types:
- rollout collections: `<run_dir>/manifest.jsonl`, `*.json`, `*.npz`
- preference datasets: `preferences.jsonl` + `.meta.json`
- DPO outputs: `policy_pref_best.ckpt`, `policy_pref_last.ckpt`, `dpo_state.pt`, `metrics.jsonl`
- PPO outputs: `policy_ppo_best.ckpt`, `policy_ppo_last.ckpt`, `ppo_state.pt`, `metrics.jsonl`
- evaluation outputs: `result_summary.json` plus optional sweep TSV/logs

## Stage Boundaries
### BC baseline
- Existing rollout-best ACT checkpoint

### Phase 1 output
- DPO-optimized checkpoint(s)

### Phase 2 output
- PPO-like optimized checkpoint(s)

### Evaluation contract
Each stage can be compared with the same fixed simulator benchmark protocol using the post-training evaluation helper and/or rollout-best sweep tools.
