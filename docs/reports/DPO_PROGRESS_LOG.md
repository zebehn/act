# DPO Progress Log

## Related Diagrams
- Diagram index: `docs/specs/diagrams/README.md`
- ACT internal model: `docs/specs/diagrams/act_model_structure.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_model_structure.puml)
- Scoring comparison: `docs/specs/diagrams/act_dpo_scoring_comparison.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_dpo_scoring_comparison.puml)
- Training pipeline: `docs/specs/diagrams/act_variational_dpo_pipeline.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_pipeline.puml)

> Note: rendered links resolve once branch `act-variational-dpo-redesign` is pushed to GitHub.

## Scope
This log tracks iterative work on improving the DPO-based post-training pipeline for ACT on `sim_transfer_cube_scripted`.

## Baseline Checks
### Original report target
- Reported BC result in `docs/reports/EXPERIMENT_REPORT_MPS_ACT.md`: **0.88 success** / **518.04 avg return** on 50 rollouts using `policy_best.ckpt`.
- That original report command did **not** use `--temporal_agg`.

### Current reproducible BC baselines
Re-run on March 8, 2026 in the current environment:
- Non-temporal-agg BC (`policy_best.ckpt`): **0.48 success**, **312.54 avg return**
  - Artifact: `/tmp/act_mps_act_train/result_policy_best.txt`
- Temporal-agg BC (`policy_best.ckpt`, `--temporal_agg`): **0.54 success**, **378.86 avg return**
  - Artifact overwritten in the same checkpoint dir by the latest eval command.

Observation:
- In the current runtime, temporal aggregation improves the BC control from `0.48` to `0.54`.
- The current code/runtime does not reproduce the earlier `0.88` report.

## Root Cause Found in Rollout Collection
Initial DPO preference rollouts were collapsing because the post-training collector sampled from the newly added Gaussian action head even when loading a legacy BC checkpoint that lacked trained `action_log_std` and `value_head` weights.

Symptoms:
- First larger non-temporal collection produced **all-failure** buckets.
- DPO trained on those preferences consistently regressed badly.

## Implementation Change
Patched `posttrain/rollouts.py` and `scripts/collect_posttrain_rollouts.py`:
- detect legacy BC checkpoints by checking missing keys from `load_state_dict`
- add `sampling_strategy=auto|model|mean_gaussian|deterministic`
- default `auto` to `mean_gaussian` for legacy BC checkpoints
- add `exploration_std` control
- increase collection defaults from `4x2` to `24x4`
- store sampling metadata in `collection_meta.json`

## Experiment 1: Larger Non-Temporal Collection + DPO
### Rollout collection
- Artifact root: `/tmp/act_posttrain_policybest_rollouts_20260308T201035`
- Settings: 24 seed groups × 4 rollouts, `sampling_strategy=mean_gaussian`, `exploration_std=0.05`
- Result: **96 rollouts**, **25 successes**

### Preference build
- Preference file: `/tmp/act_posttrain_policybest_prefs_20260308T201035.jsonl`
- Result: **24 pairs**, **13 chosen-success pairs**

### DPO run
- Output dir: `/tmp/act_posttrain_policybest_dpo_20260308T201035`
- Settings: `max_steps=24`

### Evaluation
- `policy_pref_best.ckpt`: **0.14 success**, **85.46 avg return**
- `policy_pref_last.ckpt`: **0.16 success**, **96.3 avg return**

Conclusion:
- Better preference diversity alone did **not** make DPO outperform BC without temporal aggregation.

## Experiment 2: Temporal Aggregation Everywhere
### BC control
- Temporal-agg BC baseline: **0.54 success**, **378.86 avg return**

### Rollout collection
- Artifact root: `/tmp/act_posttrain_policybest_rollouts_tagg_20260308T211853`
- Settings: 24 seed groups × 4 rollouts, `--temporal_agg`, `sampling_strategy=mean_gaussian`, `exploration_std=0.05`
- Result: **96 rollouts**, **21 successes**, **88.48 avg return**

### Preference build
- Preference file: `/tmp/act_posttrain_policybest_prefs_tagg_20260308T211853.jsonl`
- Result: **24 pairs**, **10 chosen-success pairs**

### DPO run
- Output dir: `/tmp/act_posttrain_policybest_dpo_tagg_20260308T211853`
- Started with `max_steps=24`, `--temporal_agg`
- Runtime was much slower than expected (~2 minutes per step)
- Stopped manually after the first checkpoint boundary at step 4
- Partial metrics showed very large-magnitude trajectory log-probs and unstable margins

### Partial temporal-agg DPO evaluation
- `policy_pref_best.ckpt`: currently evaluated at **0.52 success**, **248.18 avg return** on 50 rollouts
- `policy_pref_last.ckpt`: evaluation was still running when this log entry was written

Interim conclusion:
- Temporal aggregation plus the rollout-collection fix produces the **first DPO run that is close to BC instead of collapsing**.
- The partial temporal-agg DPO result (`0.52`) is still slightly below the temporal-agg BC control (`0.54`), but it is dramatically better than the earlier non-temporal DPO runs.

## Current Read on the Problem
The collector bug was real and fixing it materially improved the DPO pipeline. However, DPO training still appears brittle. Likely remaining issues:
1. **Trajectory log-prob scale is too large**, causing unstable margins and huge losses.
2. **Window length 32 may still be too long** for stable preference optimization.
3. **Learning rate may be too high** for this objective.
4. **Preference quality remains mixed** because many seed groups are still all-failure or low-return.
5. **Temporal aggregation in DPO replay is expensive**, making iteration slow.

## Recommended Next Iterations
1. Lower DPO learning rate (e.g. `1e-6` or `3e-6`).
2. Reduce `window_length` from `32` to `8` or `16`.
3. Normalize DPO objective by window length or action dimension.
4. Shuffle/randomize pair order each epoch instead of cycling deterministically.
5. Add a filter to skip low-information groups where chosen and rejected returns are too close.
6. Evaluate whether starting from `policy_rollout_best.ckpt` performs better than `policy_best.ckpt` under temporal aggregation.

## Key Artifacts
- BC baseline: `/tmp/act_mps_act_train/result_policy_best.txt`
- Non-temporal richer rollouts: `/tmp/act_posttrain_policybest_rollouts_20260308T201035`
- Non-temporal richer DPO: `/tmp/act_posttrain_policybest_dpo_20260308T201035`
- Temporal rollouts: `/tmp/act_posttrain_policybest_rollouts_tagg_20260308T211853`
- Temporal DPO: `/tmp/act_posttrain_policybest_dpo_tagg_20260308T211853`


## Implementation Update: DPO Log-Prob Normalization
Added a stability-oriented change after observing huge-magnitude trajectory log-probs under temporal aggregation:
- `trajectory_logprob(...)` now also returns `mean_log_prob`, `mean_entropy`, and `token_count`
- `train_posttrain_dpo.py` / `posttrain.dpo` now support `--logprob_reduction {sum,mean}`
- default is now `mean`

Rationale:
- the previous DPO objective used the sum of per-action log-probs over the whole trajectory window
- with `window_length=32` and 14-dim actions, those values became very large in magnitude
- normalizing by token count should make the margin scale less brittle and should make `beta` and `lr` easier to tune

Next planned experiment:
- rerun temporal-agg DPO with:
  - `--logprob_reduction mean`
  - lower `--lr`
  - shorter or cheaper checkpoint cadence if needed


## Experiment 3: Temporal Aggregation + Normalized DPO + Lower LR
### Config
- Preference file: `/tmp/act_posttrain_policybest_prefs_tagg_20260308T211853.jsonl`
- DPO output dir: `/tmp/act_posttrain_policybest_dpo_tagg_meanlr_20260308T222242`
- Flags:
  - `--temporal_agg`
  - `--logprob_reduction mean`
  - `--lr 3e-6`
  - `--max_steps 8`
  - `--checkpoint_interval 4`

### Training behavior
This run was much more numerically stable than the previous temporal-agg DPO attempt.
- Margins stayed near zero to moderately positive/negative instead of exploding.
- Best training loss: **0.4946** at step **3**.
- Later steps remained in a stable range instead of diverging catastrophically.

### Evaluation
- `policy_pref_best.ckpt`: **0.92 success**, **648.06 avg return** on 50 rollouts with `--temporal_agg`
  - Artifact: `/tmp/act_posttrain_policybest_dpo_tagg_meanlr_20260308T222242/result_policy_pref_best.txt`

### Comparison
- Current temporal-agg BC baseline: **0.54 success**, **378.86 avg return**
- Normalized temporal-agg DPO best: **0.92 success**, **648.06 avg return**
- Absolute gain: **+0.38 success rate**

### Takeaway
This is the first DPO configuration in the current runtime that clearly improves over BC. The combination that seems to matter is:
1. rollout collection fix for legacy BC checkpoints
2. larger rollout/preference budget
3. temporal aggregation enabled throughout
4. normalized (`mean`) DPO log-probs
5. lower DPO learning rate

### Immediate next follow-ups
1. Evaluate `policy_pref_last.ckpt` from the same run.
2. Re-run the same configuration with a second seed range to check robustness.
3. Try `policy_rollout_best.ckpt` as the BC starting checkpoint with the same stabilized DPO settings.
4. Try `window_length=16` to see whether training can remain strong with lower replay cost.


## Experiment 4: Transfer-Cube Robustness on Second Seed Range
### Config
- Rollout root: `/tmp/act_posttrain_policybest_rollouts_tagg_20260309T004624`
- Preference file: `/tmp/act_posttrain_policybest_prefs_tagg_20260309T004624.jsonl`
- DPO output dir: `/tmp/act_posttrain_policybest_dpo_tagg_meanlr_20260309T004624`
- Same stabilized settings as Experiment 3, but collection used `seed_start=5000`.

### Collection quality
- 96 rollouts
- 29 successes
- 24 preference pairs
- 14 chosen-success pairs
- 7 rejected-success pairs

### Evaluation
- `policy_pref_best.ckpt`: **0.40 success**, **293.66 avg return** on 50-rollout temporal-agg eval
  - Artifact: `/tmp/act_posttrain_policybest_dpo_tagg_meanlr_20260309T004624/result_policy_pref_best.txt`

### Robustness read
This strong regression relative to the Experiment-3 result (`0.92`) indicates the current stabilized DPO recipe is **not yet robust across seed ranges**. It can work very well, but it is still sensitive to which rollout-preference dataset is collected.

## Experiment 5: Insertion Task with Stabilized DPO Flow
### BC temporal-agg baseline
- Checkpoint: `/tmp/act_insertion_mps_act_train/policy_rollout_best.ckpt`
- Eval result: **0.58 success**, **418.8 avg return**
- Artifact: baseline metrics saved under `/tmp/act_insertion_mps_act_train/result_policy_rollout_best.txt` during the current rerun

### Collector adaptation for insertion
Insertion was more fragile under exploration than transfer cube. The first insertion rollout attempt with fully exploratory candidates produced zero early successes, so the collector was updated to support `--deterministic_candidates N`.

For the successful insertion collection run:
- rollout root: `/tmp/act_posttrain_insertion_rollouts_tagg_20260309T015909`
- config included `--deterministic_candidates 1`
- result: 96 rollouts, 13 successes

### Preference build
- preference file: `/tmp/act_posttrain_insertion_prefs_tagg_20260309T015909.jsonl`
- 24 pairs
- 13 chosen-success pairs
- 0 rejected-success pairs

### DPO run
- output dir: `/tmp/act_posttrain_insertion_dpo_tagg_meanlr_20260309T015909`
- settings matched the stabilized recipe:
  - `--temporal_agg`
  - `--logprob_reduction mean`
  - `--lr 3e-6`
  - `--max_steps 8`

### Evaluation
- `policy_pref_best.ckpt`: **0.60 success**, **440.92 avg return**
  - Artifact: `/tmp/act_posttrain_insertion_dpo_tagg_meanlr_20260309T015909/result_policy_pref_best.txt`

### Comparison
- Insertion BC temporal-agg baseline: **0.58** success / **418.8** avg return
- Insertion DPO best: **0.60** success / **440.92** avg return
- Absolute gain: **+0.02 success rate**

### Updated takeaways
1. The stabilized DPO recipe can produce **large gains** on transfer cube, but current robustness is limited.
2. The same recipe gives a **small but real improvement** on insertion when the collector includes deterministic anchor candidates.
3. Seed-range sensitivity is still the main unresolved issue for transfer-cube DPO.
4. Insertion likely benefits from a more conservative exploration schedule than transfer cube.

### Recommended next experiments
1. Add deterministic anchor candidates to transfer cube too, then repeat the second-seed robustness run.
2. Try `window_length=16` for both transfer and insertion to reduce replay cost and possibly improve robustness.
3. Evaluate `policy_pref_last.ckpt` for insertion to see whether later steps outperform the best-loss checkpoint.
4. Repeat the insertion DPO run on a second seed range to test whether the +0.02 gain is stable.


## Experiment 6: End-to-End Validation of Variational Chunk DPO
### Purpose
Validate the newly implemented `variational_chunk_elbo` score mode end-to-end on transfer cube.

### Config
- preference file: `/tmp/act_posttrain_policybest_prefs_tagg_20260308T211853.jsonl`
- DPO output dir: `/tmp/act_variational_dpo_transfer_20260309T085910`
- flags:
  - `--score_mode variational_chunk_elbo`
  - `--posterior_decode_mode mean`
  - `--variational_kl_coef 1.0`
  - `--logprob_reduction mean`
  - `--lr 3e-6`
  - `--max_steps 4`
  - `--temporal_agg`

### Training behavior
- Runtime was much faster than the earlier replay-heavy temporal-agg DPO variants.
- Loss/margin stayed finite and training completed cleanly.
- This confirms the new score mode is operational end-to-end.

### Evaluation
- `policy_pref_best.ckpt`: **0.64 success**, **444.06 avg return**
  - Artifact: `/tmp/act_variational_dpo_transfer_20260309T085910/result_policy_pref_best.txt`

### Comparison
- BC temporal-agg baseline: **0.54 success**, **378.86 avg return**
- Variational DPO best: **0.64 success**, **444.06 avg return**
- Absolute gain: **+0.10 success rate**

### Interpretation
This validates the revised implementation direction:
- the new latent-aware variational score mode is implementable,
- it completes end-to-end training and evaluation,
- it improves over the current BC baseline on transfer cube,
- but it does not yet match the best surrogate-DPO run (`0.92`), so further tuning remains necessary.


## Experiment 7: End-to-End Validation of Variational Chunk DPO on Insertion
### Purpose
Validate the new `variational_chunk_elbo` score mode end-to-end on `sim_insertion_scripted`.

### Config
- preference file: `/tmp/act_posttrain_insertion_prefs_tagg_20260309T015909.jsonl`
- DPO output dir: `/tmp/act_variational_dpo_insertion_20260309T091248`
- flags:
  - `--score_mode variational_chunk_elbo`
  - `--posterior_decode_mode mean`
  - `--variational_kl_coef 1.0`
  - `--logprob_reduction mean`
  - `--lr 3e-6`
  - `--max_steps 4`
  - `--temporal_agg`

### Evaluation
- `policy_pref_best.ckpt`: **0.58 success**, **393.76 avg return**
  - Artifact: `/tmp/act_variational_dpo_insertion_20260309T091248/result_policy_pref_best.txt`

### Comparison
- Insertion BC temporal-agg baseline: **0.58 success**, **418.8 avg return**
- Variational insertion DPO best: **0.58 success**, **393.76 avg return**

### Interpretation
The variational score mode successfully completed end-to-end on insertion and matched the BC success rate, but it did not improve average return and did not outperform the stronger surrogate Gaussian-replay DPO variant. This suggests the variational formulation is viable, but insertion likely needs task-specific tuning (e.g. shorter windows, BC regularization, or deterministic anchor-aware preference filtering).
