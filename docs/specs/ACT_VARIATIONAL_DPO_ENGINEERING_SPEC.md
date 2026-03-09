# ACT Variational DPO Engineering Specification

## Status
Draft for implementation on branch `act-variational-dpo-redesign`.

## Diagrams
- Diagram index: `docs/specs/diagrams/README.md`
- ACT internal model diagram: `docs/specs/diagrams/act_model_structure.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_model_structure.puml)
- Architecture diagram: `docs/specs/diagrams/act_variational_dpo_architecture.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_architecture.puml)
- Training pipeline diagram: `docs/specs/diagrams/act_variational_dpo_pipeline.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_pipeline.puml)
- Scoring sequence diagram: `docs/specs/diagrams/act_variational_dpo_sequence.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_sequence.puml)
- Surrogate vs variational scoring comparison: `docs/specs/diagrams/act_dpo_scoring_comparison.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_dpo_scoring_comparison.puml)

> Note: the rendered links use the current branch name (`act-variational-dpo-redesign`) and work once that branch is pushed to GitHub.

## Terminology
- **Surrogate Gaussian-replay DPO**: the earlier DPO-style method that scores replayed chunks with inference-time Gaussian action log-probs.
- **Variational DPO**: the revised preference objective family grounded in ACT's training-time latent path.
- **`variational_chunk_elbo`**: the implemented score mode name for the current variational chunk-scoring path.
- **Deterministic anchor candidates**: one or more low-noise rollout candidates collected alongside exploratory variants within each seed group.
- **BC regularization**: an optional auxiliary term that keeps preference fine-tuning close to ACT's behavioral cloning objective.

## Goal
Replace the current surrogate Gaussian-replay DPO replay score with a latent-aware ACT chunk score that is closer to a principled variational preference objective for continuous action chunks.

## Non-Goals
- Full exact marginal likelihood estimation with expensive multi-sample integration
- Major ACT architecture redesign beyond what is needed for variational chunk scoring
- PPO redesign in this phase

## Problem Statement
The current post-training pipeline computes chosen/rejected scores from inference-time Gaussian chunk replay. This ignores ACT's latent-variable training path and does not tightly connect to the model's chunk likelihood. We need a score for preference optimization that:
1. uses ACT's encoder/latent pathway,
2. assigns a continuous likelihood-like score to action chunks,
3. supports chosen/rejected chunk-window comparisons,
4. remains computationally tractable in simulator replay.

## High-Level Design
### Scoring modes
Introduce explicit DPO score modes:
- `gaussian_replay`: existing surrogate replay path
- `variational_chunk_elbo`: new latent-aware path

### Variational chunk score
For each chunk start timestep `t`, construct:
- observation context `o_t`
- normalized action chunk `a_t^(H)`
- padding mask for chunk tail if window is short

Use ACT's training-time path with provided actions to obtain:
- posterior parameters `mu, logvar`
- decoder output chunk mean `a_hat`
- decoder action std/log-std

Define chunk score:
- decoder Gaussian log-likelihood of target actions under `a_hat`
- minus latent KL term against the ACT prior

Aggregate chunk scores across a chosen/rejected window by sum or mean.

## Required Code Changes
### `detr/models/detr_vae.py`
Add model support to expose training-path auxiliaries needed for scoring:
- posterior `mu`, `logvar`
- decoder output `a_hat`
- decoder action variance/log-std
- optional latent sample or posterior mean decode mode

### `policy.py`
Add ACTPolicy helpers for chunk scoring, e.g.:
- `score_action_chunk(...)`
- support deterministic posterior-mean scoring mode to reduce Monte Carlo noise

### `posttrain/rollouts.py`
Add chunk extraction utilities:
- build normalized chunk tensors from rollout arrays
- build padding masks
- expose reusable window iteration for preference scoring

### `posttrain/dpo.py`
Add:
- `--score_mode`
- `--bc_reg_coef`
- `--posterior_decode_mode sample|mean`
- `--window_length_override` if needed for experimentation
- variational chosen/rejected score computation
- optional hybrid objective: `variational DPO + BC regularization`

### `scripts/train_posttrain_dpo.py`
Pass through new CLI args.

## Data / Artifact Contracts
### Existing preference files remain valid
No schema changes required for chosen/rejected metadata.

### New training artifacts
DPO config should record:
- `score_mode`
- `posterior_decode_mode`
- `bc_reg_coef`
- score reduction settings

## Algorithms
### Variational chunk ELBO score
For chunk `(o, a)`:
1. run ACT training-time forward with `actions=a`
2. obtain `a_hat`, `mu`, `logvar`, `action_log_std`
3. compute masked Gaussian log-likelihood
4. compute KL to prior
5. return `chunk_score = log_likelihood - kl_coef * kl`

Default `kl_coef` for scoring phase should be configurable and initially set to `1.0`.

### Window score
For chosen/rejected windows:
- collect chunk scores over timesteps in the window
- aggregate with `mean` by default for stability

### Preference objective
Default objective:
- DPO logistic margin on window scores
Optional stabilization:
- add BC regularization on chosen chunks only
- objective: `loss = dpo_loss + bc_reg_coef * (-chosen_score_mean)`

## Initial Experimental Matrix
### Transfer cube
- baseline checkpoint: `/tmp/act_mps_act_train/policy_best.ckpt`
- score mode: `variational_chunk_elbo`
- temporal aggregation: on
- collection seeds: `3000`, `5000`
- DPO lr: `3e-6`
- max steps: `8`
- window length: `32`, then `16`

### Insertion
- baseline checkpoint: `/tmp/act_insertion_mps_act_train/policy_rollout_best.ckpt`
- deterministic anchor candidates: `1`
- same DPO score mode and optimizer defaults

## Validation Criteria
Success criteria for this phase:
1. End-to-end DPO run completes with `variational_chunk_elbo`
2. Score magnitudes remain numerically stable
3. Transfer-cube best run matches or exceeds temporal-agg BC baseline
4. Insertion run matches or exceeds temporal-agg BC baseline
5. Robustness on a second transfer seed range improves over the prior `0.40` failure case

## Risks
- Training-time ACT scoring may be much slower than replay scoring
- Posterior-sampled chunk scores may add noise
- KL scaling may dominate if not normalized carefully
- Need to preserve backward compatibility with existing Gaussian replay mode

## Rollout / Evaluation Protocol
Use the existing 50-rollout simulator evaluator for final comparison:
- transfer cube: `imitate_episodes.py --eval --temporal_agg`
- insertion: `imitate_episodes.py --eval --temporal_agg`

## Exit Conditions
This specification is considered implemented when:
- the new score mode is live,
- at least one transfer and one insertion end-to-end experiment complete,
- results are logged in `docs/reports/DPO_PROGRESS_LOG.md`.
