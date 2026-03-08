# ACT Post-Training RL Reproducibility Checklist

## Required environment
- [ ] `ACT_DATA_DIR` set to the intended dataset root
- [ ] `PYTHONPATH` includes both repo root and `detr`
- [ ] `MPLCONFIGDIR` set for non-interactive runs
- [ ] Backend recorded (`mps`, `cuda`, or `cpu`)

## Required baseline artifacts
- [ ] BC checkpoint path recorded
- [ ] `dataset_stats.pkl` path recorded
- [ ] rollout-best vs validation-best checkpoint choice recorded

## Rollout collection
- [ ] task name recorded
- [ ] rollout count / seed groups recorded
- [ ] rollouts per seed recorded
- [ ] `seed_start` recorded
- [ ] temporal aggregation setting recorded
- [ ] checkpoint provenance recorded

## Preference generation
- [ ] preference output path recorded
- [ ] pair count recorded
- [ ] preference header source checkpoint recorded
- [ ] ranking rule recorded (`success_first_return_tiebreak`)
- [ ] window length recorded

## Phase 1 / DPO
- [ ] init checkpoint recorded
- [ ] reference checkpoint recorded
- [ ] DPO output directory recorded
- [ ] `beta` recorded
- [ ] optimizer LR recorded
- [ ] max steps recorded

## Phase 2 / PPO-like RL
- [ ] init checkpoint recorded
- [ ] PPO output directory recorded
- [ ] rollout count per update recorded
- [ ] update count recorded
- [ ] max timesteps recorded
- [ ] clip/value/entropy coefficients recorded

## Evaluation
- [ ] fixed rollout count recorded
- [ ] `seed_start` recorded
- [ ] temporal aggregation setting recorded
- [ ] success rate recorded
- [ ] average return recorded
- [ ] comparison against BC baseline recorded

## Reporting
- [ ] experiment report created
- [ ] artifact locations listed
- [ ] known deviations / failures documented
