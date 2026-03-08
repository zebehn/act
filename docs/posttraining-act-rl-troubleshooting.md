# ACT Post-Training RL Troubleshooting

## MPS issues
### Unsupported operator on MPS
If a post-training op fails on MPS, first check whether the implementation can use a numerically equivalent expression supported by MPS. Example encountered during development:
- DPO loss originally used `logsigmoid`, which was not available on MPS in this environment.
- Replacing it with a numerically stable `logaddexp` form avoided the backend error.

### CPU fallback warnings
A warning such as `aten::sgn.out` falling back to CPU is not fatal, but it can slow training or evaluation. Track these warnings in experiment reports.

## Legacy checkpoint loading
Older BC checkpoints do not contain post-training-only parameters such as the value head or action log-std. The updated `ACTPolicy` handles the known missing keys, but any other unexpected mismatch should be treated as a real compatibility issue.

## No preference pairs built
If preference generation creates zero pairs:
- verify rollout collection produced more than one candidate per seed
- check whether all rollouts in each seed bucket have identical success/return rank
- increase rollout diversity (more seeds or more rollouts per seed)

## Sweep script parse failures
If the generic sweep scripts fail because model-construction helpers reject new CLI flags, verify that `detr/main.py` still accepts the post-training eval flags and that `parse_known_args()` compatibility remains intact.

## Long runtimes
Post-training rollout replay is expensive because the policy must repeatedly process simulator observations and image inputs.
For smoke validation:
- reduce `num_rollouts`
- reduce `max_timesteps`
- use CPU or MPS based on stability
- keep video saving disabled unless specifically needed

## Determinism mismatches
If repeated runs produce different preference files or evaluation outputs:
- verify `seed_start` is fixed
- verify temporal aggregation setting is the same for collection and evaluation
- ensure preference pairing is using the documented success-first / return-tiebreak rule
