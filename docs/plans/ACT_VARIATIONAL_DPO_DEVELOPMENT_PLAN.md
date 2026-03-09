# ACT Variational DPO Development Plan

## Related Diagrams
- Diagram index: `docs/specs/diagrams/README.md`
- ACT internal model: `docs/specs/diagrams/act_model_structure.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_model_structure.puml)
- Scoring comparison: `docs/specs/diagrams/act_dpo_scoring_comparison.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_dpo_scoring_comparison.puml)
- Training pipeline: `docs/specs/diagrams/act_variational_dpo_pipeline.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_pipeline.puml)
- Scoring sequence: `docs/specs/diagrams/act_variational_dpo_sequence.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_sequence.puml)

> Note: rendered links resolve once branch `act-variational-dpo-redesign` is pushed to GitHub.

## Terminology
This plan uses the following names consistently:
- **Surrogate Gaussian-replay DPO** for the earlier replay-log-prob method
- **Variational DPO** for the revised latent-aware objective
- **`variational_chunk_elbo`** for the implemented variational score mode
- **Deterministic anchor candidates** for per-seed low-noise rollout anchors

## Phase 1 — Design Freeze
- Preserve the reformulation memo as the design-history record.
- Implement the engineering spec without removing the existing surrogate Gaussian-replay DPO path.

## Phase 2 — Core Implementation
1. Add variational chunk scoring utilities to ACT.
2. Add DPO score-mode plumbing and CLI flags.
3. Add optional BC regularization for preference fine-tuning.
4. Keep the current Gaussian replay mode as a fallback.

## Phase 3 — Sanity Validation
1. `py_compile` modified modules.
2. Run a smoke DPO step on transfer cube.
3. Verify artifacts and metrics fields are written.

## Phase 4 — End-to-End Transfer Experiment
1. Collect preferences on transfer cube with temporal aggregation.
2. Train DPO with `variational_chunk_elbo` on seed range `3000`.
3. Evaluate on the standard 50-rollout transfer protocol.
4. Repeat on seed range `5000` for robustness.

## Phase 5 — End-to-End Insertion Experiment
1. Reproduce insertion temporal-agg BC baseline.
2. Collect insertion preferences with deterministic anchors.
3. Train insertion DPO with `variational_chunk_elbo`.
4. Evaluate on the standard 50-rollout insertion protocol.

## Phase 6 — Assessment
- Compare against BC baselines.
- Record strengths, failures, and sensitivity to seed ranges.
- Recommend the next iteration.
