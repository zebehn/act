# Diagram Index

This folder contains PlantUML (`.puml`) diagrams for the ACT variational DPO redesign.

## Files

> Rendered PlantUML links below use the GitHub branch `act-variational-dpo-redesign`. Push the branch to GitHub if you want the proxy-rendered links to resolve remotely.

- `act_model_structure.puml` — internal ACT model structure, including the latent path used for chunk scoring. [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_model_structure.puml)
- `act_variational_dpo_architecture.puml` — high-level architecture of rollout preferences, chunk scoring, reference comparison, and variational DPO loss. [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_architecture.puml)
- `act_variational_dpo_pipeline.puml` — end-to-end training and evaluation pipeline. [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_pipeline.puml)
- `act_variational_dpo_sequence.puml` — sequence view of one chosen/rejected window being scored during variational DPO. [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_sequence.puml)
- `act_dpo_scoring_comparison.puml` — side-by-side comparison of surrogate Gaussian replay scoring vs variational chunk ELBO scoring. [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_dpo_scoring_comparison.puml)

## Suggested Reading Order
1. `act_model_structure.puml`
2. `act_dpo_scoring_comparison.puml`
3. `act_variational_dpo_architecture.puml`
4. `act_variational_dpo_sequence.puml`
5. `act_variational_dpo_pipeline.puml`

## Related Docs
- Engineering spec: `docs/specs/ACT_VARIATIONAL_DPO_ENGINEERING_SPEC.md`
- Development plan: `docs/plans/ACT_VARIATIONAL_DPO_DEVELOPMENT_PLAN.md`
- Reformulation memo: `docs/reports/ACT_DPO_REFORMULATION_MEMO.md`
- Progress log: `docs/reports/DPO_PROGRESS_LOG.md`
