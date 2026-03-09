# ACT DPO Reformulation Memo

## Related Diagrams
- Diagram index: `docs/specs/diagrams/README.md`
- ACT internal model: `docs/specs/diagrams/act_model_structure.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_model_structure.puml)
- Surrogate vs variational scoring: `docs/specs/diagrams/act_dpo_scoring_comparison.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_dpo_scoring_comparison.puml)
- Variational DPO architecture: `docs/specs/diagrams/act_variational_dpo_architecture.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_architecture.puml)
- Variational DPO sequence: `docs/specs/diagrams/act_variational_dpo_sequence.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_sequence.puml)
- Training pipeline: `docs/specs/diagrams/act_variational_dpo_pipeline.puml` — [Rendered](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/zebehn/act/act-variational-dpo-redesign/docs/specs/diagrams/act_variational_dpo_pipeline.puml)

> Note: rendered links resolve once branch `act-variational-dpo-redesign` is pushed to GitHub.

## Purpose
This memo records a design review of the current ACT post-training approach, prompted by the concern that our present objective may look like a DPO-inspired pairwise ranking loss rather than a strict DPO formulation. It captures:
- the original comment motivating the redesign,
- the current implementation and where it diverges from strict DPO,
- a revised research framing,
- a concrete formulation draft for a latent-aware ACT preference objective,
- an implementation roadmap.

This document is intended as a history-tracking artifact so later work can resume from a clear conceptual checkpoint.

## Triggering Comment
Original feedback:

> 원래 DPO는 선호된 출력과 비선호 출력의 log-probability 차이를 직접 최적화합니다. 그런데 ACT는 CVAE 계열의 연속값 action chunk 생성기라서, “chunk likelihood를 어떻게 정의하고 계산할 것인가”가 진짜 연구 문제가 됩니다. 연구의 핵심을 **“continuous chunk policy용 DPO 정식화”**로 세우면 충분히 새로워질 수 있습니다. 반대로 chosen/rejected에 대한 reconstruction error만으로 밀어붙이면, 리뷰어는 “이건 strict DPO가 아니라 pairwise ranking loss 아니냐”라고 볼 가능성이 큽니다. chosen/rejected action chunk의 log-likelihood를 정확히 어떻게 정의하는지, latent z를 어떻게 다루는지, ACT의 training objective와 DPO objective가 어떻게 연결되는지를 분명히 설계해야 합니다.

## Short Response
The comment is correct. The current implementation is useful experimentally, but it does not yet constitute a fully convincing strict DPO formulation for ACT as a latent-variable continuous chunk policy. The main unresolved question is not preference pairing itself; it is the definition and estimation of chunk likelihood under ACT's latent-variable structure.

## Current ACT-DPO Pipeline
### Preference construction
The current pipeline builds preferences at the rollout level:
- rollouts are grouped by seed,
- chosen and rejected rollouts are selected by success first and return as a tiebreak,
- a deterministic fixed window is extracted from the chosen/rejected pair.

This is implemented in `posttrain/preferences.py` where full-rollout metadata is ranked and then converted into a windowed pair representation.

### DPO training
The trainer:
- replays the chosen and rejected windows,
- computes policy and reference log-probability-like scores,
- applies a logistic DPO-style margin objective.

This is implemented in `posttrain/dpo.py` using `trajectory_logprob(...)` from `posttrain/rollouts.py`.

### ACT likelihood surrogate
The present likelihood surrogate comes from:
- `ACTPolicy.get_action_distribution(...)` returning a Gaussian distribution over chunk actions in `policy.py`,
- `detr/models/detr_vae.py` exposing decoder mean plus a learned diagonal `action_log_std`.

The current post-training path therefore treats ACT as if it defines a tractable Gaussian chunk policy at replay time.

## Part 1 — Where the Current Implementation Diverges from Strict DPO
### Strict DPO target
Standard DPO optimizes differences in policy log-probabilities:

\[
\log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x)
\]

relative to a frozen reference policy.

### ACT mismatch
ACT is not naturally a direct autoregressive discrete policy. It is a CVAE-style chunk generator:
- during training, latent variables are inferred from action chunks,
- at inference, actions are generated from a latent prior or latent surrogate,
- the true marginal chunk likelihood is latent-integrated.

For ACT, the natural chunk policy is:

\[
\pi_\theta(a^{(H)} \mid o) = \int p_\theta(z \mid o) \, p_\theta(a^{(H)} \mid o, z) \, dz
\]

or, in the original ACT-style prior case,

\[
\pi_\theta(a^{(H)} \mid o) = \int p(z) \, p_\theta(a^{(H)} \mid o, z) \, dz.
\]

The current implementation does not estimate this marginal explicitly.

### Code-grounded mismatch table
| Area | Current implementation | Why this is not yet strict DPO |
|---|---|---|
| Preference unit | Full rollout is ranked, then converted to a fixed window in `posttrain/preferences.py` | Preference source is episode-level, not directly chunk-level |
| Optimization unit | DPO is applied to replayed windows in `posttrain/dpo.py` | Acceptable as a design choice, but should be framed as episode-to-window distillation |
| Policy likelihood | `trajectory_logprob(...)` sums or averages Gaussian action log-probs over replayed chunks in `posttrain/rollouts.py` | This is a surrogate chunk density, not the exact ACT marginal likelihood |
| Latent treatment | In inference mode, `DETRVAE.forward(...)` uses a zero latent rather than integrating over latent uncertainty in `detr/models/detr_vae.py` | The latent integral defining \(\pi_\theta(a\mid o)\) is bypassed |
| Reference comparison | Policy and reference are compared using the same surrogate replay score in `posttrain/dpo.py` | The DPO algebra is preserved, but the score is only DPO-like if the likelihood estimate is not exact |
| BC–DPO connection | ACT BC training uses a VAE-style training path when actions are provided, but DPO replay uses inference-time Gaussian chunk scores | The training objective and post-training objective are not yet derived from one unified latent-policy likelihood |

### Bottom line
The current method should be described more honestly as:
- **episode-ranked, window-optimized, latent-agnostic surrogate Gaussian-replay DPO-style preference optimization for ACT**

rather than strict DPO for ACT.

## Part 2 — Revised Research Framing
### Proposed reframing
The research novelty should be centered as:

**Latent-aware preference optimization for continuous action chunk policies**

or, more concretely:

**Variational DPO for continuous chunk-generating latent policies**

This makes the core research problem explicit:
- how to define chunk likelihood for a latent chunk policy,
- how to estimate it tractably,
- how to connect ACT BC pretraining and preference fine-tuning within one objective family.

### Why this framing is stronger
This framing is stronger because it makes the actual hard problem explicit rather than hiding it behind a familiar DPO label. The novelty is not just “we applied DPO to robot actions”; it is “we defined and optimized preference-consistent chunk likelihoods for a latent continuous chunk policy.”

## Revised Formulation Draft
### 1. Chunk policy definition
Let:
- \(o_t\) be the observation context at time \(t\),
- \(a_t^{(H)} = (a_t, a_{t+1}, \dots, a_{t+H-1})\) be the action chunk.

Define the ACT chunk policy as:

\[
\pi_\theta(a_t^{(H)} \mid o_t) = \int p_\theta(z_t \mid o_t) \, p_\theta(a_t^{(H)} \mid o_t, z_t) \, dz_t.
\]

If the ACT prior remains unconditional, replace \(p_\theta(z_t \mid o_t)\) with \(p(z_t)\).

### 2. Decoder likelihood
Use an explicit continuous decoder likelihood, for example a diagonal Gaussian:

\[
p_\theta(a_t^{(H)} \mid o_t, z_t)
=
\mathcal{N}
\big(
\mu_\theta(o_t, z_t),
\operatorname{diag}(\sigma_\theta^2(o_t, z_t))
\big).
\]

This gives a principled chunk likelihood rather than relying on a reconstruction score alone.

### 3. Exact DPO route
A strict DPO route would require estimating:

\[
\log \pi_\theta(a_t^{(H)} \mid o_t)
=
\log \int p_\theta(z_t \mid o_t) p_\theta(a_t^{(H)} \mid o_t, z_t) dz_t.
\]

This can be approximated with Monte Carlo or importance weighting, but it is expensive.

### 4. Variational DPO route
A more practical route is to use an ELBO-based score:

\[
\log \pi_\theta(a_t^{(H)} \mid o_t)
\ge
\mathbb{E}_{q_\phi(z_t \mid o_t, a_t^{(H)})}
\left[
\log p_\theta(a_t^{(H)} \mid o_t, z_t)
\right]
-
\mathrm{KL}
\left(
q_\phi(z_t \mid o_t, a_t^{(H)})
\Vert
p_\theta(z_t \mid o_t)
\right).
\]

Call this quantity:

\[
\mathcal{S}_\theta(o_t, a_t^{(H)})
\]

and interpret it as a variational chunk score.

### 5. Window-level score
For a replay window \(W\), define:

\[
\mathcal{S}_\theta(W)
=
\sum_{t \in W}
\mathcal{S}_\theta(o_t, a_t^{(H)}).
\]

A normalized average version is also valid and likely more stable:

\[
\bar{\mathcal{S}}_\theta(W)
=
\frac{1}{|W|}
\sum_{t \in W}
\mathcal{S}_\theta(o_t, a_t^{(H)}).
\]

### 6. Variational DPO objective
For chosen and rejected windows \(W^+, W^-\), define:

\[
\mathcal{L}_{\mathrm{VDPO}}
=
-
\log \sigma
\Big(
\beta
\big[
(\mathcal{S}_\theta(W^+) - \mathcal{S}_\theta(W^-))
-
(\mathcal{S}_{\mathrm{ref}}(W^+) - \mathcal{S}_{\mathrm{ref}}(W^-))
\big]
\Big).
\]

This preserves the DPO structure while making clear that the score is a variational estimate of chunk likelihood.

## ACT BC Objective and DPO Objective Connection
### BC pretraining
ACT BC pretraining is naturally ELBO-shaped:

\[
\mathcal{L}_{\mathrm{BC}} = -\mathcal{L}_{\mathrm{ELBO}}.
\]

### Preference fine-tuning
A principled post-training objective can be:

\[
\mathcal{L}
=
\lambda_{\mathrm{pref}} \mathcal{L}_{\mathrm{VDPO}}
+
\lambda_{\mathrm{BC}} \mathcal{L}_{\mathrm{ELBO}}.
\]

This hybrid objective gives two benefits:
- it ties preference optimization back to ACT's latent generative training objective,
- it reduces collapse risk during post-training.

## Revised Training Pipeline Draft
### Stage 0 — baseline BC
Train ACT with its standard latent-variable objective and keep:
- policy weights,
- encoder \(q_\phi(z\mid o,a)\),
- prior or conditional prior module,
- decoder variance parameterization.

### Stage 1 — rollout collection
Collect rollout candidates under a robust collection policy:
- temporal aggregation where appropriate,
- deterministic anchor candidates plus exploratory variants,
- seed-grouped candidate buckets.

### Stage 2 — preference construction
Build episode-level preferences:
- group by matched seed / initial state bucket,
- choose preferred vs dispreferred rollout using success first, return tiebreak,
- convert episode preference into window-level optimization units.

### Stage 3 — latent-aware scoring
For each chunk in the chosen and rejected windows:
- infer latent with \(q_\phi(z\mid o,a)\),
- compute decoder log-likelihood,
- subtract latent regularization term,
- accumulate a window score.

### Stage 4 — variational DPO fine-tuning
Apply the variational DPO objective relative to the frozen reference model.

### Stage 5 — hybrid stabilization
Optionally keep a BC ELBO regularizer to preserve action realism and prevent degradation.

### Stage 6 — evaluation
Evaluate under the same closed-loop protocol used for BC baselines:
- same rollout count,
- same temporal aggregation setting,
- multiple seed ranges for robustness,
- mean/std across repeated preference collections.

## Recommended Terminology
Until the latent-aware score is implemented, avoid calling the current method strict DPO. Preferred terms:
- **DPO-style preference optimization for ACT**
- **latent-agnostic surrogate Gaussian-replay DPO**
- **surrogate chunk-likelihood preference optimization**

Once the ELBO-based score is implemented, preferred terms:
- **Variational DPO for ACT**
- **Latent-aware DPO for continuous chunk policies**

## Immediate Implementation Priorities
1. Add an encoder-based chunk scoring path for chosen/rejected windows.
2. Define a conditional prior or explicitly document use of the fixed ACT prior.
3. Implement ELBO chunk scoring and use it inside DPO training.
4. Add a hybrid `variational DPO + BC` objective option.
5. Measure robustness over multiple rollout seed ranges rather than best-run only.

## Summary
The core revision is conceptual as much as algorithmic. The current pipeline showed that preference fine-tuning can help, but the research-grade question is deeper: how should a latent continuous chunk policy define and optimize preference-consistent chunk likelihoods? The revised approach makes that question explicit and turns it into the actual contribution.
