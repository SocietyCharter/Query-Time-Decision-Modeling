# Architecture

QTDM Arbiter is built around a constrained decision pipeline. The public proof path is `semantic_distribution`.

```text
query
  -> retrieve precedent cases
  -> extract labels
  -> compute weights
  -> build weighted empirical outcome distribution
  -> optional bounded semantic tilt
  -> quantile prediction
  -> interval/confidence/refusal gates
  -> evidence-linked response
```

## Core Package

- `qtdm_arbiter.models`: Pydantic request and response schemas.
- `qtdm_arbiter.integration`: HTTP retrieval client contract.
- `qtdm_arbiter.core.neighborhood`: retrieval normalization, neighborhood validation, and query variants.
- `qtdm_arbiter.core.features`: generic feature matrix extraction for legacy estimators.
- `qtdm_arbiter.core.distribution`: weighted empirical summaries, quantiles, histograms, KDE summaries, and overlap utilities.
- `qtdm_arbiter.core.estimator`: semantic distribution estimator plus legacy KNN/ridge/logistic/isotonic fallback paths.
- `qtdm_arbiter.core.confidence`: support scoring and adaptive retrieval weights.
- `qtdm_arbiter.core.refusal`: refusal gates for sparse, weak, diffuse, or unsupported neighborhoods.
- `qtdm_arbiter.core.reasoning`: optional bounded semantic tilt. The LLM returns tilt only, not the target value.

## Domain Adapter

The core package is domain-neutral. Exoplanet-specific fields, unit hints, and neighbor formatting live in:

```text
qtdm_arbiter/domains/exoplanet/
```

The public demo uses exoplanets because the original proof work used exoplanet-style records, but the arbiter mechanism is not hardwired to planetary science.

## Demo Components

- `examples/fixtures/exoplanet_cases.json`: small static labeled fixture dataset.
- `qtdm_arbiter.examples.fixture_retrieval`: in-memory retrieval backend.
- `qtdm_arbiter.examples.run_demo`: service-free local demonstration.
- `examples/mini_eval.py`: toy markdown-table reproducibility check.

## Retrieval Contract

For live use, the default `RetrievalClient` expects:

- `POST /search/findings`
- `POST /search/content`
- `GET /labels`

Any backend can be used if it returns compatible JSON with stable IDs, text, scores, and labels or target fields.

## Guardrails

- The prediction is grounded in retrieved historical outcomes.
- Semantic tilt is optional and bounded to `[-2, 2]`.
- Tilt changes numeric regression predictions only through weighted quantile selection.
- The arbiter refuses when support is too weak.
- Responses expose confidence, support diagnostics, and evidence case IDs.
- Audit logs hash raw query text instead of storing it directly.
