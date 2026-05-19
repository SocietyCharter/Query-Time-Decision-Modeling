# Architecture

QTDM Arbiter is built around a deliberately constrained decision pipeline.

## Data Flow

```text
request
  -> retrieval client
  -> neighborhood filtering
  -> feature and label extraction
  -> weighted empirical distribution
  -> optional semantic tilt
  -> prediction and interval
  -> confidence and refusal gates
  -> response with evidence IDs and diagnostics
```

## Core Components

- `qtdm_arbiter.models`: request and response schemas.
- `qtdm_arbiter.integration`: HTTP client for a retrieval service.
- `qtdm_arbiter.core.neighborhood`: query expansion, supporting/counter neighborhoods, and result validation.
- `qtdm_arbiter.core.features`: numeric feature matrix and query vector extraction.
- `qtdm_arbiter.core.distribution`: weighted summaries, quantiles, overlap, and distribution mixing.
- `qtdm_arbiter.core.estimator`: weighted KNN, ridge, logistic, isotonic, and semantic-distribution estimators.
- `qtdm_arbiter.core.confidence`: support scoring, adaptive weighting, and confidence components.
- `qtdm_arbiter.core.refusal`: refusal and fuzzy-estimate gates.
- `qtdm_arbiter.core.reasoning`: optional semantic tilt call. The LLM returns tilt only, not the target value.
- `qtdm_arbiter.core.dcs`: optional structured decision-context synthesis.

## Design Guardrails

- The prediction is grounded in retrieved historical outcomes.
- The model can refuse when the neighborhood is too weak.
- The LLM is optional and constrained.
- Audit logs hash the raw query summary instead of storing full sensitive text.
- A response always exposes support, confidence, and evidence-case identifiers.

## Retrieval Contract

The default `RetrievalClient` expects three HTTP surfaces:

- `POST /search/findings`
- `POST /search/content`
- `GET /labels`

The package does not require a specific vector database. Any service that returns compatible JSON can back the arbiter.

