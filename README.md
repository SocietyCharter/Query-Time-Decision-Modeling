# Query Time Decision Modeling

Query-Time Decision Modeling (QTDM) Arbiter is a reproducible proof-of-concept decision engine. It takes a query, retrieves similar labeled precedent cases, builds a weighted empirical outcome distribution, optionally applies bounded semantic tilt, and returns a grounded prediction with uncertainty, confidence diagnostics, evidence IDs, and refusal behavior.

This repository is intentionally narrow. It demonstrates the arbiter mechanism: evidence-bounded prediction from retrieved precedent. It is not a generic AI framework.

## What This Repo Does

- Runs a local, service-free QTDM demo with a small static exoplanet fixture dataset.
- Retrieves similar labeled cases through an in-memory demo retrieval client.
- Builds a weighted empirical distribution from retrieved labels.
- Uses `semantic_distribution` as the flagship decision path.
- Optionally maps bounded semantic tilt through `p = Phi(z_sem)` and `prediction = weighted_quantile(p)`.
- Returns a prediction interval, support/confidence diagnostics, evidence case IDs, and explicit refusal states.
- Includes a toy mini-eval to make the proof-of-concept behavior reproducible.

## What It Does Not Do

- It does not prove broad generality.
- It does not include the full private deployment.
- It does not let an LLM directly invent the target value.
- It does not claim the toy fixture eval is the NASA benchmark.

This package is extracted from a larger private deployment; public fixtures are intentionally small.

## Why QTDM Exists

Many retrieval-augmented systems stop at returning similar documents. Many LLM systems produce plausible but ungrounded numerical answers. QTDM sits between those approaches: retrieval supplies precedent, classical distributional estimation supplies the decision surface, and optional language-model reasoning is constrained to a bounded placement signal.

The design goal is decision support with inspectable evidence. If local precedent is weak, sparse, diffuse, or unlabeled, the arbiter should refuse or return semantic support only instead of pretending to know.

## Prerequisites

- Python 3.11 or newer
- `pip` and `venv`
- No external service for the local demo or test suite
- Optional: a compatible retrieval service for live decisions
- Optional: an LLM endpoint for real semantic tilt instead of the mocked demo tilt

## General Requirements

QTDM requests use the accepted target types:

- `regression`
- `binary_classification`
- `ranking`

Live retrieval backends should return:

- Stable case IDs such as `finding_id` or `chunk_id`
- Text fields such as `summary`, `chunk_text`, or `content`
- Similarity or ranking scores when available
- Target labels or numeric payload fields for the requested `target_name`

The default HTTP retrieval client expects:

- `POST /search/findings`
- `POST /search/content`
- `GET /labels`

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

## One-Command Local Demo

```bash
python3 -m qtdm_arbiter.examples.run_demo
```

The demo runs end-to-end from local fixtures and prints:

- request
- retrieved cases
- weighted distribution summary
- prediction
- interval
- confidence/support diagnostics
- evidence case IDs
- refusal behavior for a bad query

Example excerpt:

```text
## weighted distribution summary
- sample_count: 6
- mean: 249.54
- median: 251.0
- q10: 234.0
- q90: 265.0

## prediction
- prediction: 235.0
- interval: [215.0, 255.0]
- model_used: semantic_distribution+semantic_tilt

## refusal behavior
{
  "status": "refused",
  "refusal_reason": "insufficient_neighbors"
}
```

## Mini Eval

```bash
python3 examples/mini_eval.py
```

This prints a markdown table comparing:

- naive mean
- weighted median / weighted KNN
- QTDM `semantic_distribution`
- QTDM `semantic_distribution` + mock semantic tilt

This is a toy reproducibility check using the included fixture dataset, not the NASA benchmark and not a universal superiority claim.

## Architecture

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

KNN, ridge, logistic, and isotonic estimators remain available as fallback or legacy modes. The public proof path is `semantic_distribution`.

## API Example

```json
{
  "request_id": "demo-exoplanet-eqt",
  "target_type": "regression",
  "target_name": "pl_eqt",
  "entity_type": "content",
  "query_summary": "temperate rocky planet around an M-type star in the habitable zone",
  "filters": {},
  "features": {},
  "policy": {
    "mode": "semantic_distribution",
    "domain": "exoplanet",
    "k": 6,
    "min_support": 0.2,
    "semantic_tilt": true,
    "mock_semantic_tilt": -0.35
  },
  "data_types": ["exoplanet_record"]
}
```

Start the API:

```bash
uvicorn qtdm_arbiter.api:create_app --factory --host 127.0.0.1 --port 8001
```

Send a request:

```bash
curl -s http://127.0.0.1:8001/arbiter/decide \
  -H 'Content-Type: application/json' \
  -d @examples/demo_request.json
```

## Public Proof-of-Concept Status

This repo proves the public arbiter mechanism:

- evidence-bounded prediction
- bounded semantic tilt
- refusal gates
- inspectable outputs
- runnable demo and mini-eval without private services

Larger validation still requires fixed public datasets, preregistered splits, calibration curves, and interval-width reporting.

## Repository Layout

```text
qtdm_arbiter/
  api/                  FastAPI route and app factory
  audit/                JSONL audit writer with hashed query logging
  core/                 domain-neutral weighting, distributions, confidence, refusal gates
  domains/exoplanet/    exoplanet demo adapter, field hints, unit hints
  examples/             runnable local demo modules
  integration/          retrieval service client
  models/               Pydantic request/response models
  tests/                acceptance and unit tests
  tools/                evaluation and calibration utilities
examples/
  fixtures/             public fixture data
  mini_eval.py          toy reproducibility check
docs/                   public architecture and usage notes
```

## Tests

```bash
python3 -m pytest
```
