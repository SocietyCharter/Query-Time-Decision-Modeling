# Query Time Decision Modeling

Query-Time Decision Modeling (QTDM) is a retrieval-conditioned decision system designed by Jesse Brownfield. It turns similar historical cases into a bounded prediction, confidence score, and evidence trail while keeping language-model reasoning inside explicit guardrails.

The intent is straightforward: use retrieval to find the best available precedent, use classical estimation to build the decision surface, and use an LLM only as a constrained semantic placement signal. The language model does not directly predict the target value. It can only move the estimate within a distribution built from retrieved evidence.

## Core Idea

1. Retrieve comparable cases from a vector or content retrieval service.
2. Extract the observed target values from those cases.
3. Build a weighted empirical distribution.
4. Optionally ask an LLM for semantic tilt only.
5. Convert that tilt into a quantile inside the observed distribution.
6. Return the prediction, uncertainty band, diagnostics, and supporting case IDs.

## Why This Exists

Many retrieval-augmented systems stop after returning similar documents. Many LLM systems go the other direction and produce plausible but ungrounded numerical answers. QTDM sits between those approaches: retrieval supplies precedent, classical estimation supplies the decision signal, and the LLM is constrained to explanation and bounded ranking/tilt.

The project is built for decision support where unsupported confidence is worse than abstention. A QTDM response exposes the retrieved evidence, support diagnostics, confidence signals, and refusal reasons so a caller can see why the system answered or why it declined.

## Results Snapshot

The accompanying white paper reports an initial 50-case blind exoplanet demonstration using NASA Exoplanet Archive-derived records. This is proof-of-concept evidence, not a universal benchmark claim.

| Metric | LLM-only | Weighted KNN | QTDM |
| --- | ---: | ---: | ---: |
| Mean Absolute Error | 270.9 | 164.4 | 16.3 |
| RMSE | N/A | 328.0 | 75.2 |
| Median Absolute Error | N/A | 33.5 | 0.0 |
| Nominal 80% interval coverage | N/A | 96% | 98% |
| Harness-reported exact hits | N/A | 0/50 | 35/50 |

On that demonstration, QTDM reduced MAE by 94% against LLM-only inference and 90% against weighted KNN. A separate cross-domain routing diagnostic refused 20/20 out-of-distribution queries, which is the intended behavior when local evidence is weak.

The important part is not that QTDM always wins. The important part is the architecture: the system can use LLM reasoning without letting the LLM invent the final number. Broader validation still needs larger public datasets, fixed splits, preregistered tolerance rules, interval-width reporting, and calibration curves.

## Public Package

This repo is a public package of the QTDM arbiter component. It is intentionally separated from the original private deployment and does not include private data, credentials, runtime logs, or operational infrastructure.

For setup, API usage, request fields, response interpretation, and evaluation utilities, see [How to Use Query Time Decision Modeling](docs/HOW_TO_USE.md).

## Features

- Retrieval-conditioned regression, classification, and ranking support
- Weighted nearest-neighbor and local-model fallback paths
- Semantic distribution mode for evidence-first prediction
- Refusal gates for weak neighborhoods, sparse labels, diffuse similarity, and low support
- Confidence diagnostics with effective sample size, label coverage, feature completeness, and coherence
- Optional semantic tilt and decision-context synthesis stages
- FastAPI endpoint and CLI entrypoint
- Unit tests and small public fixtures

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

## Run Tests

```bash
.venv/bin/python -m pytest
```

## CLI Example

The CLI expects a retrieval service with `/search/findings`, `/search/content`, and `/labels` endpoints. Tests use stubbed clients, so a live service is not required for unit testing.

```bash
QTDM_RETRIEVAL_URL=http://localhost:8000 \
python -m qtdm_arbiter.arbiter_decide \
  --query "temperate rocky planet around an M-type star" \
  --target-type regression \
  --target-name pl_eqt \
  --entity-type content \
  --data-types exoplanet_record \
  --policy '{"k": 10, "min_support": 0.2, "mode": "semantic_distribution"}'
```

## API Example

```bash
uvicorn qtdm_arbiter.api:create_app --factory --host 127.0.0.1 --port 8001
```

```bash
curl -s http://127.0.0.1:8001/arbiter/decide \
  -H 'Content-Type: application/json' \
  -d @examples/demo_request.json
```

## Repository Layout

```text
qtdm_arbiter/
  api/          FastAPI route and app factory
  audit/        JSONL audit writer with hashed query logging
  core/         weighting, feature extraction, estimators, refusal gates
  integration/  retrieval service client
  models/       Pydantic request/response models
  tools/        evaluation and calibration utilities
  tests/        unit tests with stubbed retrieval clients
examples/       small request/response examples
docs/           public architecture notes
```

## Resume Bullet

Designed and built Query-Time Decision Modeling (QTDM), a retrieval-conditioned decision engine that combines vector-neighborhood evidence, weighted empirical distributions, refusal gates, and constrained LLM semantic tilt to produce explainable predictions with confidence diagnostics.
