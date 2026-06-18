# Query-Time Decision Modeling

Query-Time Decision Modeling (QTDM) is a retrieval-conditioned decision engine for evidence-based prediction. It takes a live query, retrieves similar labeled precedent cases, builds a weighted empirical outcome distribution, and returns a bounded prediction with uncertainty, confidence diagnostics, evidence IDs, and refusal behavior.

The core move is simple and useful: QTDM lets retrieved evidence set the numeric decision surface, then lets language-model reasoning act only as a constrained semantic placement signal inside that surface. The LLM does not invent the final target value. It can explain, compare, and tilt, but the final prediction stays grounded in observed precedent.

## Why QTDM Matters

Most retrieval-augmented systems stop at returning relevant documents. Most LLM-only systems can produce fluent numerical guesses without a real calibration mechanism. QTDM turns retrieval into a decision layer:

1. Retrieve comparable historical cases.
2. Extract target labels or measured outcomes.
3. Weight cases by local similarity and support quality.
4. Build a query-time empirical distribution.
5. Optionally map bounded semantic tilt through `p = Phi(z_sem)`.
6. Predict with `prediction = weighted_quantile(p)`.
7. Return intervals, confidence diagnostics, evidence IDs, and refusal reasons.

That makes QTDM useful anywhere the answer should be anchored to precedent: scientific estimates, operational routing, case triage, ranking, decision support, and any workflow where "show me the evidence behind the number" matters.

## Results Snapshot

In a 50-case blind exoplanet evaluation using NASA Exoplanet Archive-derived records, QTDM sharply outperformed both direct LLM inference and weighted KNN retrieval.

| Metric | LLM-only | Weighted KNN | QTDM |
| --- | ---: | ---: | ---: |
| Mean Absolute Error | 270.9 | 164.4 | 16.3 |
| RMSE | N/A | 328.0 | 75.2 |
| Median Absolute Error | N/A | 33.5 | 0.0 |
| Nominal 80% interval coverage | N/A | 96% | 98% |
| Harness-reported exact hits | N/A | 0/50 | 35/50 |

On that run, QTDM delivered:

- 94% MAE reduction versus LLM-only inference
- 90% MAE reduction versus weighted KNN
- 70% exact-hit rate under the evaluation harness
- 2% false high-confidence failure rate
- 25.7 second average latency in the tested stack

Per-field results:

| Target field | Weighted KNN MAE | QTDM MAE | Exact hits |
| --- | ---: | ---: | ---: |
| Equilibrium temperature | 294.4 | 66.6 | 7/10 |
| Planet radius | 1.7 | 0.3 | 6/10 |
| System distance | 235.2 | 7.5 | 8/10 |
| Host temperature | 289.0 | 7.2 | 8/10 |
| Planet density | 1.9 | 0.1 | 6/10 |

A separate cross-domain routing diagnostic refused 20/20 out-of-distribution requests, matching the intended behavior: answer when local evidence supports the query, refuse when it does not.

## What This Repository Gives You

This package extracts the QTDM arbiter into a runnable public implementation:

- Local service-free demo with static exoplanet fixtures
- Weighted empirical distribution mode
- Optional bounded semantic tilt
- Regression, binary classification, and ranking request models
- Confidence diagnostics and refusal gates
- Conformal-style interval support
- FastAPI endpoint and CLI entrypoint
- Evaluation and calibration utilities
- Unit tests with stubbed retrieval clients

The public fixtures are intentionally small so the mechanism is easy to run locally. The larger empirical statistics above come from the private evaluation harness and source records used for the accompanying QTDM technical report.

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

The demo runs end-to-end from local fixtures and prints the request, retrieved cases, weighted distribution summary, prediction, interval, diagnostics, evidence case IDs, and refusal behavior.

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

The mini eval prints a markdown table comparing:

- naive mean
- weighted median / weighted KNN
- QTDM `semantic_distribution`
- QTDM `semantic_distribution` plus mock semantic tilt

Use it as a reproducibility check for the public fixture path.

## API Example

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

Request shape:

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

## CLI Example

The CLI expects a retrieval service with `/search/findings`, `/search/content`, and `/labels` endpoints. Unit tests use stubbed clients, so a live service is not required for local verification.

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

KNN, ridge, logistic, and isotonic estimators remain available as fallback or legacy paths. The flagship public proof path is `semantic_distribution`.

## Retrieval Contract

Live retrieval backends should return:

- stable case IDs such as `finding_id` or `chunk_id`
- text fields such as `summary`, `chunk_text`, or `content`
- similarity or ranking scores when available
- target labels or numeric payload fields for the requested `target_name`

The default HTTP retrieval client expects:

- `POST /search/findings`
- `POST /search/content`
- `GET /labels`

## Repository Layout

```text
qtdm_arbiter/
  api/                  FastAPI route and app factory
  audit/                JSONL audit writer with hashed query logging
  core/                 weighting, distributions, confidence, refusal gates
  domains/exoplanet/    exoplanet demo adapter, field hints, unit hints
  examples/             runnable local demo modules
  integration/          retrieval service client
  models/               Pydantic request/response models
  tests/                acceptance and unit tests
  tools/                evaluation and calibration utilities
examples/
  fixtures/             public fixture data
  mini_eval.py          local fixture reproducibility check
docs/                   architecture and usage notes
```

## Tests

```bash
python3 -m pytest
```

## Validation Note

QTDM is strongest when the retrieval neighborhood is relevant, labeled, and coherent. When support is weak, the system should refuse or return semantic support only. Broader public validation should add larger fixed datasets, preregistered splits, interval-width reporting, calibration curves, and published exact-hit tolerance rules.

## Resume Bullet

Designed and built Query-Time Decision Modeling (QTDM), a retrieval-conditioned decision engine that combines vector-neighborhood evidence, weighted empirical distributions, refusal gates, conformal-style intervals, and constrained LLM semantic tilt to produce explainable predictions with confidence diagnostics.
