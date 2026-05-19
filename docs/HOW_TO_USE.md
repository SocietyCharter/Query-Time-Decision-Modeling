# How to Use Query Time Decision Modeling

This guide shows how to install the public QTDM arbiter package, run the local demo, start the API, and send a decision request.

QTDM is not a standalone database. It expects a compatible retrieval service for live decisions. The included demo and tests use local fixtures and stubbed clients, so you can validate the public proof-of-concept without running a retrieval backend.

## 1. Install

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

## 2. Run the Local Demo

```bash
python3 -m qtdm_arbiter.examples.run_demo
```

The demo uses `examples/fixtures/exoplanet_cases.json` and an in-memory retrieval client. It prints the request, retrieved cases, weighted distribution summary, prediction, interval, confidence diagnostics, evidence IDs, and one refusal example.

## 3. Run the Mini Eval

```bash
python3 examples/mini_eval.py
```

This is a toy reproducibility check over the public fixture dataset. It is not the NASA benchmark.

## 4. Run the Test Suite

```bash
.venv/bin/python -m pytest
```

The tests exercise semantic distribution, semantic tilt through weighted quantiles, confidence scoring, conformal intervals, refusal behavior, feature extraction, and end-to-end demo scripts.

## 5. Understand the Retrieval Contract

The default HTTP retrieval client expects these endpoints:

- `POST /search/findings`
- `POST /search/content`
- `GET /labels`

The retrieval service should return cases with:

- A stable case identifier such as `finding_id` or `chunk_id`
- A text field such as `summary`, `chunk_text`, or `content`
- Similarity scores when available
- Target labels or numeric payload fields relevant to the requested target

The arbiter is intentionally retrieval-backend agnostic. A vector database, search service, or custom case store can be used if it returns compatible JSON.

## 6. Run a CLI Decision

Set the retrieval service URL, then call the arbiter module:

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

If the retrieved neighborhood is weak, sparse, incoherent, or missing usable labels, QTDM should refuse instead of returning an unsupported prediction.

## 7. Run the API

Start the FastAPI app:

```bash
uvicorn qtdm_arbiter.api:create_app --factory --host 127.0.0.1 --port 8001
```

Send the included demo request:

```bash
curl -s http://127.0.0.1:8001/arbiter/decide \
  -H 'Content-Type: application/json' \
  -d @examples/demo_request.json
```

## 8. Request Shape

A typical request contains:

```json
{
  "request_id": "demo-exoplanet-eqt",
  "target_type": "regression",
  "target_name": "pl_eqt",
  "entity_type": "content",
  "query_summary": "temperate rocky planet around an M-type star",
  "filters": {},
  "features": {},
  "data_types": ["exoplanet_record"],
  "policy": {
    "k": 10,
    "min_support": 0.2,
    "mode": "semantic_distribution",
    "domain": "exoplanet"
  }
}
```

Common target types:

- `regression`: numeric prediction with interval and diagnostics
- `binary_classification`: binary probability with support diagnostics
- `ranking`: ordered candidates when the retrieval payload supports ranking

## 9. Response Fields

The response includes:

- `status`: `ok`, `refused`, or another explicit state
- `prediction`: the selected value or class when supported
- `prediction_low` / `prediction_high`: interval bounds when available
- `confidence`: support-derived confidence score
- `support_score`: observable evidence support
- `diagnostics`: neighborhood quality, label coverage, feature completeness, and related signals
- `evidence_case_ids`: retrieved cases used to support the decision
- `refusal_reason`: explanation when the system declines to answer

The design goal is inspectability. A caller should be able to see not only what QTDM predicted, but whether the local evidence justified answering.

## 10. Optional LLM Semantic Tilt

QTDM can call an LLM for semantic tilt, but the LLM does not directly produce the final numeric answer. It emits a bounded placement signal that is mapped into the weighted empirical distribution built from retrieved cases.

This keeps the final prediction tied to observed evidence while still allowing semantic context to influence where the query sits inside the retrieved neighborhood.

## 11. Evaluation Utilities

The `qtdm_arbiter/tools/` directory contains evaluation and calibration helpers:

- `eval_qtdm.py`: baseline evaluation harness
- `eval_qtdm_comparative.py`: comparative baseline runner
- `eval_ranking.py`: ranking-oriented evaluation
- `eval_three_condition.py`: three-condition diagnostic runner
- `calibrate_support.py`: support-weight calibration helper
- `render_eval_report.py`: report renderer for evaluation outputs

These tools are intended for experimentation and replication work. They may require a live retrieval service and domain-specific test sets.

## 12. Where to Start in the Code

- `qtdm_arbiter/arbiter_decide.py`: main decision pipeline and CLI entrypoint
- `qtdm_arbiter/api/routes.py`: API route wrapper
- `qtdm_arbiter/integration/retrieval_client.py`: default HTTP retrieval client
- `qtdm_arbiter/core/distribution.py`: weighted empirical distribution utilities
- `qtdm_arbiter/core/confidence.py`: confidence and support scoring
- `qtdm_arbiter/core/refusal.py`: refusal gates
- `qtdm_arbiter/core/reasoning.py`: optional bounded semantic tilt
- `qtdm_arbiter/domains/exoplanet/`: exoplanet demo adapter
