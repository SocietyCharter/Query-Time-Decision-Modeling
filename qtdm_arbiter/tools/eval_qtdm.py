from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdm_arbiter.arbiter_decide import RETRIEVAL_URL, run_request
from qtdm_arbiter.models.request import ArbiterRequest
from qtdm_arbiter.models.response import ArbiterResponse


def load_test_set(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if "query" not in payload or "target_type" not in payload or "target_name" not in payload:
            raise ValueError(f"{path}:{line_number} is missing one of query/target_type/target_name")
        if "true_value" not in payload:
            raise ValueError(f"{path}:{line_number} is missing true_value")
        cases.append(payload)
    return cases


def build_request(case: Dict[str, Any], index: int) -> ArbiterRequest:
    return ArbiterRequest(
        request_id=str(case.get("request_id") or f"eval-{index:03d}-{uuid.uuid4().hex[:8]}"),
        target_type=str(case["target_type"]),
        target_name=str(case["target_name"]),
        entity_type=case.get("entity_type"),
        query_summary=str(case["query"]),
        filters=dict(case.get("filters", {}) or {}),
        features=dict(case.get("features", {}) or {}),
        policy=dict(case.get("policy", {}) or {}),
        data_types=list(case.get("data_types", []) or []),
    )


def evaluate_cases(
    cases: Sequence[Dict[str, Any]],
    *,
    retrieval_url: str,
    run_request_fn: Callable[[ArbiterRequest, str | None], ArbiterResponse] = run_request,
) -> Dict[str, Any]:
    per_case: List[Dict[str, Any]] = []
    absolute_errors: List[float] = []
    squared_errors: List[float] = []
    interval_widths: List[float] = []
    support_scores: List[float] = []
    confidences: List[float] = []
    refusals = 0
    support_only = 0
    escalations = 0
    coverage_hits = 0
    coverage_total = 0

    for index, case in enumerate(cases, start=1):
        request = build_request(case, index)
        response = run_request_fn(request, retrieval_url)
        answered = response.status not in {"refused", "semantic_support_only"} and response.prediction is not None
        if response.status == "refused":
            refusals += 1
        if response.status == "semantic_support_only":
            support_only += 1
        if response.status == "needs_escalation":
            escalations += 1
        if answered:
            support_scores.append(float(response.support_score))
            confidences.append(float(response.confidence))

        error_abs = None
        error_sq = None
        if request.target_type == "regression" and answered:
            error_abs = abs(float(response.prediction) - float(case["true_value"]))
            error_sq = error_abs ** 2
            absolute_errors.append(error_abs)
            squared_errors.append(error_sq)

        inside_80 = None
        prediction_low = getattr(response, "prediction_low", None)
        prediction_high = getattr(response, "prediction_high", None)
        if answered and prediction_low is not None and prediction_high is not None:
            inside_80 = float(prediction_low) <= float(case["true_value"]) <= float(prediction_high)
            coverage_hits += int(bool(inside_80))
            coverage_total += 1
            interval_widths.append(float(prediction_high) - float(prediction_low))

        per_case.append(
            {
                "request_id": response.request_id,
                "query": case["query"],
                "target_type": request.target_type,
                "target_name": request.target_name,
                "true_value": float(case["true_value"]),
                "prediction": response.prediction,
                "prediction_low": prediction_low,
                "prediction_high": prediction_high,
                "status": response.status,
                "support_score": float(response.support_score),
                "confidence": float(response.confidence),
                "refusal_reason": response.refusal_reason,
                "diagnostics": dict(response.diagnostics or {}),
                "distribution_summary": dict(response.distribution_summary or {}),
                "comparator_summary": dict(response.comparator_summary or {}),
                "error_abs": round(error_abs, 6) if error_abs is not None else None,
                "error_sq": round(error_sq, 6) if error_sq is not None else None,
                "inside_80": inside_80,
                "description": case.get("description"),
            }
        )

    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None
    mean_support = sum(support_scores) / len(support_scores) if support_scores else None
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    coverage_80 = (coverage_hits / coverage_total) if coverage_total else None
    interval_80_width = sum(interval_widths) / len(interval_widths) if interval_widths else None

    return {
        "n_cases": len(cases),
        "mae": round(mae, 6) if mae is not None else None,
        "rmse": round(rmse, 6) if rmse is not None else None,
        "coverage_80": round(coverage_80, 6) if coverage_80 is not None else None,
        "interval_80_width": round(interval_80_width, 6) if interval_80_width is not None else None,
        "refusal_rate": round(refusals / len(cases), 6) if cases else 0.0,
        "semantic_support_only_rate": round(support_only / len(cases), 6) if cases else 0.0,
        "needs_escalation_rate": round(escalations / len(cases), 6) if cases else 0.0,
        "mean_support": round(mean_support, 6) if mean_support is not None else None,
        "mean_confidence": round(mean_confidence, 6) if mean_confidence is not None else None,
        "per_case": per_case,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QTDM Arbiter predictions against a labeled JSONL seed set.")
    parser.add_argument("--test-set", required=True, help="Path to JSONL eval cases.")
    parser.add_argument("--retrieval-url", default=RETRIEVAL_URL, help="Retrieval base URL passed into run_request().")
    parser.add_argument("--out", required=True, help="Where to write the JSON report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_test_set(Path(args.test_set))
    report = evaluate_cases(cases, retrieval_url=str(args.retrieval_url))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
