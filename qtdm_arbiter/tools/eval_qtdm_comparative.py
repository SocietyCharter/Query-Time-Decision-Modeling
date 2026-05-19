from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

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
        for required in ("query", "target_type", "target_name", "true_value"):
            if required not in payload:
                raise ValueError(f"{path}:{line_number} is missing {required}")
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


def _with_policy(request: ArbiterRequest, extra_policy: Dict[str, Any]) -> ArbiterRequest:
    payload = request.model_dump()
    payload["policy"] = {**request.policy, **extra_policy}
    return ArbiterRequest(**payload)


# Baseline definitions (mapped to actual arbiter_decide policy keys)

def baseline_comparative_ridge(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Default comparative pipeline: ridge+KNN weighted, no LLM."""
    return run_request(_with_policy(request, {"estimator": "ridge"}), retrieval_url)


def baseline_comparative_knn(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Weighted KNN only — no ridge regression layer."""
    return run_request(_with_policy(request, {"estimator": "knn"}), retrieval_url)


def baseline_comparative_knn5(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Weighted KNN with only 5 neighbors."""
    return run_request(_with_policy(request, {"estimator": "knn", "k": 5}), retrieval_url)


def baseline_comparative_knn3(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Weighted KNN with only 3 neighbors."""
    return run_request(_with_policy(request, {"estimator": "knn", "k": 3}), retrieval_url)


def baseline_comparative_ridge10(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Ridge with only 10 neighbors."""
    return run_request(_with_policy(request, {"estimator": "ridge", "k": 10}), retrieval_url)


def baseline_comparative_tilt(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Comparative pipeline with LLM semantic tilt pass enabled."""
    return run_request(
        _with_policy(request, {"semantic_tilt": True}),
        retrieval_url,
    )


def baseline_fuzzy_only(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Fuzzy similarity mode - skip label-coverage gates."""
    return run_request(
        _with_policy(request, {"fuzzy": True}),
        retrieval_url,
    )


def baseline_high_support(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Comparative pipeline with strict support threshold (0.8)."""
    return run_request(
        _with_policy(request, {"min_support": 0.8}),
        retrieval_url,
    )


def baseline_conformal_wide(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    """Comparative pipeline with wider conformal intervals (alpha=0.40)."""
    return run_request(
        _with_policy(request, {"conformal_alpha": 0.40}),
        retrieval_url,
    )


BASELINES: Dict[str, Callable[[ArbiterRequest, Optional[str]], ArbiterResponse]] = {
    "comparative_ridge": baseline_comparative_ridge,
    "comparative_knn": baseline_comparative_knn,
    "comparative_knn5": baseline_comparative_knn5,
    "comparative_knn3": baseline_comparative_knn3,
    "comparative_ridge10": baseline_comparative_ridge10,
    "comparative_tilt": baseline_comparative_tilt,
    "fuzzy_only": baseline_fuzzy_only,
    "high_support": baseline_high_support,
    "conformal_wide": baseline_conformal_wide,
}


def _compute_calibration(confidences: List[float], correct: List[bool]) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}
    for conf, is_correct in zip(confidences, correct):
        bucket_idx = min(int(conf * 5), 4)
        if bucket_idx not in buckets:
            buckets[bucket_idx] = {"sum_conf": 0.0, "n_correct": 0, "n": 0}
        buckets[bucket_idx]["sum_conf"] += conf
        buckets[bucket_idx]["n_correct"] += int(is_correct)
        buckets[bucket_idx]["n"] += 1
    result = []
    for idx in sorted(buckets.keys()):
        item = buckets[idx]
        result.append(
            {
                "bucket_low": round(idx * 0.2, 1),
                "bucket_high": round((idx + 1) * 0.2, 1),
                "mean_confidence": round(item["sum_conf"] / item["n"], 4),
                "accuracy": round(item["n_correct"] / item["n"], 4),
                "n": item["n"],
            }
        )
    return result


def compute_metrics(per_case: List[Dict[str, Any]], target_type: str, true_values: List[float]) -> Dict[str, Any]:
    n_cases = len(per_case)
    answered = [c for c in per_case if c["status"] not in {"refused", "semantic_support_only"} and c["prediction"] is not None]
    numeric_answered = [c for c in answered if c["error_abs"] is not None]
    refusals = sum(1 for c in per_case if c["status"] == "refused")
    support_only = sum(1 for c in per_case if c["status"] == "semantic_support_only")
    escalations = sum(1 for c in per_case if c["status"] == "needs_escalation")

    mae = None
    rmse = None
    median_absolute_error = None
    if numeric_answered:
        errors = [float(c["error_abs"]) for c in numeric_answered]
        mae = round(sum(errors) / len(errors), 6)
        rmse = round(math.sqrt(sum(err * err for err in errors) / len(errors)), 6)
        median_absolute_error = round(float(np.median(np.asarray(errors, dtype=float))), 6)

    interval_hits = [bool(c["inside_80"]) for c in answered if c["inside_80"] is not None]
    interval_widths = [float(c["interval_width"]) for c in answered if c["interval_width"] is not None]
    coverage_80 = round(sum(interval_hits) / len(interval_hits), 6) if interval_hits else None
    interval_80_width = round(sum(interval_widths) / len(interval_widths), 6) if interval_widths else None

    effective_ks = [float(c["K_eff"]) for c in answered if c["K_eff"] is not None]
    label_coverages = [float(c["label_coverage"]) for c in answered if c["label_coverage"] is not None]
    overlaps_base = [float(c["overlap_with_base"]) for c in answered if c["overlap_with_base"] is not None]
    overlaps_counter = [float(c["overlap_with_counter"]) for c in answered if c["overlap_with_counter"] is not None]
    jsd_base = [float(c["jsd_vs_base"]) for c in answered if c["jsd_vs_base"] is not None]
    jsd_counter = [float(c["jsd_vs_counter"]) for c in answered if c["jsd_vs_counter"] is not None]
    margins = [float(c["top_class_margin"]) for c in answered if c["top_class_margin"] is not None]
    confidences = [float(c["confidence"]) for c in answered]

    correct_flags: List[bool] = []
    if target_type == "binary_classification":
        for case in answered:
            correct_flags.append(int(round(float(case["prediction"]))) == int(round(float(case["true_value"]))))
    else:
        std_true = float(np.std(np.asarray(true_values, dtype=float))) if len(true_values) >= 2 else 1.0
        threshold = max(0.2 * std_true, 1.0)
        for case in answered:
            correct_flags.append(case["error_abs"] is not None and float(case["error_abs"]) <= threshold)

    high_confidence_error_values = [float(c["error_abs"]) for c in numeric_answered if float(c["confidence"]) >= 0.7]
    high_confidence_error = round(sum(high_confidence_error_values) / len(high_confidence_error_values), 6) if high_confidence_error_values else None
    false_high_confidence_rate = (
        round(
            sum(1 for case, ok in zip(answered, correct_flags) if float(case["confidence"]) >= 0.7 and not ok)
            / max(sum(1 for case in answered if float(case["confidence"]) >= 0.7), 1),
            6,
        )
        if answered and any(float(case["confidence"]) >= 0.7 for case in answered)
        else 0.0
    )

    return {
        "n_cases": n_cases,
        "n_answered": len(answered),
        "refusal_rate": round(refusals / n_cases, 6) if n_cases else 0.0,
        "semantic_support_only_rate": round(support_only / n_cases, 6) if n_cases else 0.0,
        "needs_escalation_rate": round(escalations / n_cases, 6) if n_cases else 0.0,
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": median_absolute_error,
        "interval_80_coverage": coverage_80,
        "interval_80_width": interval_80_width,
        "empirical_distribution_width": interval_80_width,
        "high_confidence_error": high_confidence_error,
        "false_high_confidence_rate": false_high_confidence_rate,
        "overlap_with_base": round(sum(overlaps_base) / len(overlaps_base), 6) if overlaps_base else None,
        "overlap_with_counter": round(sum(overlaps_counter) / len(overlaps_counter), 6) if overlaps_counter else None,
        "JSD_vs_base": round(sum(jsd_base) / len(jsd_base), 6) if jsd_base else None,
        "JSD_vs_counter": round(sum(jsd_counter) / len(jsd_counter), 6) if jsd_counter else None,
        "K_eff": round(sum(effective_ks) / len(effective_ks), 6) if effective_ks else None,
        "label_coverage": round(sum(label_coverages) / len(label_coverages), 6) if label_coverages else None,
        "top_class_margin": round(sum(margins) / len(margins), 6) if margins else None,
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
        "confidence_calibration_by_bucket": _compute_calibration(confidences, correct_flags) if confidences else [],
        "per_case": per_case,
    }


def run_baseline(
    baseline_name: str,
    baseline_fn: Callable[[ArbiterRequest, Optional[str]], ArbiterResponse],
    cases: List[Dict[str, Any]],
    retrieval_url: str,
) -> Dict[str, Any]:
    per_case: List[Dict[str, Any]] = []
    true_values: List[float] = []
    for index, case in enumerate(cases, start=1):
        request = build_request(case, index)
        true_values.append(float(case["true_value"]))
        t0 = time.perf_counter()
        try:
            response = baseline_fn(request, retrieval_url)
        except Exception as exc:
            response = ArbiterResponse(
                request_id=request.request_id,
                status="refused",
                prediction=None,
                prediction_type="error",
                confidence=0.0,
                support_score=0.0,
                model_used=baseline_name,
                neighbors_requested=int(request.policy.get("k", 25)),
                neighbors_used=0,
                evidence_case_ids=[],
                fallback_used=False,
                refusal_reason=f"error:{exc}",
            )
        latency = time.perf_counter() - t0

        answered = response.status not in {"refused", "semantic_support_only"} and response.prediction is not None
        error_abs = None
        if answered and request.target_type != "categorical":
            try:
                error_abs = abs(float(response.prediction) - float(case["true_value"]))
            except (TypeError, ValueError):
                error_abs = None
        pred_low = getattr(response, "prediction_low", None)
        pred_high = getattr(response, "prediction_high", None)
        inside_80 = None
        interval_width = None
        if answered and pred_low is not None and pred_high is not None:
            inside_80 = float(pred_low) <= float(case["true_value"]) <= float(pred_high)
            interval_width = float(pred_high) - float(pred_low)
        distribution_summary = response.distribution_summary or {}
        comparator_summary = response.comparator_summary or {}
        confidence_components = response.confidence_components or {}
        per_case.append(
            {
                "request_id": response.request_id,
                "query": case["query"],
                "target_type": request.target_type,
                "target_name": request.target_name,
                "true_value": float(case["true_value"]),
                "prediction": response.prediction,
                "prediction_low": pred_low,
                "prediction_high": pred_high,
                "status": response.status,
                "confidence": float(response.confidence),
                "support_score": float(response.support_score),
                "model_used": response.model_used,
                "error_abs": round(error_abs, 6) if error_abs is not None else None,
                "inside_80": inside_80,
                "interval_width": round(interval_width, 6) if interval_width is not None else None,
                "latency_s": round(latency, 6),
                "refusal_reason": response.refusal_reason,
                "description": case.get("description"),
                "K_eff": distribution_summary.get("K_eff") or confidence_components.get("K_eff"),
                "label_coverage": confidence_components.get("label_coverage"),
                "overlap_with_base": comparator_summary.get("overlap_with_base"),
                "overlap_with_counter": comparator_summary.get("overlap_with_counter"),
                "jsd_vs_base": comparator_summary.get("jsd_vs_base"),
                "jsd_vs_counter": comparator_summary.get("jsd_vs_counter"),
                "top_class_margin": comparator_summary.get("class_margin") or distribution_summary.get("class_margin"),
                "reasoning_estimate": response.reasoning_estimate,
                "reasoning_model": response.reasoning_model,
                "reasoning_excerpt": None if not response.reasoning else response.reasoning[:200],
            }
        )
    metrics = compute_metrics(per_case, cases[0]["target_type"] if cases else "regression", true_values)
    metrics["baseline"] = baseline_name
    return metrics


def print_summary_table(results: Dict[str, Any]) -> None:
    baselines = results.get("baselines", {})
    if not baselines:
        print("No baselines evaluated.")
        return
    header = f"{'Baseline':<32} {'MAE':>8} {'Cov80':>8} {'K_eff':>8} {'FHCF%':>8} {'Latency':>8}"
    print(header)
    print("-" * len(header))
    for name, data in baselines.items():
        mae = f"{data['mae']:.4f}" if data.get("mae") is not None else "N/A"
        cov = f"{data['interval_80_coverage'] * 100:.1f}%" if data.get("interval_80_coverage") is not None else "N/A"
        k_eff = f"{data['K_eff']:.2f}" if data.get("K_eff") is not None else "N/A"
        fhcf = f"{data['false_high_confidence_rate'] * 100:.1f}%" if data.get("false_high_confidence_rate") is not None else "N/A"
        latencies = [case["latency_s"] for case in data.get("per_case", []) if case.get("latency_s") is not None]
        latency = f"{(sum(latencies) / len(latencies)):.3f}" if latencies else "N/A"
        print(f"{name:<32} {mae:>8} {cov:>8} {k_eff:>8} {fhcf:>8} {latency:>8}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparative evaluation harness for QTDM Arbiter baselines.")
    parser.add_argument("--test-set", required=True, help="Path to JSONL eval cases.")
    parser.add_argument("--retrieval-url", default=RETRIEVAL_URL, help="Retrieval base URL.")
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--out-dir", required=True, help="Output directory for results.")
    parser.add_argument("--baselines", default=",".join(BASELINES.keys()), help="Comma-separated list of baselines to run.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_test_set(Path(args.test_set))
    baselines_to_run = [item.strip() for item in args.baselines.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_set": str(args.test_set),
        "retrieval_url": args.retrieval_url,
        "baselines": {},
    }
    all_failures: List[Dict[str, Any]] = []
    for baseline_name in baselines_to_run:
        if baseline_name not in BASELINES:
            print(f"WARNING: unknown baseline '{baseline_name}', skipping.", file=sys.stderr)
            continue
        print(f"Running baseline: {baseline_name} ...")
        baseline_metrics = run_baseline(baseline_name, BASELINES[baseline_name], cases, args.retrieval_url)
        results["baselines"][baseline_name] = baseline_metrics
        for per_case in baseline_metrics.get("per_case", []):
            if per_case["confidence"] >= 0.7 and per_case["error_abs"] is not None and per_case["error_abs"] > 0:
                all_failures.append(
                    {
                        "baseline": baseline_name,
                        "request_id": per_case["request_id"],
                        "query": per_case["query"],
                        "true_value": per_case["true_value"],
                        "prediction": per_case["prediction"],
                        "confidence": per_case["confidence"],
                        "error_abs": per_case["error_abs"],
                    }
                )
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failures_path = out_dir / "failures.jsonl"
    failures_path.write_text("\n".join(json.dumps(item) for item in all_failures), encoding="utf-8")
    print(f"Results written to {results_path}")
    print(f"Failures written to {failures_path} ({len(all_failures)} entries)")
    print()
    print_summary_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
