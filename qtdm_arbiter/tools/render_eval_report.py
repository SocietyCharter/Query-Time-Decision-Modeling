"""Standalone Markdown report renderer for QTDM comparative evaluation results.

Reads a results.json produced by eval_qtdm_comparative.py and renders a structured
Markdown report with summary, per-mode tables, domain breakdown, calibration,
refusal behavior, high-confidence failures, worst cases, best QTDM wins, and
recommended next fixes.

No imports from qtdm_arbiter — this is a pure stdlib tool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, decimals: int = 4) -> str:
    """Format a numeric value or return 'N/A' for None."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _pct(value: float | None) -> str:
    """Format a rate as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _determine_target_type(baselines: Dict[str, Any]) -> str:
    """Infer whether the eval is regression or binary_classification."""
    for bl_data in baselines.values():
        if bl_data.get("mae") is not None:
            return "regression"
        if bl_data.get("f1") is not None:
            return "binary_classification"
    return "regression"


def _compute_threshold(per_case: List[Dict[str, Any]]) -> float:
    """Compute error threshold: 0.2 * std of true_values, or 1.0 if std < 0.01."""
    true_vals = [pc["true_value"] for pc in per_case if pc.get("true_value") is not None]
    if len(true_vals) < 2:
        return 1.0
    mean_tv = sum(true_vals) / len(true_vals)
    variance = sum((v - mean_tv) ** 2 for v in true_vals) / len(true_vals)
    std = math.sqrt(variance)
    threshold = 0.2 * std
    return threshold if std >= 0.01 else 1.0


def _r2(per_case: List[Dict[str, Any]]) -> float | None:
    """Compute R² from per_case entries."""
    answered = [pc for pc in per_case if pc.get("prediction") is not None and pc.get("error_abs") is not None]
    if len(answered) < 2:
        return None
    true_vals = [pc["true_value"] for pc in answered]
    mean_tv = sum(true_vals) / len(true_vals)
    ss_tot = sum((v - mean_tv) ** 2 for v in true_vals)
    if ss_tot == 0:
        return None
    ss_res = sum(pc["error_abs"] ** 2 for pc in answered)
    return 1.0 - (ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def render_summary(results: Dict[str, Any]) -> str:
    """Section 1: Summary with winner and narrative."""
    run_id = results.get("run_id", "unknown")
    timestamp = results.get("timestamp", "unknown")
    test_set = results.get("test_set", "unknown")
    baselines = results.get("baselines", {})
    target_type = _determine_target_type(baselines)

    lines: List[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Timestamp**: {timestamp}")
    lines.append(f"- **Test set**: {test_set}")
    lines.append(f"- **Baselines evaluated**: {', '.join(sorted(baselines.keys()))}")
    lines.append(f"- **Target type**: {target_type}")
    lines.append("")

    # Determine winner
    winner_name = ""
    winner_val = None
    if target_type == "regression":
        for name, bl in baselines.items():
            mae = bl.get("mae")
            if mae is not None and (winner_val is None or mae < winner_val):
                winner_val = mae
                winner_name = name
    else:
        for name, bl in baselines.items():
            f1 = bl.get("f1")
            if f1 is not None and (winner_val is None or f1 > winner_val):
                winner_val = f1
                winner_name = name

    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Winner**: `{winner_name}` ({'MAE' if target_type == 'regression' else 'F1'} = {_fmt(winner_val)})")
    lines.append("")

    # Narrative
    qtdm_full = baselines.get("qtdm_full", {})
    non_qtdm_baselines = {
        n: b for n, b in baselines.items()
        if n not in ("qtdm_full", "qtdm_no_conformal", "qtdm_no_support_calibration", "qtdm_no_refusal")
        and "placeholder" not in n
    }

    best_non_qtdm_name = ""
    best_non_qtdm_val = None
    if target_type == "regression":
        for name, bl in non_qtdm_baselines.items():
            mae = bl.get("mae")
            if mae is not None and (best_non_qtdm_val is None or mae < best_non_qtdm_val):
                best_non_qtdm_val = mae
                best_non_qtdm_name = name
    else:
        for name, bl in non_qtdm_baselines.items():
            f1 = bl.get("f1")
            if f1 is not None and (best_non_qtdm_val is None or f1 > best_non_qtdm_val):
                best_non_qtdm_val = f1
                best_non_qtdm_name = name

    qtdm_metric = qtdm_full.get("mae") if target_type == "regression" else qtdm_full.get("f1")
    narrative_parts = []
    if qtdm_metric is not None and best_non_qtdm_val is not None:
        if target_type == "regression":
            diff = best_non_qtdm_val - qtdm_metric
            if diff > 0:
                narrative_parts.append(
                    f"QTDM full achieved MAE {_fmt(qtdm_metric)}, "
                    f"outperforming the best non-QTDM baseline ({best_non_qtdm_name}, MAE {_fmt(best_non_qtdm_val)}) "
                    f"by {_fmt(diff, 4)}."
                )
            else:
                narrative_parts.append(
                    f"QTDM full achieved MAE {_fmt(qtdm_metric)}, "
                    f"which did not beat {best_non_qtdm_name} (MAE {_fmt(best_non_qtdm_val)})."
                )
        else:
            diff = qtdm_metric - best_non_qtdm_val
            if diff > 0:
                narrative_parts.append(
                    f"QTDM full achieved F1 {_fmt(qtdm_metric)}, "
                    f"outperforming {best_non_qtdm_name} (F1 {_fmt(best_non_qtdm_val)}) "
                    f"by {_fmt(diff, 4)}."
                )
            else:
                narrative_parts.append(
                    f"QTDM full achieved F1 {_fmt(qtdm_metric)}, "
                    f"which did not beat {best_non_qtdm_name} (F1 {_fmt(best_non_qtdm_val)})."
                )
    else:
        narrative_parts.append("Insufficient data to compare QTDM full against non-QTDM baselines.")

    narrative_parts.append(
        f"Across {len(baselines)} baselines, "
        f"{sum(1 for b in baselines.values() if b.get('n_answered', 0) > 0)} produced answered predictions."
    )
    lines.append(" ".join(narrative_parts))
    lines.append("")
    return "\n".join(lines)


def render_results_by_mode(results: Dict[str, Any]) -> str:
    """Section 2: Results by mode table sorted by MAE/F1."""
    baselines = results.get("baselines", {})
    target_type = _determine_target_type(baselines)

    lines: List[str] = []
    lines.append("## Results by Mode")
    lines.append("")

    def sort_key(item: tuple) -> tuple:
        name, bl = item
        if target_type == "regression":
            mae = bl.get("mae")
            return (mae is None, mae if mae is not None else float("inf"))
        else:
            f1 = bl.get("f1")
            return (f1 is None, -(f1 if f1 is not None else float("inf")))

    sorted_baselines = sorted(baselines.items(), key=sort_key)

    if target_type == "regression":
        lines.append("| Baseline | N Answered | MAE | RMSE | R² | Mean Confidence | Mean Latency (s) |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, bl in sorted_baselines:
            lines.append(
                f"| {name} | {bl.get('n_answered', 'N/A')} | {_fmt(bl.get('mae'))} | "
                f"{_fmt(bl.get('rmse'))} | {_fmt(bl.get('r2'))} | "
                f"{_fmt(bl.get('mean_confidence'))} | {_fmt(bl.get('mean_latency_s'))} |"
            )
    else:
        lines.append("| Baseline | N Answered | Accuracy | F1 | Brier | Mean Confidence | Mean Latency (s) |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, bl in sorted_baselines:
            lines.append(
                f"| {name} | {bl.get('n_answered', 'N/A')} | {_fmt(bl.get('accuracy'))} | "
                f"{_fmt(bl.get('f1'))} | {_fmt(bl.get('brier'))} | "
                f"{_fmt(bl.get('mean_confidence'))} | {_fmt(bl.get('mean_latency_s'))} |"
            )

    lines.append("")
    return "\n".join(lines)


def render_results_by_domain(results: Dict[str, Any]) -> str:
    """Section 3: Results by domain grouped by per_case domain field."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## Results by Domain")
    lines.append("")

    domain_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    has_domain = False

    for bl_name, bl_data in baselines.items():
        per_case = bl_data.get("per_case", [])
        for pc in per_case:
            domain = pc.get("domain")
            if domain:
                has_domain = True
                domain_data[domain].append(pc)

    if not has_domain:
        lines.append("N/A — no domain labels in test set.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Domain | Cases | MAE | Mean Confidence | Refusal Rate |")
    lines.append("|---|---|---|---|---|")

    for domain in sorted(domain_data.keys()):
        cases = domain_data[domain]
        n = len(cases)
        errors = [pc["error_abs"] for pc in cases if pc.get("error_abs") is not None]
        mae = sum(errors) / len(errors) if errors else None
        confs = [pc["confidence"] for pc in cases if pc.get("confidence") is not None]
        mean_conf = sum(confs) / len(confs) if confs else None
        refusals = sum(1 for pc in cases if pc.get("status") == "refused")
        refusal_rate = refusals / n if n > 0 else None
        lines.append(
            f"| {domain} | {n} | {_fmt(mae)} | {_fmt(mean_conf)} | {_pct(refusal_rate)} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_confidence_calibration(results: Dict[str, Any]) -> str:
    """Section 4: Confidence calibration tables per baseline."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## Confidence Calibration")
    lines.append("")

    for bl_name in sorted(baselines.keys()):
        bl_data = baselines[bl_name]
        calibration = bl_data.get("calibration", [])
        lines.append(f"### {bl_name}")
        lines.append("")

        if not calibration:
            lines.append("No calibration data available.")
            lines.append("")
            continue

        lines.append("| Bucket | Mean Confidence | Actual Accuracy | N Cases |")
        lines.append("|---|---|---|---|")

        for bucket in calibration:
            bucket_low = bucket.get("bucket_low", 0)
            bucket_high = bucket.get("bucket_high", 0.1)
            mean_conf = bucket.get("mean_confidence", 0)
            accuracy = bucket.get("accuracy", 0)
            n = bucket.get("n", 0)

            note = ""
            if accuracy is not None and mean_conf is not None:
                gap = accuracy - mean_conf
                if gap < -0.1:
                    note = " ⚠️ over-confident"
                elif gap > 0.1:
                    note = " ⚠️ under-confident"

            lines.append(
                f"| [{_fmt(bucket_low, 1)}, {_fmt(bucket_high, 1)}) | "
                f"{_fmt(mean_conf, 3)} | {_fmt(accuracy, 3)} | {n} |{note}"
            )

        lines.append("")

    return "\n".join(lines)


def render_refusal_behavior(results: Dict[str, Any]) -> str:
    """Section 5: Refusal behavior table."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## Refusal Behavior")
    lines.append("")
    lines.append("| Baseline | Refusal Rate | Fuzzy Rate | Most Common Refusal Reasons |")
    lines.append("|---|---|---|---|")

    for bl_name in sorted(baselines.keys()):
        bl_data = baselines[bl_name]
        refusal_rate = bl_data.get("refusal_rate")
        fuzzy_rate = bl_data.get("fuzzy_rate")

        reasons: List[str] = []
        for pc in bl_data.get("per_case", []):
            rr = pc.get("refusal_reason")
            if rr:
                reasons.append(rr)

        reason_counter = Counter(reasons)
        top_reasons = ", ".join(
            f"{r} ({c})" for r, c in reason_counter.most_common(3)
        ) or "none"

        lines.append(
            f"| {bl_name} | {_pct(refusal_rate)} | {_pct(fuzzy_rate)} | {top_reasons} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_high_confidence_failures(results: Dict[str, Any]) -> str:
    """Section 6: High-confidence failures list."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## High-Confidence Failures")
    lines.append("")
    lines.append("Cases where confidence ≥ 0.7 and error > threshold.")
    lines.append("")

    all_failures: List[Dict[str, Any]] = []

    for bl_name, bl_data in baselines.items():
        per_case = bl_data.get("per_case", [])
        threshold = _compute_threshold(per_case)

        for pc in per_case:
            conf = pc.get("confidence", 0)
            error = pc.get("error_abs")
            if conf is not None and error is not None and conf >= 0.7 and error > threshold:
                all_failures.append({
                    "baseline": bl_name,
                    "query": pc.get("query", ""),
                    "true_value": pc.get("true_value"),
                    "prediction": pc.get("prediction"),
                    "confidence": conf,
                    "error_abs": error,
                })

    if not all_failures:
        lines.append("No high-confidence failures found.")
        lines.append("")
        return "\n".join(lines)

    all_failures.sort(key=lambda x: x["error_abs"], reverse=True)

    lines.append("| Query (truncated) | True | Prediction | Confidence | Baseline |")
    lines.append("|---|---|---|---|---|")

    for f in all_failures:
        query = f["query"][:60] + ("..." if len(f["query"]) > 60 else "")
        lines.append(
            f"| {query} | {_fmt(f['true_value'])} | {_fmt(f['prediction'])} | "
            f"{_fmt(f['confidence'], 3)} | {f['baseline']} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_worst_cases(results: Dict[str, Any]) -> str:
    """Section 7: Worst cases per baseline (top 5)."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## Worst Cases")
    lines.append("")

    for bl_name in sorted(baselines.keys()):
        bl_data = baselines[bl_name]
        per_case = bl_data.get("per_case", [])

        errored = [pc for pc in per_case if pc.get("error_abs") is not None]
        errored.sort(key=lambda x: x["error_abs"], reverse=True)
        worst = errored[:5]

        lines.append(f"### {bl_name}")
        lines.append("")

        if not worst:
            lines.append("No errored cases.")
            lines.append("")
            continue

        lines.append("| Query (truncated) | True | Pred | Error | Confidence |")
        lines.append("|---|---|---|---|---|")

        for pc in worst:
            query = pc.get("query", "")[:60] + ("..." if len(pc.get("query", "")) > 60 else "")
            lines.append(
                f"| {query} | {_fmt(pc['true_value'])} | {_fmt(pc.get('prediction'))} | "
                f"{_fmt(pc['error_abs'])} | {_fmt(pc.get('confidence', 0), 3)} |"
            )

        lines.append("")

    return "\n".join(lines)


def render_best_qtdm_wins(results: Dict[str, Any]) -> str:
    """Section 8: Best QTDM wins — top 10 cases where qtdm_full beats all non-placeholder baselines."""
    baselines = results.get("baselines", {})

    lines: List[str] = []
    lines.append("## Best QTDM Wins")
    lines.append("")
    lines.append("Cases where `qtdm_full` outperformed all non-placeholder baselines by the largest margin.")
    lines.append("")

    qtdm_full_data = baselines.get("qtdm_full")
    if not qtdm_full_data:
        lines.append("No `qtdm_full` baseline data available.")
        lines.append("")
        return "\n".join(lines)

    non_placeholder = {
        n: b for n, b in baselines.items()
        if n != "qtdm_full" and "placeholder" not in n
    }

    if not non_placeholder:
        lines.append("No non-placeholder baselines to compare against.")
        lines.append("")
        return "\n".join(lines)

    def build_lookup(bl_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for pc in bl_data.get("per_case", []):
            rid = pc.get("request_id", "")
            if rid:
                lookup[rid] = pc
        return lookup

    qtdm_lookup = build_lookup(qtdm_full_data)
    other_lookups = {n: build_lookup(b) for n, b in non_placeholder.items()}

    wins: List[Dict[str, Any]] = []

    for rid, qtdm_pc in qtdm_lookup.items():
        qtdm_error = qtdm_pc.get("error_abs")
        if qtdm_error is None:
            continue

        beats_all = True
        best_margin = 0.0

        for other_name, other_lookup in other_lookups.items():
            other_pc = other_lookup.get(rid)
            if other_pc is None:
                continue
            other_error = other_pc.get("error_abs")
            if other_error is None:
                continue
            if qtdm_error >= other_error:
                beats_all = False
                break
            margin = other_error - qtdm_error
            best_margin = max(best_margin, margin)

        if beats_all and best_margin > 0:
            wins.append({
                "request_id": rid,
                "query": qtdm_pc.get("query", ""),
                "true_value": qtdm_pc.get("true_value"),
                "qtdm_prediction": qtdm_pc.get("prediction"),
                "qtdm_error": qtdm_error,
                "best_margin": best_margin,
            })

    wins.sort(key=lambda x: x["best_margin"], reverse=True)
    top_wins = wins[:10]

    if not top_wins:
        lines.append("No cases where QTDM full beat all non-placeholder baselines.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Query (truncated) | True | QTDM Pred | QTDM Error | Best Margin |")
    lines.append("|---|---|---|---|---|")

    for w in top_wins:
        query = w["query"][:60] + ("..." if len(w["query"]) > 60 else "")
        lines.append(
            f"| {query} | {_fmt(w['true_value'])} | {_fmt(w['qtdm_prediction'])} | "
            f"{_fmt(w['qtdm_error'])} | {_fmt(w['best_margin'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_recommended_fixes(results: Dict[str, Any]) -> str:
    """Section 9: Recommended next fixes with static analysis rules."""
    baselines = results.get("baselines", {})
    target_type = _determine_target_type(baselines)

    lines: List[str] = []
    lines.append("## Recommended Next Fixes")
    lines.append("")

    recommendations: List[str] = []

    qtdm_full = baselines.get("qtdm_full", {})
    ridge_only = baselines.get("ridge_only", {})
    qtdm_no_conformal = baselines.get("qtdm_no_conformal", {})

    for bl_name, bl_data in baselines.items():
        fhc = bl_data.get("false_hc_failure_rate")
        if fhc is not None and fhc > 0.05:
            recommendations.append(
                f"- **{bl_name}**: false_hc_failure_rate = {_fmt(fhc)} > 0.05 — "
                "Tighten confidence gate or raise min_support threshold."
            )

    for bl_name, bl_data in baselines.items():
        rr = bl_data.get("refusal_rate")
        if rr is not None and rr > 0.3:
            recommendations.append(
                f"- **{bl_name}**: refusal_rate = {_pct(rr)} > 30% — "
                "Review min_support and min_neighbors thresholds."
            )

    for bl_name, bl_data in baselines.items():
        calibration = bl_data.get("calibration", [])
        over_confident_buckets = [
            b for b in calibration
            if b.get("accuracy") is not None and b.get("mean_confidence") is not None
            and b["accuracy"] - b["mean_confidence"] < -0.1
        ]
        if calibration and len(over_confident_buckets) > len(calibration) * 0.5:
            recommendations.append(
                f"- **{bl_name}**: systematic over-confidence detected in calibration — "
                "Run calibrate_support.py with this eval set."
            )

    if target_type == "regression":
        qtdm_mae = qtdm_full.get("mae")
        ridge_mae = ridge_only.get("mae")
        if qtdm_mae is not None and ridge_mae is not None and qtdm_mae >= ridge_mae:
            recommendations.append(
                "- **qtdm_full** MAE is not better than **ridge_only** — "
                "Review conformal interval tuning and support score weighting."
            )

        no_conf_mae = qtdm_no_conformal.get("mae")
        if qtdm_mae is not None and no_conf_mae is not None and no_conf_mae < qtdm_mae:
            recommendations.append(
                "- **qtdm_no_conformal** beats **qtdm_full** on MAE — "
                "Conformal intervals may be introducing noise — check alpha."
            )

    if not recommendations:
        recommendations.append("- QTDM full pipeline is performing as expected.")

    lines.extend(recommendations)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render_report(results: Dict[str, Any]) -> str:
    """Render the full Markdown report from results dict."""
    sections = [
        render_summary(results),
        render_results_by_mode(results),
        render_results_by_domain(results),
        render_confidence_calibration(results),
        render_refusal_behavior(results),
        render_high_confidence_failures(results),
        render_worst_cases(results),
        render_best_qtdm_wins(results),
        render_recommended_fixes(results),
    ]
    return "\n".join(sections)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Render a Markdown evaluation report from results.json."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to results.json from eval_qtdm_comparative.py."
    )
    parser.add_argument(
        "--out", required=True,
        help="Path to write the Markdown report."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    results_path = Path(args.results)
    out_path = Path(args.out)

    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        return 1

    results = json.loads(results_path.read_text(encoding="utf-8"))
    report = render_report(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
