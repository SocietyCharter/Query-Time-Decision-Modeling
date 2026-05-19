"""Toy reproducibility check for the public QTDM fixture dataset."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qtdm_arbiter.core.distribution import weighted_quantile
from qtdm_arbiter.core.estimator import estimate_semantic_distribution
from qtdm_arbiter.core.reasoning import normal_cdf
from qtdm_arbiter.examples.fixture_retrieval import InMemoryDemoRetrievalClient, load_fixture_cases


EVAL_QUERIES = [
    ("temperate rocky planet around an M-type star", 251.0),
    ("dense rocky super earth near cool habitable zone", 235.0),
    ("warm sub neptune around M dwarf habitable-zone-like insolation", 284.0),
    ("hot gas giant close to sun-like star", 1450.0),
]


def main() -> int:
    cases = load_fixture_cases()
    client = InMemoryDemoRetrievalClient(cases=cases)
    rows = []
    for query, truth in EVAL_QUERIES:
        neighbors = client.search_content(query, {}, 6, data_types=["exoplanet_record"])
        values = [float(row["pl_eqt"]) for row in neighbors]
        weights = _normalize([float(row["_semantic_score"]) for row in neighbors])
        naive = float(np.mean([float(row["pl_eqt"]) for row in cases]))
        weighted_median = weighted_quantile(values, weights, 0.5)
        qdist = estimate_semantic_distribution("regression", values, weights)["prediction"]
        tilt = _mock_tilt(query)
        tilted = weighted_quantile(values, weights, normal_cdf(tilt))
        rows.append(
            {
                "query": query,
                "truth": truth,
                "naive_mean": naive,
                "weighted_knn": weighted_median,
                "qtdm_semantic_distribution": float(qdist),
                "qtdm_semantic_distribution_plus_mock_tilt": float(tilted),
            }
        )

    print("# Mini eval")
    print()
    print("Toy reproducibility check using the included fixture dataset. This is not the NASA benchmark and does not claim universal superiority.")
    print()
    print("| method | MAE |")
    print("| --- | ---: |")
    for method in ("naive_mean", "weighted_knn", "qtdm_semantic_distribution", "qtdm_semantic_distribution_plus_mock_tilt"):
        mae = _mae(rows, method)
        print(f"| {method} | {mae:.2f} |")
    print()
    print("| query | truth | naive_mean | weighted_knn | semantic_distribution | semantic_distribution + mock tilt |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {query} | {truth:.1f} | {naive_mean:.1f} | {weighted_knn:.1f} | {qtdm_semantic_distribution:.1f} | {qtdm_semantic_distribution_plus_mock_tilt:.1f} |".format(
                **row
            )
        )
    return 0


def _normalize(values: List[float]) -> List[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / len(clipped) for _ in clipped]
    return [value / total for value in clipped]


def _mock_tilt(query: str) -> float:
    lower = query.lower()
    if "hot gas giant" in lower:
        return 1.2
    if "sub neptune" in lower or "warmer" in lower:
        return 0.35
    if "cool" in lower:
        return -0.55
    return -0.15


def _mae(rows: List[Dict[str, Any]], method: str) -> float:
    errors = [abs(float(row[method]) - float(row["truth"])) for row in rows if not math.isnan(float(row[method]))]
    return float(np.mean(errors))


if __name__ == "__main__":
    raise SystemExit(main())

