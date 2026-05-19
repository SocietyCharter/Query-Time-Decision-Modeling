from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

from qtdm_arbiter.core.confidence import _default_support_weights


COMPONENT_KEYS = (
    "neighbor_count",
    "sim_coherence",
    "feature_completeness",
    "target_variance",
    "estimator_agreement",
)


def calibrate_support_weights(eval_payload: Dict[str, Any]) -> Dict[str, float]:
    rows = list(eval_payload.get("per_case", []) or [])
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in rows:
        error_abs = row.get("error_abs")
        diagnostics = row.get("diagnostics", {}) or {}
        components = diagnostics.get("support_components", {}) or {}
        if error_abs is None:
            continue
        if not all(key in components for key in COMPONENT_KEYS):
            continue
        X_rows.append([float(components[key]) for key in COMPONENT_KEYS])
        y_rows.append(float(error_abs))

    if len(X_rows) < 2:
        return _default_support_weights()

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    model = LinearRegression()
    model.fit(X, y)
    raw = np.abs(np.asarray(model.coef_, dtype=float))
    if raw.size != len(COMPONENT_KEYS) or float(np.sum(raw)) <= 0.0:
        return _default_support_weights()
    normalized = raw / np.sum(raw)
    return {key: round(float(value), 6) for key, value in zip(COMPONENT_KEYS, normalized)}


def load_eval_results(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_support_weights(path: Path, weights: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate QTDM support weights from eval output.")
    parser.add_argument("--eval-results", required=True, help="Path to eval_qtdm JSON output.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "config" / "support_weights.json"),
        help="Where to write calibrated weights.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_eval_results(Path(args.eval_results))
    weights = calibrate_support_weights(payload)
    out_path = Path(args.out)
    write_support_weights(out_path, weights)
    print(json.dumps(weights, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
