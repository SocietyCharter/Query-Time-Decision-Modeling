from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from qtdm_arbiter.core.distribution import distribution_overlap, normalize_weights


def _default_support_weights() -> Dict[str, float]:
    return {
        "neighbor_count": 0.30,
        "sim_coherence": 0.25,
        "feature_completeness": 0.25,
        "target_variance": 0.10,
        "estimator_agreement": 0.10,
    }


def _load_support_weights() -> Dict[str, float]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "support_weights.json"
    if not config_path.exists():
        return _default_support_weights()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return _default_support_weights()
    defaults = _default_support_weights()
    if not isinstance(payload, dict):
        return defaults
    merged = dict(defaults)
    for key in defaults:
        if key in payload:
            try:
                merged[key] = float(payload[key])
            except (TypeError, ValueError):
                continue
    total = sum(max(value, 0.0) for value in merged.values())
    if total <= 0.0:
        return defaults
    return {key: max(value, 0.0) / total for key, value in merged.items()}


_SUPPORT_WEIGHTS = _load_support_weights()


def effective_k(weights: Sequence[float]) -> float:
    w = normalize_weights(weights)
    if w.size == 0:
        return 0.0
    return float(1.0 / max(float(np.sum(w ** 2)), 1e-9))


def weight_entropy(weights: Sequence[float]) -> float:
    w = normalize_weights(weights)
    if w.size == 0:
        return 0.0
    safe = np.clip(w, 1e-12, None)
    return float(-np.sum(safe * np.log(safe)))


def _adaptive_lambda(scores: List[float]) -> float:
    if len(scores) < 2:
        return 3.0
    sigma = float(np.std(np.asarray(scores, dtype=float)))
    if sigma < 1e-6:
        return 1.0
    bandwidth = 1.06 * sigma * (len(scores) ** -0.2)
    if bandwidth <= 1e-9:
        return 1.0
    return float(np.clip(1.0 / bandwidth, 0.5, 10.0))


def compute_weights_adaptive_temp(scores: List[float], tau: float = 0.008) -> np.ndarray:
    if not scores:
        return np.array([], dtype=float)
    d = 1.0 - np.asarray(scores, dtype=float)
    d_shifted = d - float(np.min(d))
    tau_value = max(float(tau), 1e-6)
    logits = -d_shifted / tau_value
    logits -= float(np.max(logits))
    raw = np.exp(logits)
    return normalize_weights(raw)


def search_tau_for_k_eff(
    scores: Sequence[float],
    *,
    k: int | None = None,
    target_low: float | None = None,
    target_high: float | None = None,
    tau_min: float = 1e-4,
    tau_max: float = 1.0,
    steps: int = 72,
) -> Dict[str, Any]:
    if not scores:
        return {
            "weights": np.array([], dtype=float),
            "tau": 0.008,
            "k_eff": 0.0,
            "target_low": 0.0,
            "target_high": 0.0,
            "flat": True,
        }
    score_array = np.asarray(scores, dtype=float)
    k_total = int(k or len(score_array) or 1)
    k_eff_target_low = float(target_low if target_low is not None else max(5, 0.30 * k_total))
    k_eff_target_high = float(target_high if target_high is not None else max(8, 0.60 * k_total))
    target_mid = 0.5 * (k_eff_target_low + k_eff_target_high)

    tau_grid = np.geomspace(float(max(tau_min, 1e-6)), float(max(tau_max, tau_min * 10.0)), int(max(steps, 12)))
    best: Dict[str, Any] | None = None
    for tau in tau_grid:
        weights = compute_weights_adaptive_temp(score_array.tolist(), tau=float(tau))
        k_eff_value = effective_k(weights)
        in_band = k_eff_target_low <= k_eff_value <= k_eff_target_high
        distance = 0.0 if in_band else abs(k_eff_value - target_mid)
        candidate = {
            "weights": weights,
            "tau": float(tau),
            "k_eff": k_eff_value,
            "distance": distance,
            "in_band": in_band,
        }
        if best is None:
            best = candidate
            continue
        if candidate["in_band"] and not best["in_band"]:
            best = candidate
            continue
        if candidate["in_band"] == best["in_band"] and candidate["distance"] < best["distance"]:
            best = candidate
            continue
        if candidate["in_band"] == best["in_band"] and math.isclose(candidate["distance"], best["distance"]) and candidate["tau"] < best["tau"]:
            best = candidate
    assert best is not None
    flat = bool(
        float(np.std(score_array)) < 0.01
        or best["k_eff"] / max(float(k_total), 1.0) > 0.85
    )
    best["target_low"] = k_eff_target_low
    best["target_high"] = k_eff_target_high
    best["flat"] = flat
    return best


def compute_weights(
    scores: List[float],
    strategy: str = "adaptive_k_eff",
    lam: float | None = None,
    tau: float | None = None,
    target_low: float | None = None,
    target_high: float | None = None,
) -> np.ndarray:
    if not scores:
        return np.array([], dtype=float)
    score_array = np.asarray(scores, dtype=float)
    if strategy == "inverse_distance":
        raw = 1.0 / np.maximum(1.0 - score_array + 1e-5, 1e-5)
        return normalize_weights(raw)
    if strategy in {"adaptive_temp", "adaptive_k_eff"}:
        if tau is not None:
            return compute_weights_adaptive_temp(scores, tau=float(tau))
        return search_tau_for_k_eff(scores, k=len(scores), target_low=target_low, target_high=target_high)["weights"]
    lambda_value = float(lam) if lam is not None else _adaptive_lambda(scores)
    shifted = score_array - np.max(score_array)
    raw = np.exp(lambda_value * shifted)
    return normalize_weights(raw)



def numeric_narrowness(q10: float | None, q90: float | None, base_q05: float | None, base_q95: float | None) -> float:
    if None in (q10, q90, base_q05, base_q95):
        return 0.0
    width_80 = float(q90) - float(q10)
    range_ref = max(float(base_q95) - float(base_q05), 1e-9)
    return float(np.clip(1.0 - min(max(width_80 / range_ref, 0.0), 1.0), 0.0, 1.0))


def class_margin(prob_vector: Dict[str, float] | Sequence[float]) -> float:
    if isinstance(prob_vector, dict):
        values = sorted((float(v) for v in prob_vector.values()), reverse=True)
    else:
        values = sorted((float(v) for v in prob_vector), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def compute_distribution_confidence(
    *,
    weights: Sequence[float],
    scores: Sequence[float],
    label_coverage: float,
    source_quality: float,
    distribution_summary: Dict[str, Any],
    base_summary: Dict[str, Any] | None,
    counter_summary: Dict[str, Any] | None,
    target_type: str,
    overlap_with_base: float | None,
    overlap_with_counter: float | None,
    flat_semantic_neighborhood: bool,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    weights_arr = normalize_weights(weights)
    scores_arr = np.asarray(scores, dtype=float)
    k_eff_value = effective_k(weights_arr)
    k_eff_target = max(8.0, 0.45 * max(float(len(weights_arr)), 1.0))
    k_eff_score = float(np.clip(k_eff_value / k_eff_target, 0.0, 1.0))

    density = float(np.mean(scores_arr)) if scores_arr.size else 0.0
    density = float(np.clip(density, 0.0, 1.0))

    if target_type in {"regression", "ranking"}:
        narrowness = numeric_narrowness(
            distribution_summary.get("q10"),
            distribution_summary.get("q90"),
            None if not base_summary else base_summary.get("q05"),
            None if not base_summary else base_summary.get("q95"),
        )
        class_or_narrowness = narrowness
    else:
        probabilities = distribution_summary.get("probabilities", {})
        class_or_narrowness = class_margin(probabilities)

    counter_overlap = float(overlap_with_counter if overlap_with_counter is not None else 1.0)
    counter_penalty = float(np.clip(1.0 - counter_overlap, 0.0, 1.0))

    comparator_overlap = float(overlap_with_base if overlap_with_base is not None else 1.0)
    comparator_margin = float(np.clip(1.0 - comparator_overlap, 0.0, 1.0))

    dominant_case = bool(weights_arr.size and float(np.max(weights_arr)) > 0.50)
    semantic_weighting_flat = bool(flat_semantic_neighborhood)

    confidence = (
        k_eff_score
        * float(np.clip(label_coverage, 0.0, 1.0))
        * float(np.clip(source_quality, 0.0, 1.0))
        * density
        * float(np.clip(class_or_narrowness, 0.0, 1.0))
        * counter_penalty
        * max(comparator_margin, 0.1)
    )
    if dominant_case:
        confidence *= 0.70
    if semantic_weighting_flat:
        confidence *= 0.70
    final_confidence = float(np.clip(confidence, 0.0, 1.0))

    components = {
        "K_eff": round(k_eff_value, 4),
        "K_eff_score": round(k_eff_score, 4),
        "label_coverage": round(float(np.clip(label_coverage, 0.0, 1.0)), 4),
        "source_quality": round(float(np.clip(source_quality, 0.0, 1.0)), 4),
        "density": round(density, 4),
        "narrowness_or_class_margin": round(float(np.clip(class_or_narrowness, 0.0, 1.0)), 4),
        "counter_penalty": round(counter_penalty, 4),
        "comparator_margin": round(comparator_margin, 4),
        "final_confidence": round(final_confidence, 4),
    }
    diagnostics = {
        "semantic_weighting_flat": semantic_weighting_flat,
        "dominant_case": dominant_case,
        "weight_entropy": round(weight_entropy(weights_arr), 4),
        "max_weight": round(float(np.max(weights_arr)) if weights_arr.size else 0.0, 4),
        "weight_std": round(float(np.std(weights_arr)) if weights_arr.size else 0.0, 4),
        "counter_overlap": round(counter_overlap, 4),
        "comparator_overlap": round(comparator_overlap, 4),
    }
    return final_confidence, components, diagnostics


def compute_support(
    n_usable: int,
    scores: List[float],
    X: np.ndarray,
    y: np.ndarray,
    y_hat_model: float | None,
    y_hat_knn: float | None,
    feature_completeness: float | None = None,
    target_coverage: float | None = None,
    proxy_target: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    feature_completeness_value = float(feature_completeness if feature_completeness is not None else _matrix_completeness(X))
    target_coverage_value = float(target_coverage if target_coverage is not None else (float(np.mean(~np.isnan(y))) if len(y) else 0.0))
    sim_coherence = float(np.std(np.asarray(scores, dtype=float))) if scores else 1.0

    valid_y = y[~np.isnan(y)]
    target_variance = float(np.var(valid_y)) if len(valid_y) > 1 else 1.0
    agreement = _estimator_agreement(valid_y, y_hat_model, y_hat_knn)
    neighbor_term = min(float(n_usable) / 25.0, 1.0)
    coherence_term = 1.0 - min(sim_coherence, 1.0)
    feature_term = feature_completeness_value
    variance_term = 1.0 - min(target_variance, 1.0)
    agreement_term = 1.0 - min(agreement, 1.0)

    support_score = (
        neighbor_term * _SUPPORT_WEIGHTS["neighbor_count"]
        + coherence_term * _SUPPORT_WEIGHTS["sim_coherence"]
        + feature_term * _SUPPORT_WEIGHTS["feature_completeness"]
        + variance_term * _SUPPORT_WEIGHTS["target_variance"]
        + agreement_term * _SUPPORT_WEIGHTS["estimator_agreement"]
    )
    support_score = float(max(0.0, min(support_score, 1.0)))

    # relevance_penalty removed 2026-05-01: RBF kernel crushed support scores
    # for typical retrieval (mean ~0.3 → penalty ~0.375). Violated doctrine.

    diagnostics = {
        "n_usable": int(n_usable),
        "sim_coherence": round(sim_coherence, 4),
        "feature_completeness": round(feature_completeness_value, 4),
        "target_variance": round(min(target_variance, 1.0), 4),
        "estimator_agreement": round(min(agreement, 1.0), 4),
        "target_coverage": round(target_coverage_value, 4),
        "proxy_target": proxy_target,
        "support_components": {
            "neighbor_count": round(neighbor_term, 4),
            "sim_coherence": round(coherence_term, 4),
            "feature_completeness": round(feature_term, 4),
            "target_variance": round(variance_term, 4),
            "estimator_agreement": round(agreement_term, 4),
        },
    }
    return support_score, diagnostics


def _matrix_completeness(X: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    return float(np.mean(~np.isnan(X)))


def _estimator_agreement(valid_y: np.ndarray, y_hat_model: float | None, y_hat_knn: float | None) -> float:
    if y_hat_model is None or y_hat_knn is None:
        return 1.0
    if len(valid_y) > 1:
        scale = max(float(np.nanmax(valid_y) - np.nanmin(valid_y)), 1.0)
    else:
        scale = 1.0
    return abs(float(y_hat_model) - float(y_hat_knn)) / scale
