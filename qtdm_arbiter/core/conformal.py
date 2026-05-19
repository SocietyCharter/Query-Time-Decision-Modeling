from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from qtdm_arbiter.core.distribution import weighted_quantile


def semantic_interval_from_distribution(distribution_summary: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    q10 = distribution_summary.get("q10")
    q90 = distribution_summary.get("q90")
    if q10 is None or q90 is None:
        return None
    return float(q10), float(q90)


def conformal_overlay(
    y_neighbors: np.ndarray,
    y_hat_neighbors: np.ndarray,
    weights: np.ndarray,
    y_hat_query: float,
    alpha: float = 0.20,
) -> Optional[Tuple[float, float]]:
    return weighted_conformal_interval(y_neighbors, y_hat_neighbors, weights, y_hat_query, alpha=alpha)


def asymmetric_conformal_interval(
    signed_residuals: np.ndarray,
    weights: np.ndarray,
    y_hat_corrected: float,
    alpha: float = 0.20,
) -> Optional[Tuple[float, float]]:
    mask = ~np.isnan(signed_residuals)
    if int(np.sum(mask)) < 5:
        return None
    r = signed_residuals[mask]
    w = weights[mask]
    if float(np.sum(w)) <= 0:
        return None
    w = w / np.sum(w)
    q_low = weighted_quantile(r, w, alpha / 2)
    q_high = weighted_quantile(r, w, 1.0 - alpha / 2)
    return (y_hat_corrected + q_low, y_hat_corrected + q_high)


def weighted_conformal_interval(
    y_neighbors: np.ndarray,
    y_hat_neighbors: np.ndarray,
    weights: np.ndarray,
    y_hat_query: float,
    alpha: float = 0.20,
) -> Optional[Tuple[float, float]]:
    mask = ~np.isnan(y_neighbors) & ~np.isnan(y_hat_neighbors)
    if int(np.sum(mask)) < 5:
        return None

    y_cal = np.asarray(y_neighbors[mask], dtype=float)
    y_hat_cal = np.asarray(y_hat_neighbors[mask], dtype=float)
    w_cal = np.asarray(weights[mask], dtype=float)
    if w_cal.size == 0 or float(np.sum(w_cal)) <= 0.0:
        return None
    w_cal = w_cal / np.sum(w_cal)

    residuals = np.abs(y_cal - y_hat_cal)
    sort_idx = np.argsort(residuals)
    sorted_r = residuals[sort_idx]
    sorted_w = w_cal[sort_idx]
    cumulative_w = np.cumsum(sorted_w)
    threshold = 1.0 - float(alpha)
    candidates = sorted_r[cumulative_w >= threshold]
    q_hat = float(candidates[0]) if len(candidates) else float(sorted_r[-1])

    prediction = float(y_hat_query)
    return (prediction - q_hat, prediction + q_hat)
