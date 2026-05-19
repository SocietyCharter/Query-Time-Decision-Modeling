from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    raw = np.asarray(weights, dtype=float)
    if raw.size == 0:
        return raw
    raw = np.clip(raw, 0.0, None)
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.full(len(raw), 1.0 / len(raw), dtype=float)
    return raw / total


def weighted_quantile(values: Sequence[float], weights: Sequence[float], p: float) -> float:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cumulative, float(np.clip(p, 0.0, 1.0)), side="left"))
    idx = min(max(idx, 0), len(x_sorted) - 1)
    return float(x_sorted[idx])


def weighted_ecdf(values: Sequence[float], weights: Sequence[float]) -> List[Tuple[float, float]]:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size == 0:
        return []
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    return [(float(value), float(prob)) for value, prob in zip(x_sorted, cumulative)]


def weighted_histogram(
    values: Sequence[float],
    weights: Sequence[float],
    bins: int = 40,
    value_range: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size == 0:
        return {
            "edges": [],
            "centers": [],
            "probabilities": [],
        }
    if value_range is None:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            value_range = (lo - pad, hi + pad)
        else:
            pad = max((hi - lo) * 0.05, 1e-6)
            value_range = (lo - pad, hi + pad)
    hist, edges = np.histogram(x, bins=int(max(bins, 5)), range=value_range, weights=w, density=False)
    probabilities = normalize_weights(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "edges": edges.astype(float).tolist(),
        "centers": centers.astype(float).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
    }


def weighted_kde_grid(
    values: Sequence[float],
    weights: Sequence[float],
    grid: Sequence[float] | None = None,
    bandwidth: float | None = None,
    points: int = 128,
) -> Dict[str, Any]:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size == 0:
        return {"grid": [], "density": [], "bandwidth": None}
    if grid is None:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            lo -= pad
            hi += pad
        else:
            pad = max((hi - lo) * 0.1, 1e-6)
            lo -= pad
            hi += pad
        grid_arr = np.linspace(lo, hi, int(max(points, 32)), dtype=float)
    else:
        grid_arr = np.asarray(grid, dtype=float)
    if bandwidth is None:
        bandwidth = robust_bandwidth(x, w)
    bw = max(float(bandwidth), 1e-6)
    diffs = (grid_arr[:, None] - x[None, :]) / bw
    density = np.exp(-0.5 * diffs ** 2) / np.sqrt(2.0 * np.pi)
    density = density @ w / bw
    total = float(np.trapezoid(density, grid_arr)) if len(grid_arr) > 1 else float(np.sum(density))
    if total > 0.0:
        density = density / total
    return {
        "grid": grid_arr.astype(float).tolist(),
        "density": density.astype(float).tolist(),
        "bandwidth": bw,
    }


def robust_bandwidth(values: Sequence[float], weights: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size <= 1:
        return 1.0
    mu = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mu) ** 2))
    std = max(np.sqrt(max(variance, 0.0)), 1e-6)
    k_eff = 1.0 / max(float(np.sum(w ** 2)), 1e-9)
    h = 1.06 * std * (k_eff ** (-0.2))
    return float(max(h, std * 0.05, 1e-3))


def numeric_distribution_summary(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    bins: int = 40,
    point_method: str = "weighted_median",
) -> Dict[str, Any]:
    x = np.asarray(values, dtype=float)
    w = normalize_weights(weights)
    if x.size == 0:
        return {
            "point_method": point_method,
            "mean": None,
            "median": None,
            "mode": None,
            "variance": None,
            "sample_count": 0,
            "q05": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "width_80": None,
            "histogram": {"edges": [], "centers": [], "probabilities": []},
            "kde": {"grid": [], "density": [], "bandwidth": None},
        }
    mean = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mean) ** 2))
    q05 = weighted_quantile(x, w, 0.05)
    q10 = weighted_quantile(x, w, 0.10)
    q25 = weighted_quantile(x, w, 0.25)
    median = weighted_quantile(x, w, 0.50)
    q75 = weighted_quantile(x, w, 0.75)
    q90 = weighted_quantile(x, w, 0.90)
    q95 = weighted_quantile(x, w, 0.95)
    histogram = weighted_histogram(x, w, bins=bins)
    kde = weighted_kde_grid(x, w)
    mode = None
    if kde["grid"] and kde["density"]:
        grid = np.asarray(kde["grid"], dtype=float)
        density = np.asarray(kde["density"], dtype=float)
        mode = float(grid[int(np.argmax(density))])
    if mode is None and histogram["centers"]:
        centers = np.asarray(histogram["centers"], dtype=float)
        probs = np.asarray(histogram["probabilities"], dtype=float)
        mode = float(centers[int(np.argmax(probs))])
    return {
        "point_method": point_method,
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "sample_count": int(len(x)),
        "q05": q05,
        "q10": q10,
        "q25": q25,
        "q75": q75,
        "q90": q90,
        "q95": q95,
        "width_80": float(q90 - q10),
        "histogram": histogram,
        "kde": kde,
    }


def binary_distribution_summary(
    labels: Sequence[float],
    weights: Sequence[float],
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.size == 0:
        return {
            "probabilities": {"0": 0.5, "1": 0.5},
            "prob_true": 0.5,
            "top_class": 0,
            "top_probability": 0.5,
            "second_probability": 0.5,
            "class_margin": 0.0,
        }
    positive = float(np.sum(w * y))
    total = float(np.sum(w))
    prob_true = (positive + alpha0) / max(total + alpha0 + beta0, 1e-9)
    prob_false = 1.0 - prob_true
    top_probability = max(prob_true, prob_false)
    second_probability = min(prob_true, prob_false)
    return {
        "probabilities": {"0": prob_false, "1": prob_true},
        "prob_true": prob_true,
        "top_class": 1 if prob_true >= 0.5 else 0,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "class_margin": float(top_probability - second_probability),
    }


def categorical_distribution_summary(
    labels: Sequence[Any],
    weights: Sequence[float],
    *,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    label_list = [str(label) for label in labels]
    if not label_list:
        return {
            "probabilities": {},
            "top_class": None,
            "top_probability": None,
            "second_probability": None,
            "class_margin": None,
        }
    w = np.asarray(weights, dtype=float)
    classes = sorted(set(label_list))
    raw = {cls: float(alpha) for cls in classes}
    for label, weight in zip(label_list, w):
        raw[label] = raw.get(label, float(alpha)) + float(weight)
    total = float(sum(raw.values()))
    probs = {cls: value / total for cls, value in raw.items()}
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_class, top_probability = ordered[0]
    second_probability = ordered[1][1] if len(ordered) > 1 else 0.0
    return {
        "probabilities": probs,
        "top_class": top_class,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "class_margin": float(top_probability - second_probability),
    }


def distribution_overlap(prob_a: Sequence[float], prob_b: Sequence[float]) -> float:
    a = normalize_weights(prob_a)
    b = normalize_weights(prob_b)
    if a.size == 0 or b.size == 0:
        return 1.0
    width = min(len(a), len(b))
    return float(np.sum(np.minimum(a[:width], b[:width])))


def js_divergence_optional(prob_a: Sequence[float], prob_b: Sequence[float], eps: float = 1e-9) -> float:
    a = normalize_weights(prob_a)
    b = normalize_weights(prob_b)
    width = min(len(a), len(b))
    if width == 0:
        return 0.0
    a = np.clip(a[:width], eps, None)
    b = np.clip(b[:width], eps, None)
    m = 0.5 * (a + b)
    kl_a = float(np.sum(a * np.log(a / m)))
    kl_b = float(np.sum(b * np.log(b / m)))
    return 0.5 * (kl_a + kl_b)


def numeric_distribution_overlap(
    values_a: Sequence[float],
    weights_a: Sequence[float],
    values_b: Sequence[float],
    weights_b: Sequence[float],
    *,
    bins: int = 40,
) -> float:
    x_a = np.asarray(values_a, dtype=float)
    x_b = np.asarray(values_b, dtype=float)
    if x_a.size == 0 or x_b.size == 0:
        return 1.0
    lo = float(min(np.min(x_a), np.min(x_b)))
    hi = float(max(np.max(x_a), np.max(x_b)))
    if lo == hi:
        return 1.0
    value_range = (lo, hi)
    hist_a = weighted_histogram(x_a, weights_a, bins=bins, value_range=value_range)
    hist_b = weighted_histogram(x_b, weights_b, bins=bins, value_range=value_range)
    return distribution_overlap(hist_a["probabilities"], hist_b["probabilities"])


def categorical_distribution_overlap(prob_a: Dict[str, float], prob_b: Dict[str, float]) -> float:
    classes = sorted(set(prob_a.keys()) | set(prob_b.keys()))
    if not classes:
        return 1.0
    a = np.asarray([float(prob_a.get(cls, 0.0)) for cls in classes], dtype=float)
    b = np.asarray([float(prob_b.get(cls, 0.0)) for cls in classes], dtype=float)
    return distribution_overlap(a, b)


def mix_weighted_values(
    first_values: Sequence[Any],
    first_weights: Sequence[float],
    second_values: Sequence[Any],
    second_weights: Sequence[float],
    gamma: float,
) -> Tuple[List[Any], np.ndarray]:
    values: List[Any] = []
    weights: List[float] = []
    gamma_clamped = float(np.clip(gamma, 0.0, 1.0))
    if len(first_values):
        w_first = normalize_weights(first_weights) * (1.0 - gamma_clamped)
        values.extend(list(first_values))
        weights.extend(w_first.tolist())
    if len(second_values) and gamma_clamped > 0.0:
        w_second = normalize_weights(second_weights) * gamma_clamped
        values.extend(list(second_values))
        weights.extend(w_second.tolist())
    return values, normalize_weights(weights)
