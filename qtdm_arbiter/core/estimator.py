from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression

from qtdm_arbiter.core.distribution import (
    binary_distribution_summary,
    categorical_distribution_summary,
    mix_weighted_values,
    normalize_weights,
    numeric_distribution_summary,
)


RIDGE_ALPHA = float(os.environ.get("QTDM_RIDGE_ALPHA", 1.0))
LOGISTIC_C = float(os.environ.get("QTDM_LOGISTIC_C", 1.0))


class _ProjectedIsotonicModel:
    def __init__(self, model: IsotonicRegression, feature_weights: np.ndarray) -> None:
        self.model = model
        self.feature_weights = np.asarray(feature_weights, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            scalar = np.array([float(np.dot(matrix, self.feature_weights))], dtype=float)
        else:
            scalar = matrix @ self.feature_weights
        return np.asarray(self.model.predict(scalar), dtype=float)


def select_model(
    target_type: str,
    n_usable: int,
    feature_completeness: float,
    proxy_target: bool = False,
    content_mode: bool = False,
    mode: str = "legacy",
) -> str:
    if mode != "legacy":
        if target_type == "binary_classification":
            return "semantic_distribution_binary"
        if target_type == "ranking":
            return "semantic_distribution_ranking"
        if target_type == "categorical":
            return "semantic_distribution_categorical"
        return "semantic_distribution_numeric"
    if n_usable < 10 or feature_completeness < 0.5:
        return "knn"
    if content_mode and feature_completeness < 0.4:
        return "knn"
    if target_type == "ranking":
        return "isotonic"
    if target_type == "binary_classification" and not proxy_target:
        return "logistic"
    # Regression: KNN is the primary estimator.
    # Use estimator="ridge" policy flag to force ridge regression.
    if target_type == "regression":
        return "knn"
    if target_type == "binary_classification":
        return "ridge"
    return "knn"


def estimate_semantic_distribution(
    target_type: str,
    values: Sequence[Any],
    weights: Sequence[float],
    *,
    point_method: str = "weighted_median",
) -> Dict[str, Any]:
    w = normalize_weights(weights)
    if target_type in {"regression", "ranking"}:
        summary = numeric_distribution_summary(values, w, point_method=point_method)
        prediction = summary["median"] if point_method == "weighted_median" else summary["mean"]
        return {
            "prediction": prediction,
            "prediction_mean": summary["mean"],
            "prediction_median": summary["median"],
            "prediction_mode": summary["mode"],
            "prediction_low": summary["q10"],
            "prediction_high": summary["q90"],
            "distribution_summary": summary,
            "prediction_type": "semantic_distribution_numeric" if target_type == "regression" else "semantic_distribution_ranking",
        }
    if target_type == "binary_classification":
        labels = [float(v) for v in values]
        summary = binary_distribution_summary(labels, w)
        return {
            "prediction": summary["prob_true"],
            "prediction_mean": summary["prob_true"],
            "prediction_median": summary["prob_true"],
            "prediction_mode": float(summary["top_class"]),
            "prediction_low": None,
            "prediction_high": None,
            "distribution_summary": summary,
            "prediction_type": "semantic_distribution_binary",
        }
    summary = categorical_distribution_summary(values, w)
    return {
        "prediction": summary["top_class"],
        "prediction_mean": None,
        "prediction_median": None,
        "prediction_mode": None,
        "prediction_low": None,
        "prediction_high": None,
        "distribution_summary": summary,
        "prediction_type": "semantic_distribution_categorical",
    }


def fit_and_predict(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    x_q: np.ndarray,
    alpha: float | None = None,
    logistic_c: float | None = None,
) -> Tuple[float, Any]:
    mask = ~np.isnan(y)
    X_fit = X[mask]
    y_fit = y[mask]
    w_fit = weights[mask]

    if len(y_fit) == 0:
        raise ValueError("no_labeled_rows")

    if model_name == "knn":
        prediction = float(np.average(y_fit, weights=w_fit)) if np.sum(w_fit) else float(np.mean(y_fit))
        return prediction, None

    if model_name == "ridge":
        model = Ridge(alpha=float(alpha if alpha is not None else RIDGE_ALPHA))
        model.fit(X_fit, y_fit, sample_weight=w_fit)
        return float(model.predict(np.asarray([x_q], dtype=float))[0]), model

    if model_name == "logistic":
        unique = np.unique(y_fit)
        if len(unique) < 2:
            raise ValueError("single_class_target")
        if not set(np.round(unique).astype(int)).issubset({0, 1}):
            raise ValueError("non_binary_target")
        model = LogisticRegression(
            C=float(logistic_c if logistic_c is not None else LOGISTIC_C),
            max_iter=1000,
            solver="liblinear",
        )
        model.fit(X_fit, y_fit.astype(int), sample_weight=w_fit)
        return float(model.predict_proba(np.asarray([x_q], dtype=float))[0][1]), model

    if model_name == "isotonic":
        feature_weights = np.full(X_fit.shape[1], 1.0 / max(X_fit.shape[1], 1), dtype=float)
        x_scalar = np.asarray(X_fit @ feature_weights, dtype=float)
        if len(x_scalar):
            x_scalar = x_scalar + np.linspace(0.0, 1e-6, len(x_scalar), dtype=float)
        model = IsotonicRegression(out_of_bounds="clip")
        sample_weight = w_fit if float(np.sum(w_fit)) > 0.0 else None
        model.fit(x_scalar, y_fit, sample_weight=sample_weight)
        wrapped = _ProjectedIsotonicModel(model, feature_weights)
        return float(wrapped.predict(np.asarray([x_q], dtype=float))[0]), wrapped

    raise ValueError(f"unsupported_model:{model_name}")


def mix_query_and_intersection(
    query_values: Sequence[Any],
    query_weights: Sequence[float],
    intersection_values: Sequence[Any],
    intersection_weights: Sequence[float],
    gamma: float,
) -> Tuple[Sequence[Any], np.ndarray]:
    return mix_weighted_values(query_values, query_weights, intersection_values, intersection_weights, gamma)
