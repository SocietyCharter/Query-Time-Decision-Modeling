from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.estimator import estimate_semantic_distribution, fit_and_predict, select_model


class EstimatorTests(unittest.TestCase):
    def test_select_model_prefers_knn_for_weak_support(self) -> None:
        self.assertEqual(select_model("binary_classification", 6, 0.8), "knn")
        self.assertEqual(select_model("binary_classification", 12, 0.45), "knn")
        self.assertEqual(select_model("binary_classification", 12, 0.8), "logistic")
        self.assertEqual(select_model("ranking", 15, 0.8), "isotonic")

    def test_fit_and_predict_ridge_and_logistic(self) -> None:
        X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]], dtype=float)
        weights = np.array([0.2, 0.2, 0.3, 0.3], dtype=float)
        x_q = np.array([0.85, 0.85], dtype=float)

        ridge_prediction, _ = fit_and_predict("ridge", X, np.array([0.1, 0.2, 0.9, 0.95]), weights, x_q)
        logistic_prediction, _ = fit_and_predict("logistic", X, np.array([0.0, 0.0, 1.0, 1.0]), weights, x_q)

        self.assertGreater(ridge_prediction, 0.5)
        self.assertGreater(logistic_prediction, 0.5)

    def test_fit_and_predict_isotonic_clips_to_observed_range(self) -> None:
        X = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.25],
                [0.3, 0.35],
                [0.4, 0.45],
                [0.5, 0.55],
                [0.6, 0.65],
                [0.7, 0.75],
                [0.8, 0.85],
                [0.9, 0.95],
                [1.0, 1.05],
                [1.1, 1.15],
                [1.2, 1.25],
                [1.3, 1.35],
                [1.4, 1.45],
                [1.5, 1.55],
            ],
            dtype=float,
        )
        y = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], dtype=float)
        weights = np.full(len(y), 1.0 / len(y), dtype=float)
        x_q = np.array([2.0, 2.1], dtype=float)

        prediction, model = fit_and_predict("isotonic", X, y, weights, x_q)

        self.assertIsNotNone(model)
        self.assertGreaterEqual(prediction, float(np.min(y)))
        self.assertLessEqual(prediction, float(np.max(y)))

    def test_estimate_semantic_distribution_returns_numeric_summary(self) -> None:
        result = estimate_semantic_distribution("regression", [245.0, 250.0, 260.0], [0.2, 0.5, 0.3])

        self.assertEqual(result["prediction_type"], "semantic_distribution_numeric")
        self.assertAlmostEqual(result["prediction_median"], 250.0)
        self.assertIsNotNone(result["distribution_summary"]["q90"])
