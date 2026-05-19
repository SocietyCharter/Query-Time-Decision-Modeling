from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.confidence import compute_support, compute_weights
from qtdm_arbiter.core.confidence import _adaptive_lambda


class ConfidenceTests(unittest.TestCase):
    def test_adaptive_lambda_is_gentle_for_flat_scores(self) -> None:
        lam = _adaptive_lambda([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(lam, 1.0)

    def test_adaptive_lambda_changes_with_score_spread(self) -> None:
        tight = _adaptive_lambda([0.9, 0.91, 0.905, 0.908, 0.907])
        spread = _adaptive_lambda([0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertGreaterEqual(tight, 0.5)
        self.assertLessEqual(tight, 10.0)
        self.assertGreaterEqual(spread, 0.5)
        self.assertLessEqual(spread, 10.0)
        self.assertNotEqual(tight, spread)

    def test_compute_weights_uses_adaptive_lambda_by_default(self) -> None:
        weights = compute_weights([0.9, 0.88, 0.87], strategy="softmax", lam=None)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertGreater(weights[0], weights[-1])

    def test_compute_weights_respects_explicit_lambda_override(self) -> None:
        adaptive = compute_weights([0.9, 0.7, 0.5], strategy="softmax", lam=None)
        explicit = compute_weights([0.9, 0.7, 0.5], strategy="softmax", lam=5.0)
        self.assertAlmostEqual(float(np.sum(explicit)), 1.0)
        self.assertFalse(np.allclose(adaptive, explicit))

    def test_compute_weights_supports_adaptive_temperature(self) -> None:
        weights = compute_weights([0.91, 0.88, 0.73], strategy="adaptive_temp", tau=0.05)
        effective_k = 1.0 / float(np.sum(weights ** 2))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertLess(effective_k, 3.0)

    def test_compute_support_returns_expected_diagnostics(self) -> None:
        X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=float)
        y = np.array([0.2, 0.25, 0.3], dtype=float)

        support, diagnostics = compute_support(
            3,
            [0.9, 0.85, 0.83],
            X,
            y,
            0.24,
            0.25,
            feature_completeness=0.9,
            target_coverage=1.0,
        )

        self.assertGreater(support, 0.0)
        self.assertIn("sim_coherence", diagnostics)
        self.assertAlmostEqual(diagnostics["feature_completeness"], 0.9)
