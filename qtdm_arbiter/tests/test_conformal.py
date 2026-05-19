from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.conformal import weighted_conformal_interval


class ConformalTests(unittest.TestCase):
    def test_weighted_conformal_interval_returns_expected_bounds(self) -> None:
        y_neighbors = np.array([10.0, 11.0, 9.5, 10.5, 10.2, 9.8], dtype=float)
        y_hat_neighbors = np.array([10.1, 10.8, 9.6, 10.4, 10.0, 9.9], dtype=float)
        weights = np.array([0.2, 0.18, 0.17, 0.16, 0.15, 0.14], dtype=float)

        interval = weighted_conformal_interval(y_neighbors, y_hat_neighbors, weights, y_hat_query=10.0, alpha=0.20)

        self.assertIsNotNone(interval)
        self.assertLessEqual(interval[0], 10.0)
        self.assertGreaterEqual(interval[1], 10.0)

    def test_weighted_conformal_interval_requires_minimum_labels(self) -> None:
        interval = weighted_conformal_interval(
            np.array([1.0, 2.0, np.nan, 4.0], dtype=float),
            np.array([1.1, 2.1, 3.0, 4.1], dtype=float),
            np.array([0.3, 0.3, 0.2, 0.2], dtype=float),
            y_hat_query=2.0,
            alpha=0.20,
        )
        self.assertIsNone(interval)


if __name__ == "__main__":
    unittest.main()
