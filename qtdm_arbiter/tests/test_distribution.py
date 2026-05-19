from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.distribution import (
    categorical_distribution_summary,
    mix_weighted_values,
    numeric_distribution_overlap,
    numeric_distribution_summary,
    weighted_quantile,
)


class DistributionTests(unittest.TestCase):
    def test_weighted_quantile_prefers_high_weight_mass(self) -> None:
        q50 = weighted_quantile([10.0, 20.0, 100.0], [0.1, 0.8, 0.1], 0.5)
        self.assertAlmostEqual(q50, 20.0)

    def test_numeric_distribution_summary_reports_quantiles(self) -> None:
        summary = numeric_distribution_summary([240.0, 250.0, 260.0, 255.0], [0.2, 0.4, 0.2, 0.2])
        self.assertAlmostEqual(summary["median"], 250.0)
        self.assertGreater(summary["q90"], summary["q10"])

    def test_numeric_distribution_overlap_is_low_for_separated_distributions(self) -> None:
        overlap = numeric_distribution_overlap([240.0, 245.0, 250.0], [0.3, 0.4, 0.3], [900.0, 920.0, 940.0], [0.3, 0.4, 0.3])
        self.assertLess(overlap, 0.2)

    def test_categorical_distribution_summary_computes_margin(self) -> None:
        summary = categorical_distribution_summary(["a", "a", "b"], [0.4, 0.4, 0.2])
        self.assertEqual(summary["top_class"], "a")
        self.assertGreater(summary["class_margin"], 0.0)

    def test_mix_weighted_values_respects_gamma(self) -> None:
        values, weights = mix_weighted_values([1, 2], [0.6, 0.4], [10], [1.0], 0.5)
        self.assertEqual(list(values), [1, 2, 10])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
