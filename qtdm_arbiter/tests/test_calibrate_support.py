from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.tools.calibrate_support import calibrate_support_weights


class CalibrateSupportTests(unittest.TestCase):
    def test_calibrate_support_weights_returns_normalized_weights(self) -> None:
        payload = {
            "per_case": [
                {
                    "error_abs": 0.10,
                    "diagnostics": {
                        "support_components": {
                            "neighbor_count": 0.9,
                            "sim_coherence": 0.7,
                            "feature_completeness": 0.8,
                            "target_variance": 0.6,
                            "estimator_agreement": 0.5,
                        }
                    },
                },
                {
                    "error_abs": 0.35,
                    "diagnostics": {
                        "support_components": {
                            "neighbor_count": 0.4,
                            "sim_coherence": 0.3,
                            "feature_completeness": 0.5,
                            "target_variance": 0.2,
                            "estimator_agreement": 0.4,
                        }
                    },
                },
                {
                    "error_abs": 0.22,
                    "diagnostics": {
                        "support_components": {
                            "neighbor_count": 0.6,
                            "sim_coherence": 0.55,
                            "feature_completeness": 0.65,
                            "target_variance": 0.35,
                            "estimator_agreement": 0.45,
                        }
                    },
                },
            ]
        }

        weights = calibrate_support_weights(payload)

        self.assertEqual(set(weights.keys()), {"neighbor_count", "sim_coherence", "feature_completeness", "target_variance", "estimator_agreement"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
