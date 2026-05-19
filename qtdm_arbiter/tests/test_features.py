from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.features import CANONICAL_FEATURES, _impute_matrix, extract_feature_matrix, extract_query_vector
from qtdm_arbiter.models.request import ArbiterRequest


class _FakeRetrievalClient:
    def __init__(self, labels: dict[str, dict[str, float]]) -> None:
        self._labels = labels

    def get_labels(self, finding_ids: list[str], target_name: str) -> dict[str, dict[str, float]]:
        return {finding_id: self._labels[finding_id] for finding_id in finding_ids if finding_id in self._labels}


class FeatureTests(unittest.TestCase):
    def test_extract_feature_matrix_uses_proxy_target_when_label_coverage_is_thin(self) -> None:
        findings = [
            {
                "finding_id": "f-1",
                "confidence": 0.8,
                "scores": {"relevance": 0.9, "value": 0.6},
                "retrieval_role": "primary",
                "collection_status": "retained",
                "capability_maturity": "established",
                "entities": ["a", "b"],
                "related_domains": ["x.com"],
                "timestamps": ["2026-01-01"],
                "completeness": 0.9,
                "trust_for_ranking": 0.77,
            },
            {
                "finding_id": "f-2",
                "confidence": 0.7,
                "scores": {"relevance": 0.8, "value": 0.5},
                "retrieval_role": "supporting",
                "collection_status": "queued_for_excavation",
                "capability_maturity": "emerging",
                "entities": [],
                "related_domains": ["y.com", "z.com"],
                "timestamps": [],
                "completeness": 0.7,
                "trust_for_ranking": 0.42,
            },
        ]

        X, y, metadata = extract_feature_matrix(
            findings,
            "lead_quality",
            _FakeRetrievalClient({"f-1": {"value": 1.0}}),
            proxy_threshold=0.6,
        )

        self.assertEqual(X.shape, (2, len(CANONICAL_FEATURES)))
        self.assertTrue(np.allclose(y, np.array([0.77, 0.42])))
        self.assertTrue(metadata["proxy_target"])
        self.assertAlmostEqual(metadata["real_label_coverage"], 0.5)

    def test_extract_query_vector_imputes_missing_values_from_neighborhood(self) -> None:
        neighborhood = np.zeros((2, len(CANONICAL_FEATURES)), dtype=float)
        neighborhood[:, 0] = np.array([0.2, 0.6], dtype=float)
        neighborhood[:, 1] = np.array([0.4, 0.8], dtype=float)
        request = ArbiterRequest(
            request_id="arb-1",
            target_type="regression",
            target_name="lead_quality",
            query_summary="test",
            features={"confidence": 0.5},
        )

        vector = extract_query_vector(request, neighborhood)

        self.assertAlmostEqual(vector[0], 0.5)
        self.assertAlmostEqual(vector[1], 0.6)

    def test_extract_query_vector_prefers_request_physical_features_and_matches_dropped_target_width(self) -> None:
        neighborhood = np.zeros((3, len(CANONICAL_FEATURES) - 1), dtype=float)
        request = ArbiterRequest(
            request_id="arb-2",
            target_type="regression",
            target_name="pl_eqt",
            query_summary="temperate rocky planet",
            features={"pl_rade": 1.03, "st_teff": 2560.0},
        )

        feature_names = [feature for feature in CANONICAL_FEATURES if feature != "pl_eqt"]
        vector = extract_query_vector(request, neighborhood, feature_names=feature_names)

        self.assertEqual(len(vector), len(CANONICAL_FEATURES) - 1)
        self.assertAlmostEqual(vector[feature_names.index("pl_rade")], 1.03)
        self.assertAlmostEqual(vector[feature_names.index("st_teff")], 2560.0)

    def test_impute_matrix_uses_medians_and_flags_structural_missingness(self) -> None:
        X = np.array(
            [
                [1.0, np.nan, np.nan],
                [3.0, np.nan, np.nan],
                [9.0, np.nan, np.nan],
                [11.0, np.nan, np.nan],
                [12.0, np.nan, np.nan],
                [13.0, 5.0, np.nan],
            ],
            dtype=float,
        )

        imputed, completeness, structurally_missing = _impute_matrix(X)

        self.assertAlmostEqual(completeness, 7.0 / 18.0)
        self.assertEqual(structurally_missing, [1, 2])
        self.assertAlmostEqual(imputed[0, 1], 5.0)
        self.assertAlmostEqual(imputed[1, 0], 3.0)

    def test_extract_feature_matrix_reports_structurally_missing_features(self) -> None:
        findings = [
            {
                "finding_id": "f-1",
                "confidence": 0.8,
                "scores": {"relevance": 0.9, "value": 0.6},
                "retrieval_role": "primary",
                "collection_status": "retained",
                "capability_maturity": "established",
                "entities": ["a"],
                "related_domains": ["x.com"],
                "timestamps": [],
                "completeness": 0.9,
                "trust_for_ranking": 0.77,
                "page2_exists": 1.0,
            },
            {
                "finding_id": "f-2",
                "confidence": 0.7,
                "scores": {"relevance": 0.8, "value": 0.5},
                "retrieval_role": "supporting",
                "collection_status": "queued_for_excavation",
                "capability_maturity": "emerging",
                "entities": [],
                "related_domains": ["y.com"],
                "timestamps": [],
                "completeness": 0.7,
                "trust_for_ranking": 0.42,
            },
        ]

        _, _, metadata = extract_feature_matrix(
            findings,
            "lead_quality",
            _FakeRetrievalClient({"f-1": {"value": 1.0}, "f-2": {"value": 0.0}}),
            proxy_threshold=0.1,
        )

        self.assertIn("page3_exists", metadata["structurally_missing_features"])
