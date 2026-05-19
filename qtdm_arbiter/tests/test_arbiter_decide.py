from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.arbiter_decide import run_request
from qtdm_arbiter.models.request import ArbiterRequest


class _SemanticStubRetrievalClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def search_content(self, query: str, filters: dict, limit: int, data_types: list[str] | None = None) -> list[dict]:
        lower = query.lower()
        if "contrast" in lower or "alternative" in lower or "gas giant" in lower:
            return [
                {"chunk_id": "c-1", "_semantic_score": 0.78, "chunk_text": "counter hot giant", "pl_eqt": 940.0},
                {"chunk_id": "c-2", "_semantic_score": 0.76, "chunk_text": "counter warm giant", "pl_eqt": 880.0},
                {"chunk_id": "c-3", "_semantic_score": 0.73, "chunk_text": "counter giant", "pl_eqt": 820.0},
                {"chunk_id": "c-4", "_semantic_score": 0.70, "chunk_text": "counter case", "pl_eqt": 980.0},
                {"chunk_id": "c-5", "_semantic_score": 0.69, "chunk_text": "counter case", "pl_eqt": 900.0},
            ]
        if "pl eqt" in lower or "content" in lower:
            return [
                {"chunk_id": "b-1", "_semantic_score": 0.80, "chunk_text": "baseline warm planet", "pl_eqt": 720.0},
                {"chunk_id": "b-2", "_semantic_score": 0.78, "chunk_text": "baseline warm planet", "pl_eqt": 680.0},
                {"chunk_id": "b-3", "_semantic_score": 0.76, "chunk_text": "baseline warm planet", "pl_eqt": 760.0},
                {"chunk_id": "b-4", "_semantic_score": 0.74, "chunk_text": "baseline warm planet", "pl_eqt": 710.0},
                {"chunk_id": "b-5", "_semantic_score": 0.72, "chunk_text": "baseline warm planet", "pl_eqt": 735.0},
            ]
        if "m-type" in lower or "rocky" in lower or "habitable" in lower:
            return [
                {"chunk_id": "i-1", "_semantic_score": 0.94, "chunk_text": "temperate rocky world", "pl_eqt": 250.0},
                {"chunk_id": "i-2", "_semantic_score": 0.93, "chunk_text": "temperate rocky world", "pl_eqt": 260.0},
                {"chunk_id": "i-3", "_semantic_score": 0.92, "chunk_text": "temperate rocky world", "pl_eqt": 245.0},
                {"chunk_id": "i-4", "_semantic_score": 0.91, "chunk_text": "temperate rocky world", "pl_eqt": 255.0},
                {"chunk_id": "i-5", "_semantic_score": 0.90, "chunk_text": "temperate rocky world", "pl_eqt": 240.0},
                {"chunk_id": "i-6", "_semantic_score": 0.89, "chunk_text": "temperate rocky world", "pl_eqt": 265.0},
            ]
        if "nolabel" in lower:
            return [
                {"chunk_id": "n-1", "_semantic_score": 0.91, "chunk_text": "unknown precedent"},
                {"chunk_id": "n-2", "_semantic_score": 0.89, "chunk_text": "unknown precedent"},
                {"chunk_id": "n-3", "_semantic_score": 0.87, "chunk_text": "unknown precedent"},
                {"chunk_id": "n-4", "_semantic_score": 0.85, "chunk_text": "unknown precedent"},
                {"chunk_id": "n-5", "_semantic_score": 0.83, "chunk_text": "unknown precedent"},
            ]
        return [
            {"chunk_id": "q-1", "_semantic_score": 0.95, "chunk_text": "query precedent", "pl_eqt": 248.0},
            {"chunk_id": "q-2", "_semantic_score": 0.94, "chunk_text": "query precedent", "pl_eqt": 252.0},
            {"chunk_id": "q-3", "_semantic_score": 0.93, "chunk_text": "query precedent", "pl_eqt": 256.0},
            {"chunk_id": "q-4", "_semantic_score": 0.92, "chunk_text": "query precedent", "pl_eqt": 245.0},
            {"chunk_id": "q-5", "_semantic_score": 0.91, "chunk_text": "query precedent", "pl_eqt": 259.0},
            {"chunk_id": "q-6", "_semantic_score": 0.90, "chunk_text": "query precedent", "pl_eqt": 243.0},
        ]

    def get_labels(self, finding_ids: list[str], target_name: str) -> dict[str, dict[str, float]]:
        return {}


class ArbiterDecideTests(unittest.TestCase):
    @patch("qtdm_arbiter.arbiter_decide.write_audit")
    @patch("qtdm_arbiter.arbiter_decide.RetrievalClient", _SemanticStubRetrievalClient)
    def test_run_request_returns_semantic_distribution_response(self, _write_audit) -> None:
        request = ArbiterRequest(
            request_id="arb-semantic",
            target_type="regression",
            target_name="pl_eqt",
            entity_type="content",
            query_summary="M-type star habitable zone rocky planet",
            data_types=["exoplanet_record"],
            policy={"mode": "semantic_distribution", "k": 6, "counter_queries": ["hot gas giant close orbit"]},
        )

        response = run_request(request, retrieval_url="http://stubbed")

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.model_used, "semantic_distribution")
        self.assertEqual(response.prediction_type, "semantic_distribution_numeric")
        self.assertIsNotNone(response.prediction)
        self.assertTrue(response.supporting_case_ids)
        self.assertIn("median", response.distribution_summary)
        self.assertIn("overlap_with_base", response.comparator_summary)
        self.assertIn("final_confidence", response.confidence_components)

    @patch("qtdm_arbiter.arbiter_decide.write_audit")
    @patch("qtdm_arbiter.arbiter_decide.RetrievalClient", _SemanticStubRetrievalClient)
    def test_run_request_returns_semantic_support_only_without_labels(self, _write_audit) -> None:
        request = ArbiterRequest(
            request_id="arb-support-only",
            target_type="regression",
            target_name="pl_eqt",
            entity_type="content",
            query_summary="nolabel precedent search",
            data_types=["exoplanet_record"],
            policy={"mode": "semantic_distribution", "k": 5},
        )

        response = run_request(request, retrieval_url="http://stubbed")

        self.assertEqual(response.status, "semantic_support_only")
        self.assertIsNone(response.prediction)
        self.assertEqual(response.refusal_reason, "no_real_labels")
        self.assertGreaterEqual(response.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
