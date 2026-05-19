from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.core.neighborhood import pull_neighborhood_content
from qtdm_arbiter.models.request import ArbiterRequest


class _StubRetrievalClient:
    def search_content(self, query: str, filters: dict, limit: int, data_types: list[str] | None = None) -> list[dict]:
        return [
            {
                "chunk_id": "far-semantic",
                "_semantic_score": 0.92,
                "pl_rade": 8.0,
                "st_teff": 6400.0,
                "pl_orbsmax": 0.03,
                "pl_eqt": 1700.0,
            },
            {
                "chunk_id": "close-physical",
                "_semantic_score": 0.88,
                "pl_rade": 1.0,
                "st_teff": 2550.0,
                "pl_orbsmax": 0.029,
                "pl_eqt": 255.0,
            },
        ]


class NeighborhoodTests(unittest.TestCase):
    def test_pull_neighborhood_content_feature_reranks_by_physical_similarity(self) -> None:
        request = ArbiterRequest(
            request_id="arb-content",
            target_type="regression",
            target_name="pl_eqt",
            entity_type="content",
            query_summary="temperate rocky planet",
            data_types=["exoplanet_record"],
            features={"pl_rade": 1.03, "st_teff": 2560.0, "pl_orbsmax": 0.029},
            policy={"feature_rerank": True, "rerank_alpha": 0.3, "rerank_beta": 0.6, "rerank_gamma": 0.1},
        )

        neighbors = pull_neighborhood_content(
            request,
            retrieval_url="http://stubbed",
            k=2,
            target_name="pl_eqt",
            client=_StubRetrievalClient(),
        )

        self.assertEqual(neighbors[0]["finding_id"], "close-physical")
        self.assertGreater(neighbors[0]["_rerank_score"], neighbors[1]["_rerank_score"])


if __name__ == "__main__":
    unittest.main()
