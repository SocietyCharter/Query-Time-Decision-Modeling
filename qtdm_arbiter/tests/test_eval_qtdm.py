from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdm_arbiter.models.response import ArbiterResponse
from qtdm_arbiter.tools.eval_qtdm import evaluate_cases, load_test_set


class EvalQTDMTests(unittest.TestCase):
    def test_load_test_set_reads_jsonl_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "rocky planet", "target_type": "regression", "target_name": "pl_eqt", "true_value": 251.0}),
                        json.dumps({"query": "binary route", "target_type": "binary_classification", "target_name": "service_exists", "true_value": 1.0}),
                    ]
                ),
                encoding="utf-8",
            )
            cases = load_test_set(path)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["target_name"], "pl_eqt")

    def test_evaluate_cases_computes_semantic_summary_metrics(self) -> None:
        cases = [
            {"query": "rocky planet", "target_type": "regression", "target_name": "pl_eqt", "true_value": 250.0, "policy": {}},
            {"query": "support only", "target_type": "regression", "target_name": "pl_eqt", "true_value": 260.0, "policy": {}},
            {"query": "binary flag", "target_type": "binary_classification", "target_name": "service_exists", "true_value": 1.0, "policy": {}},
        ]
        responses = iter(
            [
                ArbiterResponse(
                    request_id="eval-1",
                    status="ok",
                    prediction=248.0,
                    prediction_low=240.0,
                    prediction_high=260.0,
                    prediction_type="semantic_distribution_numeric",
                    confidence=0.8,
                    support_score=0.8,
                    model_used="semantic_distribution",
                    neighbors_requested=25,
                    neighbors_used=20,
                    evidence_case_ids=["a"],
                ),
                ArbiterResponse(
                    request_id="eval-2",
                    status="semantic_support_only",
                    prediction=None,
                    prediction_type="semantic_support_only",
                    confidence=0.2,
                    support_score=0.2,
                    model_used="semantic_support_only",
                    neighbors_requested=25,
                    neighbors_used=7,
                    evidence_case_ids=["b"],
                    refusal_reason="no_real_labels",
                ),
                ArbiterResponse(
                    request_id="eval-3",
                    status="needs_escalation",
                    prediction=0.9,
                    prediction_type="semantic_distribution_binary",
                    confidence=0.6,
                    support_score=0.6,
                    model_used="semantic_distribution",
                    neighbors_requested=25,
                    neighbors_used=18,
                    evidence_case_ids=["c"],
                ),
            ]
        )

        report = evaluate_cases(cases, retrieval_url="http://stubbed", run_request_fn=lambda request, retrieval_url: next(responses))

        self.assertEqual(report["n_cases"], 3)
        self.assertAlmostEqual(report["mae"], 2.0)
        self.assertAlmostEqual(report["rmse"], 2.0)
        self.assertAlmostEqual(report["coverage_80"], 1.0)
        self.assertAlmostEqual(report["semantic_support_only_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(report["needs_escalation_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(report["mean_support"], 0.7)
        self.assertEqual(len(report["per_case"]), 3)


if __name__ == "__main__":
    unittest.main()
