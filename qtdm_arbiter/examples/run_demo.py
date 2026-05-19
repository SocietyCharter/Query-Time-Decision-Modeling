"""Run the public QTDM proof-of-concept demo without external services."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

from qtdm_arbiter import arbiter_decide
from qtdm_arbiter.examples.fixture_retrieval import InMemoryDemoRetrievalClient
from qtdm_arbiter.models.request import ArbiterRequest


@contextmanager
def _patched_demo_backend() -> Iterator[None]:
    original_client = arbiter_decide.RetrievalClient
    original_audit = arbiter_decide.write_audit
    arbiter_decide.RetrievalClient = InMemoryDemoRetrievalClient
    arbiter_decide.write_audit = lambda request, response: None
    try:
        yield
    finally:
        arbiter_decide.RetrievalClient = original_client
        arbiter_decide.write_audit = original_audit


def main() -> int:
    request = ArbiterRequest(
        request_id="demo-exoplanet-eqt",
        target_type="regression",
        target_name="pl_eqt",
        entity_type="content",
        query_summary="temperate rocky planet around an M-type star in the habitable zone",
        data_types=["exoplanet_record"],
        policy={
            "mode": "semantic_distribution",
            "domain": "exoplanet",
            "k": 6,
            "min_support": 0.2,
            "semantic_tilt": True,
            "mock_semantic_tilt": -0.35,
        },
    )
    bad_request = ArbiterRequest(
        request_id="demo-refusal",
        target_type="regression",
        target_name="pl_eqt",
        entity_type="content",
        query_summary="unsupported bad query with no comparable precedent",
        data_types=["exoplanet_record"],
        policy={"mode": "semantic_distribution", "domain": "exoplanet", "k": 6, "min_support": 0.2},
    )

    with _patched_demo_backend():
        response = arbiter_decide.run_request(request, retrieval_url="memory://fixtures")
        refusal = arbiter_decide.run_request(bad_request, retrieval_url="memory://fixtures")

    print("# QTDM local demo")
    print()
    print("## request")
    print(json.dumps(request.model_dump(), indent=2))
    print()
    print("## retrieved cases")
    for case_id in response.evidence_case_ids:
        print(f"- {case_id}")
    print()
    print("## weighted distribution summary")
    summary = response.distribution_summary
    for key in ("sample_count", "mean", "median", "q10", "q90", "width_80"):
        print(f"- {key}: {summary.get(key)}")
    print()
    print("## prediction")
    print(f"- prediction: {response.prediction}")
    print(f"- interval: [{response.prediction_low}, {response.prediction_high}]")
    print(f"- model_used: {response.model_used}")
    print()
    print("## confidence/support diagnostics")
    print(json.dumps(response.confidence_components, indent=2))
    print(json.dumps({key: response.diagnostics.get(key) for key in ("n_usable", "sim_coherence", "real_label_coverage", "semantic_tilt", "semantic_quantile", "tilt_path")}, indent=2))
    print()
    print("## evidence case IDs")
    print(", ".join(response.evidence_case_ids))
    print()
    print("## refusal behavior")
    print(json.dumps({"status": refusal.status, "refusal_reason": refusal.refusal_reason, "neighbors_used": refusal.neighbors_used}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

