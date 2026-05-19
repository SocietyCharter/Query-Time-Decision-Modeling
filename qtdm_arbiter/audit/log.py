from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from qtdm_arbiter.models.request import ArbiterRequest
from qtdm_arbiter.models.response import ArbiterResponse


DEFAULT_AUDIT_LOG = os.environ.get(
    "QTDM_AUDIT_LOG",
    str(Path(__file__).resolve().parents[2] / "runtime" / "logs" / "arbiter_audit.jsonl"),
)


def write_audit(request: ArbiterRequest, response: ArbiterResponse, log_path: str | None = None) -> None:
    path = Path(log_path or DEFAULT_AUDIT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request_id": request.request_id,
        "query_hash": hashlib.sha256(request.query_summary.encode("utf-8")).hexdigest()[:12],
        "entity_type": request.entity_type,
        "target_name": request.target_name,
        "model_selected": response.model_used,
        "neighbor_ids": response.evidence_case_ids,
        "prediction": response.prediction,
        "confidence": response.confidence,
        "support_score": response.support_score,
        "fallback_used": response.fallback_used,
        "refusal_reason": response.refusal_reason,
        "status": response.status,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
