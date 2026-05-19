from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_serializer


class ArbiterResponse(BaseModel):
    request_id: str
    status: Literal["ok", "fallback", "refused", "semantic_support_only", "needs_escalation", "fuzzy_estimate"]
    prediction: Optional[Any] = None
    prediction_mean: Optional[float] = None
    prediction_median: Optional[float] = None
    prediction_mode: Optional[float] = None
    prediction_low: Optional[float] = None
    prediction_high: Optional[float] = None
    calibration_low: Optional[float] = None
    calibration_high: Optional[float] = None
    prediction_type: Optional[str] = None
    confidence: float
    support_score: float
    model_used: Optional[str] = None
    neighbors_requested: int
    neighbors_used: int
    evidence_case_ids: List[str]
    supporting_case_ids: List[str] = Field(default_factory=list)
    counter_case_ids: List[str] = Field(default_factory=list)
    fallback_used: bool = False
    refusal_reason: Optional[str] = None
    distribution_summary: Dict[str, Any] = Field(default_factory=dict)
    comparator_summary: Dict[str, Any] = Field(default_factory=dict)
    confidence_components: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[Dict[str, Any]] = None
    semantic_mode: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_estimate: Optional[float] = None
    reasoning_model: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def model_dump_json(self, **kwargs) -> str:  # type: ignore[override]
        import json
        return json.dumps(_sanitize(self.model_dump(**kwargs)))


def _sanitize(obj: Any) -> Any:
    """Recursively coerce numpy/non-JSON-serializable types to Python natives."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
