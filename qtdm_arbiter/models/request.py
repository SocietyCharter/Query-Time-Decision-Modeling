from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ArbiterRequest(BaseModel):
    request_id: str
    target_type: Literal["regression", "binary_classification", "ranking"]
    target_name: str
    entity_type: Optional[str] = None
    query_summary: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    features: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)
    data_types: List[str] = Field(default_factory=list)
