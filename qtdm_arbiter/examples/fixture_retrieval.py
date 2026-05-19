"""In-memory retrieval backend for public demos and tests."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "examples" / "fixtures" / "exoplanet_cases.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_fixture_cases(path: Path = FIXTURE_PATH) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


class InMemoryDemoRetrievalClient:
    """Small deterministic retrieval client matching the public HTTP contract."""

    def __init__(self, base_url: str = "memory://fixtures", cases: List[Dict[str, Any]] | None = None) -> None:
        self.base_url = base_url
        self.cases = list(cases if cases is not None else load_fixture_cases())

    def search_findings(self, query: str, filters: dict, limit: int) -> List[Dict[str, Any]]:
        return self.search_content(query, filters, limit, data_types=None)

    def search_content(
        self,
        query: str,
        filters: dict,
        limit: int,
        data_types: list[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if "unsupported" in query.lower() or "bad query" in query.lower():
            return []
        scored: List[Dict[str, Any]] = []
        query_tokens = _tokens(query)
        for case in self.cases:
            case_tokens = _tokens(" ".join([str(case.get("chunk_text", "")), str(case.get("pl_name", "")), str(case.get("st_spectype", ""))]))
            lexical = _jaccard(query_tokens, case_tokens)
            feature_bonus = _feature_bonus(query, case)
            score = max(0.05, min(0.99, 0.45 + lexical * 0.65 + feature_bonus))
            item = dict(case)
            item["_semantic_score"] = round(score, 4)
            scored.append(item)
        scored.sort(key=lambda row: float(row["_semantic_score"]), reverse=True)
        return scored[:limit]

    def get_labels(self, finding_ids: list[str], target_name: str) -> dict[str, dict[str, float]]:
        labels: Dict[str, Dict[str, float]] = {}
        by_id = {str(case.get("chunk_id")): case for case in self.cases}
        for finding_id in finding_ids:
            case = by_id.get(str(finding_id))
            if case and case.get(target_name) is not None:
                labels[str(finding_id)] = {"value": float(case[target_name])}
        return labels


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _feature_bonus(query: str, case: Dict[str, Any]) -> float:
    lower = query.lower()
    bonus = 0.0
    if "m-type" in lower or "m dwarf" in lower or "red dwarf" in lower:
        bonus += 0.12 if str(case.get("st_spectype", "")).upper().startswith("M") else -0.04
    if "habitable" in lower or "temperate" in lower:
        bonus += 0.12 if case.get("hz_flag") else -0.08
    if "rocky" in lower or "terrestrial" in lower:
        radius = _float(case.get("pl_rade"))
        bonus += 0.10 if radius and radius < 1.8 else -0.06
    if "hot gas giant" in lower or "ultra hot" in lower:
        radius = _float(case.get("pl_rade"))
        eqt = _float(case.get("pl_eqt"))
        bonus += 0.18 if radius and eqt and radius > 8.0 and eqt > 800.0 else -0.05
    return bonus


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric

