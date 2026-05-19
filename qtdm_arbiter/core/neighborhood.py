from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from qtdm_arbiter.core.features import PHYSICAL_QUERY_FEATURES
from qtdm_arbiter.integration.retrieval_client import RetrievalClient
from qtdm_arbiter.models.request import ArbiterRequest


REQUIRED_FINDING_FIELDS = (
    "finding_id",
    "_result_score",
    "confidence",
    "scores",
    "retrieval_role",
    "collection_status",
    "capability_maturity",
    "completeness",
    "trust_for_ranking",
)

_STOPWORDS = {
    "a", "an", "and", "or", "the", "to", "for", "with", "of", "in", "on",
    "at", "by", "from", "various", "different", "measured", "near", "above",
    "below", "current", "historical", "case", "cases", "query", "target",
}

_COUNTER_TOKEN_MAP = {
    "rocky": "gas giant",
    "gas": "rocky",
    "habitable": "non habitable",
    "temperate": "hot",
    "warm": "cold",
    "cool": "hot",
    "close": "far",
    "high": "low",
    "low": "high",
    "success": "failure",
    "successful": "failed",
}


def pull_neighborhood(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
    query_override: str | None = None,
    filters_override: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    retrieval_client = client or RetrievalClient(retrieval_url)
    filters = dict(filters_override or request.filters)
    filters.setdefault("min_confidence", 0.5)
    if request.entity_type:
        filters["entity_type"] = request.entity_type
    limit = min(max(k * 2, k), 80)
    return retrieval_client.search_findings(query_override or request.query_summary, filters=filters, limit=limit)


def pull_neighborhood_content(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    target_name: str,
    client: RetrievalClient | None = None,
    query_override: str | None = None,
    filters_override: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    retrieval_client = client or RetrievalClient(retrieval_url)
    filters = {key: value for key, value in (filters_override or request.filters).items() if key != "min_confidence"}
    limit = min(max(k * 2, k), 100)
    raw = retrieval_client.search_content(
        query_override or request.query_summary,
        filters=filters,
        limit=limit,
        data_types=request.data_types or None,
    )

    normalized: List[Dict[str, Any]] = []
    for chunk in raw:
        semantic_score = float(chunk.get("_semantic_score", 0.0) or 0.0)
        label_value = _extract_label(chunk, target_name)
        adapted: Dict[str, Any] = {
            "finding_id": chunk.get("chunk_id", chunk.get("source_id", "")),
            "source_id": chunk.get("source_id", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "_result_score": round(semantic_score, 4),
            "confidence": round(min(semantic_score + 0.1, 1.0), 4),
            "trust_for_ranking": round(semantic_score, 4),
            "completeness": _coerce_float(chunk.get("completeness"), default=1.0),
            "retrieval_role": "primary",
            "collection_status": "collected",
            "capability_maturity": "validated",
            "scores": {
                "relevance": round(semantic_score, 4),
                "confidence": round(min(semantic_score + 0.1, 1.0), 4),
                "value": (
                    round(min(max(label_value / max(abs(label_value), 1.0), -1.0), 1.0), 4)
                    if label_value is not None
                    else 0.5
                ),
                "rarity": 0.5,
                "freshness": 0.5,
                "exposure": 0.5,
                "body_signal_ratio": 0.8,
            },
            "_label_value": label_value,
            "_label_name": target_name,
            "entity_type": "content",
            "entities": [],
            "related_domains": [],
            "timestamps": [],
        }
        merged = dict(chunk)
        merged.update(adapted)
        normalized.append(merged)

    if bool(request.policy.get("feature_rerank", False)):
        _apply_feature_rerank(normalized, request)
    normalized.sort(key=lambda x: float(x.get("_result_score", 0.0) or 0.0), reverse=True)
    if bool(request.policy.get("feature_rerank", False)):
        normalized.sort(key=lambda x: float(x.get("_rerank_score", x.get("_result_score", 0.0)) or 0.0), reverse=True)
    return normalized


def build_query_neighborhood(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
) -> List[Dict[str, Any]]:
    return _search_and_gate(request, retrieval_url, k, client=client, query=request.query_summary)


def build_base_neighborhood(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
) -> List[Dict[str, Any]]:
    base_query = str(request.policy.get("base_query") or _build_base_query(request))
    return _search_and_gate(request, retrieval_url, k, client=client, query=base_query)


def build_facet_neighborhoods(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
) -> List[Dict[str, Any]]:
    facet_queries = list(request.policy.get("facet_queries", []) or _extract_facet_queries(request.query_summary))
    neighborhoods: List[Dict[str, Any]] = []
    for facet_query in facet_queries[:3]:
        cases = _search_and_gate(request, retrieval_url, max(6, min(k, 12)), client=client, query=facet_query)
        neighborhoods.append({"facet_query": facet_query, "cases": cases})
    return neighborhoods


def build_intersection_neighborhood(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
) -> List[Dict[str, Any]]:
    intersection_query = str(request.policy.get("intersection_query") or _build_intersection_query(request.query_summary))
    return _search_and_gate(request, retrieval_url, k, client=client, query=intersection_query)


def build_counter_neighborhood(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    client: RetrievalClient | None = None,
) -> List[Dict[str, Any]]:
    counter_queries = list(request.policy.get("counter_queries", []) or _build_counter_queries(request.query_summary, request.target_name))
    cases: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for counter_query in counter_queries[:3]:
        for case in _search_and_gate(request, retrieval_url, max(6, min(k, 12)), client=client, query=counter_query):
            case_id = str(case.get("finding_id", ""))
            if case_id and case_id not in seen:
                seen.add(case_id)
                cases.append(case)
    cases.sort(key=lambda item: float(item.get("_result_score", 0.0) or 0.0), reverse=True)
    return cases[:k]


def gate_neighborhood(findings: List[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
    usable: List[Dict[str, Any]] = []
    for finding in findings:
        if float(finding.get("confidence", 0.0) or 0.0) < float(min_confidence):
            continue
        if any(field not in finding or finding.get(field) is None for field in REQUIRED_FINDING_FIELDS):
            continue
        candidate = dict(finding)
        candidate.setdefault("entities", [])
        candidate.setdefault("related_domains", [])
        candidate.setdefault("timestamps", [])
        candidate.setdefault("entity_type", candidate.get("category"))
        usable.append(candidate)
    usable.sort(key=lambda item: float(item.get("_result_score", 0.0) or 0.0), reverse=True)
    return usable


def validate_neighborhood(gated: List[Dict[str, Any]], min_k: int) -> Tuple[List[Dict[str, Any]], str | None]:
    if len(gated) < int(min_k):
        return gated, "insufficient_neighbors"
    return gated, None


def _search_and_gate(
    request: ArbiterRequest,
    retrieval_url: str,
    k: int,
    *,
    client: RetrievalClient | None,
    query: str,
) -> List[Dict[str, Any]]:
    is_content = (request.entity_type or "").lower() == "content"
    if is_content:
        raw = pull_neighborhood_content(request, retrieval_url, k, target_name=request.target_name, client=client, query_override=query)
        return gate_neighborhood(raw, min_confidence=0.0)[:k]
    raw = pull_neighborhood(request, retrieval_url, k, client=client, query_override=query)
    return gate_neighborhood(raw, min_confidence=float(request.filters.get("min_confidence", 0.5)))[:k]


def _build_base_query(request: ArbiterRequest) -> str:
    parts: List[str] = []
    if request.entity_type:
        parts.append(str(request.entity_type))
    if request.data_types:
        parts.extend(str(item).replace("_", " ") for item in request.data_types[:2])
    parts.append(str(request.target_name).replace("_", " "))
    return " ".join(part for part in parts if part).strip() or request.query_summary


def _extract_facet_queries(query: str) -> List[str]:
    tokens = [token.strip(" ,.;:()[]{}").lower() for token in query.split()]
    usable = [token for token in tokens if token and token not in _STOPWORDS and len(token) > 2]
    unique: List[str] = []
    for token in usable:
        if token not in unique:
            unique.append(token)
    if len(unique) >= 3:
        return [" ".join(unique[:2]), " ".join(unique[1:3]), unique[0]]
    return unique[:3]


def _build_intersection_query(query: str) -> str:
    facets = _extract_facet_queries(query)
    if not facets:
        return query
    return " ".join(facets[:3])


def _build_counter_queries(query: str, target_name: str) -> List[str]:
    lower = query.lower()
    counter_queries: List[str] = []
    for token, replacement in _COUNTER_TOKEN_MAP.items():
        if token in lower:
            counter_queries.append(lower.replace(token, replacement))
    if not counter_queries:
        counter_queries.append(f"contrast {query}")
    counter_queries.append(f"alternative {target_name.replace('_', ' ')} precedent")
    deduped: List[str] = []
    for item in counter_queries:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


_DELAY_PATTERNS = {
    "arr_delay": ["arrival delay", "arr_delay"],
    "dep_delay": ["departure delay", "dep_delay"],
    "cancelled": ["cancelled"],
}

_DENSITY_TARGETS = {
    "flight_count",
    "flight_count_above_100",
    "route_frequency",
    "service_exists",
}


def _extract_label(chunk: Dict[str, Any], target_name: str) -> float | None:
    raw = chunk.get(target_name)
    if raw is not None and raw != "":
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return _extract_bts_label(chunk, target_name)


def _extract_bts_label(chunk: Dict[str, Any], target_name: str) -> float | None:
    import re

    if target_name in _DENSITY_TARGETS:
        score = chunk.get("_semantic_score", chunk.get("_result_score", None))
        if score is not None:
            return float(score)
        return 0.5

    text = chunk.get("chunk_text", "").lower()
    if target_name == "cancelled":
        return 1.0 if "cancelled" in text else 0.0
    patterns = _DELAY_PATTERNS.get(target_name, [])
    for pat in patterns:
        match = re.search(rf"{re.escape(pat)}[:\s]+(-?\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
    return None


def _apply_feature_rerank(neighbors: List[Dict[str, Any]], request: ArbiterRequest) -> None:
    query_features = request.features or {}
    shared_query = {
        feature: _coerce_float(query_features.get(feature))
        for feature in PHYSICAL_QUERY_FEATURES
        if query_features.get(feature) is not None
    }
    shared_query = {feature: value for feature, value in shared_query.items() if not math.isnan(value)}
    if not shared_query:
        return

    alpha = float(request.policy.get("rerank_alpha", 0.3))
    beta = float(request.policy.get("rerank_beta", 0.6))
    gamma = float(request.policy.get("rerank_gamma", 0.1))

    for neighbor in neighbors:
        feature_similarity, missingness = _feature_similarity(shared_query, neighbor)
        semantic_similarity = float(neighbor.get("_result_score", 0.0) or 0.0)
        final_score = alpha * semantic_similarity + beta * feature_similarity - gamma * missingness
        neighbor["_feature_similarity"] = round(feature_similarity, 4)
        neighbor["_feature_missingness"] = round(missingness, 4)
        neighbor["_rerank_score"] = round(final_score, 4)


def _feature_similarity(query_features: Dict[str, float], neighbor: Dict[str, Any]) -> Tuple[float, float]:
    shared_diffs: List[float] = []
    missing = 0
    for feature, query_value in query_features.items():
        neighbor_value = _coerce_float(neighbor.get(feature))
        if math.isnan(neighbor_value):
            missing += 1
            continue
        scale = max(abs(query_value), abs(neighbor_value), 1.0)
        shared_diffs.append(((query_value - neighbor_value) / scale) ** 2)
    total = len(query_features)
    if not shared_diffs:
        return 0.0, 1.0 if total else 0.0
    distance = float(np.sqrt(np.mean(shared_diffs)))
    similarity = float(np.clip(1.0 - distance, 0.0, 1.0))
    missingness = float(missing / total) if total else 0.0
    return similarity, missingness


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
