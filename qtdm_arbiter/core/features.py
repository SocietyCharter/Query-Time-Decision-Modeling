from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from qtdm_arbiter.integration.retrieval_client import RetrievalClient
from qtdm_arbiter.models.request import ArbiterRequest
from qtdm_arbiter.domains.exoplanet import QUERY_FEATURES as EXOPLANET_QUERY_FEATURES


CANONICAL_FEATURES = [
    "confidence",
    "value",
    "relevance",
    "exposure",
    "rarity",
    "freshness",
    "body_signal_ratio",
    "trust_for_ranking",
    "completeness",
    "entity_count",
    "domain_count",
    "timestamp_count",
    "collection_status_numeric",
    "retrieval_role_numeric",
    "capability_maturity_numeric",
    # domain-specific enrichment features (populated when available)
    "page2_exists",
    "page3_exists",
    # Optional domain adapter fields. They are NaN outside matching domains.
    *EXOPLANET_QUERY_FEATURES,
]

PHYSICAL_QUERY_FEATURES = list(EXOPLANET_QUERY_FEATURES)

COLLECTION_STATUS_MAP = {
    "ignored": 0.0,
    "ignore": 0.0,
    "dropped": 0.0,
    "retained": 1.0,
    "collected": 1.0,
    "carry_now": 1.0,
    "support": 1.0,
    "excavate": 2.0,
    "queued_for_excavation": 2.0,
    "mark_for_excavation": 2.0,
}

RETRIEVAL_ROLE_MAP = {
    "supporting": 0.0,
    "support": 0.0,
    "secondary": 0.0,
    "pivot": 1.0,
    "surface_pivot": 1.0,
    "enumerator": 1.0,
    "primary": 2.0,
}

CAPABILITY_MATURITY_MAP = {
    "tentative": 0.0,
    "planned": 0.0,
    "emerging": 1.0,
    "validated": 2.0,
    "established": 2.0,
}


def extract_feature_matrix(
    findings: List[Dict[str, Any]],
    target_name: str,
    retrieval_client: RetrievalClient,
    proxy_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Return the imputed feature matrix, target array, and extraction diagnostics.
    """
    feature_names = _feature_names_for_target(target_name)
    raw_rows = np.array([_feature_row_from_payload(finding) for finding in findings], dtype=float) if findings else np.empty((0, len(CANONICAL_FEATURES)))
    # Drop the target field from X to prevent data leakage (target in features
    # causes LOO residuals to collapse to zero).
    if target_name in CANONICAL_FEATURES:
        drop_col = CANONICAL_FEATURES.index(target_name)
        raw_rows = np.delete(raw_rows, drop_col, axis=1)
    X, feature_completeness, structurally_missing = _impute_matrix(raw_rows)

    finding_ids = [str(finding.get("finding_id", "")) for finding in findings if finding.get("finding_id")]
    labels = retrieval_client.get_labels(finding_ids, target_name)

    # Build actual_y: prefer label store, then inline _label_value from payload
    # (set by pull_neighborhood_content from direct payload field lookup), then NaN.
    def _resolve_y(finding: Dict[str, Any]) -> float:
        fid = str(finding.get("finding_id", ""))
        store_row = labels.get(fid)
        if store_row:
            v = _coerce_float(store_row.get("value"))
            if not np.isnan(v):
                return v
        # Inline label set during neighborhood construction
        inline = finding.get("_label_value")
        if inline is not None:
            v = _coerce_float(inline)
            if not np.isnan(v):
                return v
        # Direct payload field (e.g. pl_eqt on exoplanet records)
        direct = finding.get(target_name)
        if direct is not None:
            v = _coerce_float(direct)
            if not np.isnan(v):
                return v
        return float("nan")

    actual_y = np.array([_resolve_y(f) for f in findings], dtype=float) if findings else np.array([], dtype=float)

    label_coverage = float(np.mean(~np.isnan(actual_y))) if len(actual_y) else 0.0
    proxy_target = label_coverage < float(proxy_threshold)
    if proxy_target:
        y = np.array([_coerce_float(finding.get("trust_for_ranking")) for finding in findings], dtype=float)
    else:
        y = actual_y

    metadata = {
        "feature_completeness": feature_completeness,
        "feature_names": feature_names,
        "target_coverage": label_coverage,
        "real_label_coverage": label_coverage,
        "proxy_target": proxy_target,
        "structurally_missing_features": [
            feature_names[index]
            for index in structurally_missing
            if index < len(feature_names)
        ],
    }
    return X, y, metadata


def extract_query_vector(
    request: ArbiterRequest,
    X_neighborhood: np.ndarray,
    feature_names: List[str] | None = None,
) -> np.ndarray:
    """Build x_q from request features and impute missing values from neighborhood medians."""
    active_feature_names = list(feature_names or _feature_names_for_width(request.target_name, X_neighborhood))
    raw_row = np.array(_feature_row_from_payload(request.features), dtype=float)
    full_feature_map = {
        feature: raw_row[index]
        for index, feature in enumerate(CANONICAL_FEATURES)
    }
    row = np.array([full_feature_map.get(feature, float("nan")) for feature in active_feature_names], dtype=float)
    if X_neighborhood.size:
        neighborhood_medians, _ = _column_stats(X_neighborhood)
        medians = np.zeros(len(active_feature_names), dtype=float)
        width = min(len(neighborhood_medians), len(medians))
        medians[:width] = neighborhood_medians[:width]
    else:
        medians = np.zeros(len(active_feature_names), dtype=float)
    request_features = request.features or {}
    for index, feature in enumerate(active_feature_names):
        if not np.isnan(row[index]):
            continue
        if feature in request_features and request_features[feature] is not None:
            row[index] = _coerce_float(request_features[feature], default=medians[index])
        else:
            row[index] = medians[index]
    return row.astype(float)


def _feature_row_from_payload(payload: Dict[str, Any]) -> List[float]:
    scores = payload.get("scores", {}) or {}
    return [
        _coerce_float(payload.get("confidence")),
        _coerce_float(_nested_value(payload, scores, "value")),
        _coerce_float(_nested_value(payload, scores, "relevance")),
        _coerce_float(_nested_value(payload, scores, "exposure")),
        _coerce_float(_nested_value(payload, scores, "rarity")),
        _coerce_float(_nested_value(payload, scores, "freshness")),
        _coerce_float(_nested_value(payload, scores, "body_signal_ratio")),
        _coerce_float(payload.get("trust_for_ranking")),
        _coerce_float(payload.get("completeness")),
        _coerce_float(payload.get("entity_count"), default=float(len(payload.get("entities", []) or []))),
        _coerce_float(payload.get("domain_count"), default=float(len(payload.get("related_domains", []) or []))),
        _coerce_float(payload.get("timestamp_count"), default=float(len(payload.get("timestamps", []) or []))),
        _encode_value(payload.get("collection_status_numeric"), payload.get("collection_status"), COLLECTION_STATUS_MAP),
        _encode_value(payload.get("retrieval_role_numeric"), payload.get("retrieval_role"), RETRIEVAL_ROLE_MAP),
        _encode_value(payload.get("capability_maturity_numeric"), payload.get("capability_maturity"), CAPABILITY_MATURITY_MAP),
        _coerce_float(payload.get("page2_exists")),
        _coerce_float(payload.get("page3_exists")),
        *[_coerce_float(payload.get(feature)) for feature in EXOPLANET_QUERY_FEATURES],
    ]


def _nested_value(payload: Dict[str, Any], scores: Dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    return scores.get(key)


def _encode_value(numeric_value: Any, raw_value: Any, mapping: Dict[str, float]) -> float:
    numeric = _coerce_float(numeric_value)
    if not np.isnan(numeric):
        return float(numeric)
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return float("nan")
    return float(mapping.get(raw, 1.0))


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _label_value(row: Dict[str, Any] | None) -> float:
    if not row:
        return float("nan")
    return _coerce_float(row.get("value"))


def _impute_matrix(X: np.ndarray) -> Tuple[np.ndarray, float, List[int]]:
    if X.size == 0:
        return X.astype(float), 0.0, []
    completeness = float(np.mean(~np.isnan(X)))
    col_medians, missing_fracs = _column_stats(X)
    structurally_missing = [index for index, fraction in enumerate(missing_fracs) if fraction > 0.8]
    missing = np.where(np.isnan(X))
    X = X.copy()
    X[missing] = np.take(col_medians, missing[1])
    return X.astype(float), completeness, structurally_missing


def _column_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty
    medians = []
    missing_fracs = []
    for index in range(X.shape[1]):
        column = X[:, index]
        valid = column[~np.isnan(column)]
        medians.append(float(np.median(valid)) if len(valid) else 0.0)
        missing_fracs.append(float(np.mean(np.isnan(column))))
    return np.asarray(medians, dtype=float), np.asarray(missing_fracs, dtype=float)


def _feature_names_for_target(target_name: str) -> List[str]:
    if target_name in CANONICAL_FEATURES:
        return [feature for feature in CANONICAL_FEATURES if feature != target_name]
    return list(CANONICAL_FEATURES)


def _feature_names_for_width(target_name: str, X_neighborhood: np.ndarray) -> List[str]:
    if X_neighborhood.ndim == 2 and X_neighborhood.shape[1] == len(CANONICAL_FEATURES) - 1 and target_name in CANONICAL_FEATURES:
        return _feature_names_for_target(target_name)
    return list(CANONICAL_FEATURES)
