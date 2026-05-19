#!/usr/bin/env python3
# ⚠️ QTDM BUILD DOCTRINE: see qtdm_arbiter/QTDM_BUILD_DOCTRINE.md
# Point estimate = Q_N(Phi(z_sem)). LLM produces tilt only, never raw target value.
# Confidence is computed AFTER prediction, not before.
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdm_arbiter.audit.log import write_audit
from qtdm_arbiter.core.confidence import compute_support, compute_weights
from qtdm_arbiter.core.conformal import weighted_conformal_interval
from qtdm_arbiter.core.estimator import estimate_semantic_distribution, fit_and_predict, select_model
from qtdm_arbiter.core.features import CANONICAL_FEATURES, extract_feature_matrix, extract_query_vector
from qtdm_arbiter.core.neighborhood import gate_neighborhood, pull_neighborhood, pull_neighborhood_content, validate_neighborhood
from qtdm_arbiter.core.refusal import check_refusal_gates, is_label_gate
from qtdm_arbiter.integration.retrieval_client import RetrievalClient
from qtdm_arbiter.models.request import ArbiterRequest
from qtdm_arbiter.models.response import ArbiterResponse


def _load_env_file() -> None:
    env_path = PACKAGE_ROOT / "arbiter-service.env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

RETRIEVAL_URL = os.environ.get("QTDM_RETRIEVAL_URL", "http://localhost:8000")
DEFAULT_K = int(os.environ.get("QTDM_DEFAULT_K", 25))
MIN_K = int(os.environ.get("QTDM_MIN_USABLE_K", 5))
MIN_SUPPORT = float(os.environ.get("QTDM_MIN_SUPPORT_SCORE", 0.3))
RIDGE_ALPHA = float(os.environ.get("QTDM_RIDGE_ALPHA", 1.0))
LOGISTIC_C = float(os.environ.get("QTDM_LOGISTIC_C", 1.0))
SOFTMAX_LAMBDA_OVERRIDE = os.environ.get("QTDM_SOFTMAX_LAMBDA")


def run_request(request: ArbiterRequest, retrieval_url: str | None = None) -> ArbiterResponse:
    retrieval_base = retrieval_url or RETRIEVAL_URL
    retrieval_client = RetrievalClient(retrieval_base)
    k = int(request.policy.get("k", DEFAULT_K))
    min_support = float(request.policy.get("min_support", MIN_SUPPORT))
    explicit_lambda = request.policy.get("lambda")
    softmax_lambda = (
        float(explicit_lambda)
        if explicit_lambda is not None
        else (float(SOFTMAX_LAMBDA_OVERRIDE) if SOFTMAX_LAMBDA_OVERRIDE is not None else None)
    )
    weighting_strategy = str(request.policy.get("weighting_strategy", "adaptive_k_eff"))
    ridge_alpha = float(request.policy.get("alpha", RIDGE_ALPHA))
    logistic_c = float(request.policy.get("logistic_c", LOGISTIC_C))

    is_content_mode = (request.entity_type or "").lower() == "content"

    if is_content_mode:
        raw = pull_neighborhood_content(request, retrieval_base, k, target_name=request.target_name, client=retrieval_client)
        # Content neighbors are pre-gated (all have required fields); use a low confidence floor
        gated = gate_neighborhood(raw, min_confidence=0.0)
        usable, refusal_reason = validate_neighborhood(gated, MIN_K)
        usable = usable[:k]
        X, y, feature_meta = extract_feature_matrix(usable, request.target_name, retrieval_client)
    else:
        raw = pull_neighborhood(request, retrieval_base, k, client=retrieval_client)
        gated = gate_neighborhood(raw, min_confidence=float(request.filters.get("min_confidence", 0.5)))
        usable, refusal_reason = validate_neighborhood(gated, MIN_K)
        usable = usable[:k]
        X, y, feature_meta = extract_feature_matrix(usable, request.target_name, retrieval_client)
    scores = [float(finding.get("_result_score") or finding.get("_semantic_score") or 0.0) for finding in usable]
    weights = compute_weights(scores, strategy=weighting_strategy, lam=softmax_lambda)
    x_q = extract_query_vector(request, X)

    y_hat_knn = _weighted_knn(y, weights)
    support_score, diagnostics = compute_support(
        len(usable),
        scores,
        X,
        y,
        y_hat_knn,
        y_hat_knn,
        feature_completeness=feature_meta["feature_completeness"],
        target_coverage=feature_meta["target_coverage"],
        proxy_target=feature_meta["proxy_target"],
    )
    diagnostics["real_label_coverage"] = round(float(feature_meta["real_label_coverage"]), 4)
    diagnostics["structurally_missing_features"] = list(feature_meta.get("structurally_missing_features", []))

    fuzzy_mode = bool(request.policy.get("fuzzy", False))
    semantic_distribution_mode = str(request.policy.get("mode", "")).lower() == "semantic_distribution"

    if semantic_distribution_mode:
        labeled_values = []
        labeled_weights = []
        for index, finding in enumerate(usable):
            value = finding.get("_label_value")
            if value is None:
                value = finding.get(request.target_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(numeric_value):
                continue
            labeled_values.append(numeric_value)
            labeled_weights.append(float(weights[index]) if index < len(weights) else 1.0)
        evidence_case_ids = [
            str(finding.get("finding_id") or finding.get("chunk_id") or "")
            for finding in usable[:10]
            if (finding.get("finding_id") or finding.get("chunk_id"))
        ]
        if not labeled_values:
            status = "refused" if refusal_reason else "semantic_support_only"
            response = ArbiterResponse(
                request_id=request.request_id,
                status=status,
                prediction=None,
                prediction_type=None,
                confidence=round(float(support_score), 4),
                support_score=round(float(support_score), 4),
                model_used="semantic_distribution",
                neighbors_requested=k,
                neighbors_used=len(usable),
                evidence_case_ids=evidence_case_ids,
                supporting_case_ids=evidence_case_ids,
                fallback_used=False,
                refusal_reason=refusal_reason or "no_real_labels",
                distribution_summary={},
                comparator_summary={},
                confidence_components={"final_confidence": round(float(support_score), 4)},
                diagnostics=diagnostics,
                explanation=None if status == "refused" else build_explanation(None, usable, diagnostics),
            )
            write_audit(request, response)
            return response

        semantic_refusal = refusal_reason or check_refusal_gates(
            len(usable),
            diagnostics["feature_completeness"],
            diagnostics["target_coverage"],
            diagnostics["sim_coherence"],
            support_score,
            {
                "min_neighbors": MIN_K,
                "min_support": min_support,
                "min_real_label_coverage": float(request.policy.get("min_real_label_coverage", 0.1)),
            },
            real_label_coverage=diagnostics["real_label_coverage"],
            fuzzy_mode=False,
        )
        if semantic_refusal is not None:
            response = ArbiterResponse(
                request_id=request.request_id,
                status="refused",
                prediction=None,
                prediction_type=None,
                confidence=round(float(support_score), 4),
                support_score=round(float(support_score), 4),
                model_used="semantic_distribution",
                neighbors_requested=k,
                neighbors_used=len(usable),
                evidence_case_ids=evidence_case_ids,
                supporting_case_ids=evidence_case_ids,
                fallback_used=False,
                refusal_reason=semantic_refusal,
                distribution_summary={},
                comparator_summary={},
                confidence_components={"final_confidence": round(float(support_score), 4)},
                diagnostics=diagnostics,
                explanation=None,
            )
            write_audit(request, response)
            return response

        dist_result = estimate_semantic_distribution(
            request.target_type,
            np.asarray(labeled_values, dtype=float),
            np.asarray(labeled_weights, dtype=float),
            point_method=str(request.policy.get("point_method", "weighted_median")),
        )
        prediction = dist_result["prediction"]
        model_used = "semantic_distribution"
        tilt_meta: Dict[str, Any] = {}
        tilt_model = None
        semantic_tilt = None
        semantic_quantile = None
        if request.target_type == "regression" and bool(request.policy.get("semantic_tilt", False)):
            from qtdm_arbiter.core.reasoning import call_llm_tilt, tilt_to_prediction

            raw_mock_tilt = request.policy.get("mock_semantic_tilt", request.policy.get("semantic_tilt_value"))
            if raw_mock_tilt is not None:
                semantic_tilt = float(np.clip(float(raw_mock_tilt), -2.0, 2.0))
                tilt_meta = {"source": "policy_mock_tilt"}
                tilt_model = "mock_semantic_tilt"
            else:
                semantic_tilt, tilt_meta, tilt_model = call_llm_tilt(
                    query=request.query_summary,
                    target_name=request.target_name,
                    target_type=request.target_type,
                    neighbors=usable[:12],
                    distribution_summary=dist_result.get("distribution_summary", {}),
                    policy=request.policy,
                    counter_distribution_summary=None,
                )
            if semantic_tilt is not None:
                prediction, semantic_quantile = tilt_to_prediction(
                    float(semantic_tilt),
                    [float(value) for value in labeled_values],
                    [float(weight) for weight in labeled_weights],
                )
                dist_result["prediction"] = prediction
                dist_result["prediction_median"] = prediction
                model_used = "semantic_distribution+semantic_tilt"
                diagnostics["tilt_applied"] = True
                diagnostics["semantic_tilt"] = round(float(semantic_tilt), 4)
                diagnostics["semantic_quantile"] = round(float(semantic_quantile), 4)
                diagnostics["tilt_model"] = tilt_model
                diagnostics["tilt_path"] = "p=Phi(z_sem); prediction=weighted_quantile(p)"
                for key in ("data_says", "missing_context", "supporting_reason", "counter_reason", "confidence_rationale", "source"):
                    if isinstance(tilt_meta, dict) and key in tilt_meta:
                        diagnostics[key] = tilt_meta[key]
            elif tilt_meta:
                diagnostics["tilt_error"] = tilt_meta.get("error", "semantic_tilt_unavailable")

        conformal_interval = None
        if request.target_type == "regression":
            label_array = np.asarray(labeled_values, dtype=float)
            weight_array = np.asarray(labeled_weights, dtype=float)
            y_hat_neighbors = np.full(len(label_array), float(prediction), dtype=float)
            conformal_interval = weighted_conformal_interval(
                label_array,
                y_hat_neighbors,
                weight_array,
                float(prediction),
                alpha=float(request.policy.get("conformal_alpha", 0.20)),
            )
        comparator_summary = {
            "overlap_with_base": None,
            "counter_query_count": len(request.policy.get("counter_queries", []) or []),
        }
        confidence_components = {
            "final_confidence": round(float(support_score), 4),
            "label_coverage": diagnostics["real_label_coverage"],
            "neighbors_used": len(usable),
        }
        if semantic_tilt is not None:
            confidence_components["semantic_tilt"] = round(float(semantic_tilt), 4)
            confidence_components["semantic_quantile"] = round(float(semantic_quantile), 4)
        response = ArbiterResponse(
            request_id=request.request_id,
            status="ok",
            prediction=_normalize_prediction(prediction, request.target_type),
            prediction_mean=dist_result.get("prediction_mean"),
            prediction_median=dist_result.get("prediction_median"),
            prediction_mode=dist_result.get("prediction_mode"),
            prediction_low=conformal_interval[0] if conformal_interval else dist_result.get("prediction_low"),
            prediction_high=conformal_interval[1] if conformal_interval else dist_result.get("prediction_high"),
            prediction_type=dist_result.get("prediction_type"),
            confidence=round(float(support_score), 4),
            support_score=round(float(support_score), 4),
            model_used=model_used,
            neighbors_requested=k,
            neighbors_used=len(usable),
            evidence_case_ids=evidence_case_ids,
            supporting_case_ids=evidence_case_ids,
            fallback_used=False,
            refusal_reason=None,
            distribution_summary=dist_result.get("distribution_summary", {}),
            comparator_summary=comparator_summary,
            confidence_components=confidence_components,
            diagnostics=diagnostics,
            explanation=build_explanation(None, usable, diagnostics),
        )
        write_audit(request, response)
        return response

    if not refusal_reason:
        refusal_reason = check_refusal_gates(
            len(usable),
            diagnostics["feature_completeness"],
            diagnostics["target_coverage"],
            diagnostics["sim_coherence"],
            support_score,
            {"min_neighbors": MIN_K, "min_support": min_support},
            real_label_coverage=diagnostics["real_label_coverage"],
            fuzzy_mode=fuzzy_mode,
        )

    # Auto-activate fuzzy mode when only label-coverage gates are blocking us.
    # The semantic neighbourhood is valid; we just don't have ground-truth
    # labels.  Produce a similarity-weighted estimate with reduced confidence
    # rather than refusing outright.
    if refusal_reason is not None and is_label_gate(refusal_reason):
        fuzzy_mode = True
        refusal_reason = check_refusal_gates(
            len(usable),
            diagnostics["feature_completeness"],
            diagnostics["target_coverage"],
            diagnostics["sim_coherence"],
            support_score,
            {"min_neighbors": MIN_K, "min_support": min_support},
            real_label_coverage=diagnostics["real_label_coverage"],
            fuzzy_mode=True,
        )
        diagnostics["fuzzy_mode"] = True

    prediction = None
    fitted = None
    model_used = None
    fallback_used = False

    # Fuzzy estimate: use the weighted mean of retrieval scores (0-1) as a
    # similarity-strength proxy when no ground-truth labels are available.
    # Confidence is discounted to reflect that this is semantic proximity,
    # not a trained prediction.
    if refusal_reason is None and diagnostics.get("fuzzy_mode"):
        fuzzy_scores = np.asarray(scores, dtype=float)
        if len(fuzzy_scores) > 0:
            prediction = float(np.average(fuzzy_scores, weights=weights))
        else:
            prediction = 0.0
        model_used = "fuzzy_similarity"
        fallback_used = True
        # Discount confidence: neighbourhood support exists but no labels.
        support_score = round(support_score * 0.55, 4)
        diagnostics["fuzzy_confidence_discount"] = 0.55

    if refusal_reason is None and not diagnostics.get("fuzzy_mode"):
        # Allow policy to force a specific estimator
        forced_estimator = str(request.policy.get("estimator", ""))
        if forced_estimator in ("knn", "ridge", "isotonic", "logistic"):
            model_name = forced_estimator
        else:
            model_name = select_model(
                request.target_type,
                len(usable),
                diagnostics["feature_completeness"],
                proxy_target=feature_meta["proxy_target"],
            )
        try:
            if model_name == "knn":
                prediction = y_hat_knn
                model_used = "knn"
            else:
                prediction, fitted = fit_and_predict(
                    model_name,
                    X,
                    y,
                    weights,
                    x_q,
                    alpha=ridge_alpha,
                    logistic_c=logistic_c,
                )
                model_used = model_name
                support_score, diagnostics = compute_support(
                    len(usable),
                    scores,
                    X,
                    y,
                    prediction,
                    y_hat_knn,
                    feature_completeness=feature_meta["feature_completeness"],
                    target_coverage=feature_meta["target_coverage"],
                    proxy_target=feature_meta["proxy_target"],
                )
                diagnostics["real_label_coverage"] = round(float(feature_meta["real_label_coverage"]), 4)
                diagnostics["structurally_missing_features"] = list(feature_meta.get("structurally_missing_features", []))
                refusal_reason = check_refusal_gates(
                    len(usable),
                    diagnostics["feature_completeness"],
                    diagnostics["target_coverage"],
                    diagnostics["sim_coherence"],
                    support_score,
                    {"min_neighbors": MIN_K, "min_support": min_support},
                    real_label_coverage=diagnostics["real_label_coverage"],
                )
        except ValueError:
            prediction = y_hat_knn
            model_used = "knn" if y_hat_knn is not None else None
            fallback_used = y_hat_knn is not None
            diagnostics["fit_unstable"] = True

    if refusal_reason is not None and request.target_type == "ranking" and usable:
        fallback_used = True
        model_used = "ranking_only"
        refusal_reason = None
        diagnostics["ranking_only_fallback"] = True
    elif refusal_reason is not None and refusal_reason not in ("no_real_labels", "sparse_labels") and y_hat_knn is not None and support_score >= (min_support * 0.5):
        prediction = y_hat_knn
        model_used = "knn"
        fallback_used = True
        refusal_reason = None

    conformal_interval = None
    if (
        refusal_reason is None
        and prediction is not None
        and request.target_type == "regression"
        and not diagnostics.get("fuzzy_mode")
    ):
        if fitted is not None and hasattr(fitted, "predict") and model_used not in ("knn", "fuzzy_similarity"):
            try:
                y_hat_neighbors = np.asarray(fitted.predict(X), dtype=float)
            except Exception:
                y_hat_neighbors = np.full(len(y), y_hat_knn if y_hat_knn is not None else float("nan"), dtype=float)
        else:
            y_hat_neighbors = np.full(len(y), y_hat_knn if y_hat_knn is not None else float("nan"), dtype=float)
        conformal_interval = weighted_conformal_interval(
            y,
            y_hat_neighbors,
            weights,
            float(prediction),
            alpha=float(request.policy.get("conformal_alpha", 0.20)),
        )

    # ── Semantic tilt pass (LLM-in-the-loop) ──
    # Per doctrine: LLM produces z_sem ∈ [-2,2], converted to quantile, applied to distribution.
    tilt_applied = False
    tilt_metadata = None
    if (
        refusal_reason is None
        and prediction is not None
        and request.target_type == "regression"
        and not diagnostics.get("fuzzy_mode")
        and request.policy.get("semantic_tilt", False)
        and len(y) > 0
    ):
        try:
            from qtdm_arbiter.core.reasoning import (
                call_llm_tilt,
                tilt_to_prediction,
                normal_cdf,
            )
            from qtdm_arbiter.core.distribution import weighted_quantile

            # Build distribution summary
            y_clean = y[~np.isnan(y)]
            w_clean = weights[:len(y)]
            dist_summary = {}
            for q_name, q_val in [("q10", 0.10), ("q25", 0.25), ("median", 0.50), ("q75", 0.75), ("q90", 0.90)]:
                if len(y_clean) > 0:
                    dist_summary[q_name] = float(weighted_quantile(y_clean, w_clean[:len(y_clean)], q_val))
            if len(y_clean) > 0:
                dist_summary["mean"] = float(np.average(y_clean, weights=w_clean[:len(y_clean)]))

            # Get counter distribution if available
            counter_dist_summary = None
            if diagnostics.get("counter_distribution"):
                counter_y = np.array(diagnostics["counter_distribution"].get("y", []))
                counter_w = np.array(diagnostics["counter_distribution"].get("weights", []), dtype=float)
                if len(counter_y) > 0:
                    counter_w = counter_w / max(counter_w.sum(), 1e-9)
                    counter_dist_summary = {}
                    for q_name, q_val in [("q10", 0.10), ("q25", 0.25), ("median", 0.50), ("q75", 0.75), ("q90", 0.90)]:
                        counter_dist_summary[q_name] = float(weighted_quantile(counter_y, counter_w, q_val))
                    counter_dist_summary["mean"] = float(np.average(counter_y, weights=counter_w))

            # Call LLM for semantic tilt
            semantic_tilt, tilt_meta, tilt_model = call_llm_tilt(
                query=request.query_summary,
                target_name=request.target_name,
                target_type=request.target_type,
                neighbors=usable[:12],
                distribution_summary=dist_summary,
                policy=request.policy,
                counter_distribution_summary=counter_dist_summary,
            )

            if semantic_tilt is not None:
                # Convert tilt → quantile → prediction from distribution
                y_valid_list = list(y_clean)
                w_valid_list = list(w_clean[:len(y_clean)])
                tilted_pred, p_sem = tilt_to_prediction(semantic_tilt, y_valid_list, w_valid_list)

                # Only apply tilt if it's different enough from base prediction
                if not np.isnan(tilted_pred) and abs(tilted_pred - prediction) > 0.01 * abs(prediction):
                    old_prediction = prediction
                    prediction = tilted_pred
                    tilt_applied = True
                    tilt_metadata = tilt_meta
                    diagnostics["tilt_applied"] = True
                    diagnostics["semantic_tilt"] = round(semantic_tilt, 4)
                    diagnostics["semantic_quantile"] = round(p_sem, 4)
                    diagnostics["tilt_model"] = tilt_model
                    diagnostics["tilt_old_prediction"] = round(old_prediction, 4)
                    diagnostics["data_says"] = tilt_meta.get("data_says", "")
                    diagnostics["missing_context"] = tilt_meta.get("missing_context", [])
                    diagnostics["supporting_reason"] = tilt_meta.get("supporting_reason", "")
                    diagnostics["counter_reason"] = tilt_meta.get("counter_reason", "")
                    model_used = f"{model_used}+tilt"

                    # Recompute conformal interval with tilted prediction
                    if fitted is not None and hasattr(fitted, "predict"):
                        try:
                            y_hat_neighbors_tilt = np.asarray(fitted.predict(X), dtype=float)
                        except Exception:
                            y_hat_neighbors_tilt = np.full(len(y), tilted_pred, dtype=float)
                    else:
                        y_hat_neighbors_tilt = np.full(len(y), tilted_pred, dtype=float)
                    conformal_interval = weighted_conformal_interval(
                        y, y_hat_neighbors_tilt, weights, float(prediction),
                        alpha=float(request.policy.get("conformal_alpha", 0.20)),
                    )
        except Exception as e:
            diagnostics["tilt_error"] = str(e)

    # Determine status.  Fuzzy estimates get their own status so callers can
    # distinguish them from trained predictions or hard refusals.
    if diagnostics.get("fuzzy_mode") and prediction is not None:
        status = "fuzzy_estimate"
    elif prediction is None and not diagnostics.get("ranking_only_fallback"):
        status = "refused"
    elif fallback_used:
        status = "fallback"
    else:
        status = "ok"
    response = ArbiterResponse(
        request_id=request.request_id,
        status=status,
        prediction=None if diagnostics.get("ranking_only_fallback") else _normalize_prediction(prediction, request.target_type),
        prediction_low=conformal_interval[0] if conformal_interval else None,
        prediction_high=conformal_interval[1] if conformal_interval else None,
        prediction_type=_prediction_type(request.target_type, diagnostics),
        confidence=support_score,
        support_score=support_score,
        model_used=model_used,
        neighbors_requested=k,
        neighbors_used=len(usable),
        evidence_case_ids=[str(finding.get("finding_id") or finding.get("chunk_id") or "") for finding in usable[:10] if (finding.get("finding_id") or finding.get("chunk_id"))],
        fallback_used=fallback_used,
        refusal_reason=refusal_reason,
        diagnostics=diagnostics,
        explanation=None if status == "refused" else build_explanation(fitted, usable, diagnostics),
    )
    write_audit(request, response)
    return response


def build_explanation(fitted_model: Any, usable: list[Dict[str, Any]], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    if fitted_model is not None and hasattr(fitted_model, "coef_"):
        coefficients = np.ravel(np.asarray(fitted_model.coef_, dtype=float))
        order = np.argsort(np.abs(coefficients))[::-1][:3]
        top_features = [CANONICAL_FEATURES[index] for index in order]
    else:
        # Derive top features from what actually varies across the neighborhood.
        # For content/density mode, route-level fields are the real signal.
        is_content = usable and usable[0].get("entity_type") == "content"
        if is_content:
            # Summarize which route fields drove neighbor selection
            origins = {u.get("origin", "") for u in usable if u.get("origin")}
            dests = {u.get("dest", "") for u in usable if u.get("dest")}
            carriers = {u.get("carrier", "") for u in usable if u.get("carrier")}
            top_features = []
            if origins:
                top_features.append(f"origin:{','.join(sorted(origins)[:3])}")
            if dests:
                top_features.append(f"dest:{','.join(sorted(dests)[:3])}")
            if carriers:
                top_features.append(f"carriers:{','.join(sorted(carriers)[:4])}")
            if not top_features:
                top_features = ["semantic_score"]
        else:
            top_features = ["relevance", "trust_for_ranking", "body_signal_ratio"]

    closest_cases = [
        {"id": str(item.get("finding_id") or item.get("chunk_id") or ""), "score": round(float(item.get("_result_score") or item.get("_semantic_score") or 0.0), 4)}
        for item in usable[:3]
        if (item.get("finding_id") or item.get("chunk_id"))
    ]
    support_rationale = (
        f"{diagnostics.get('n_usable', 0)} usable neighbors, tight similarity "
        f"(std {diagnostics.get('sim_coherence', 0.0):.2f}), "
        f"{diagnostics.get('feature_completeness', 0.0) * 100:.0f}% feature completeness"
    )
    return {
        "top_features": top_features,
        "closest_cases": closest_cases,
        "support_rationale": support_rationale,
    }


def _prediction_type(target_type: str, diagnostics: Dict[str, Any]) -> str:
    if diagnostics.get("ranking_only_fallback"):
        return "ranking"
    if diagnostics.get("fuzzy_mode"):
        return "similarity_score"
    if target_type == "binary_classification":
        return "probability"
    if target_type == "ranking":
        return "score"
    return "value"


def _weighted_knn(y: np.ndarray, weights: np.ndarray) -> float | None:
    if len(y) == 0 or len(weights) == 0:
        return None
    mask = ~np.isnan(y)
    if not np.any(mask):
        return None
    y_valid = y[mask]
    weights_valid = weights[mask]
    if np.sum(weights_valid) <= 0.0:
        return float(np.mean(y_valid))
    return float(np.average(y_valid, weights=weights_valid))


def _normalize_prediction(prediction: float | None, target_type: str) -> float | None:
    if prediction is None:
        return None
    value = float(prediction)
    if target_type == "binary_classification":
        return max(0.0, min(1.0, value))
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-callable CLI for QTDM Arbiter.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--target-type", required=True, choices=["regression", "binary_classification", "ranking"])
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--entity-type")
    parser.add_argument("--filters")
    parser.add_argument("--features")
    parser.add_argument("--policy")
    parser.add_argument(
        "--data-types",
        nargs="*",
        default=[],
        help="Filter content neighborhood to these data_type values (e.g. exoplanet_record)",
    )
    parser.add_argument("--request-id")
    parser.add_argument("--out")
    parser.add_argument("--fuzzy", action="store_true",
                        help="Force fuzzy mode: skip label-coverage gates and return "
                             "similarity-weighted estimate with discounted confidence.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    policy = json.loads(args.policy or "{}")
    if getattr(args, "fuzzy", False):
        policy["fuzzy"] = True
    request = ArbiterRequest(
        request_id=args.request_id or f"arb-{uuid.uuid4().hex[:8]}",
        target_type=args.target_type,
        target_name=args.target_name,
        entity_type=args.entity_type,
        query_summary=args.query,
        filters=json.loads(args.filters or "{}"),
        features=json.loads(args.features or "{}"),
        policy=policy,
        data_types=args.data_types or [],
    )
    response = run_request(request)
    payload = json.loads(response.model_dump_json())
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
