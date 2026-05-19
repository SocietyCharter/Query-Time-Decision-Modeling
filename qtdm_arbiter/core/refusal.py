from __future__ import annotations

from typing import Any, Dict


_LABEL_GATES = frozenset({"sparse_labels", "no_real_labels"})


def check_refusal_gates(
    n_usable: int,
    feature_completeness: float,
    target_coverage: float,
    sim_coherence: float,
    support_score: float,
    config: dict,
    real_label_coverage: float = 1.0,
    fuzzy_mode: bool = False,
) -> str | None:
    min_neighbors = int(config.get("min_neighbors", 5))
    min_support = float(config.get("min_support", 0.3))
    min_real_label_coverage = float(config.get("min_real_label_coverage", 0.1))
    if n_usable < min_neighbors:
        return "insufficient_neighbors"
    if feature_completeness < 0.4:
        return "insufficient_features"
    if not fuzzy_mode:
        if target_coverage < 0.4:
            return "sparse_labels"
        if real_label_coverage < min_real_label_coverage:
            return "no_real_labels"
    if sim_coherence > 0.4:
        return "diffuse_neighborhood"
    if support_score < min_support:
        return "low_support"
    return None


def evaluate_distribution_gates(
    *,
    neighbors_used: int,
    neighbors_requested: int,
    label_coverage: float,
    k_eff: float,
    max_weight: float,
    flat_semantic_neighborhood: bool,
    overlap_with_counter: float | None,
    overlap_with_base: float | None,
    width_ratio: float | None,
    class_margin: float | None,
    distribution_multimodal: bool,
    min_k: int = 5,
    min_label_coverage: float = 0.25,
    allow_single_precedent: bool = False,
) -> str | None:
    if neighbors_used < int(min_k):
        return "insufficient_neighbors"
    if label_coverage < float(min_label_coverage):
        return "no_real_labels"
    if flat_semantic_neighborhood and neighbors_requested > 0:
        return "needs_escalation_flat_semantic_neighborhood"
    if max_weight > 0.50 and not allow_single_precedent:
        return "dominant_precedent"
    if k_eff < 4.0 and not allow_single_precedent:
        return "low_effective_sample_size"
    if overlap_with_counter is not None and overlap_with_counter > 0.70:
        return "high_counter_overlap"
    if width_ratio is not None and width_ratio > 0.80:
        return "distribution_too_wide"
    if overlap_with_base is not None and overlap_with_base > 0.85:
        return "low_comparator_separation"
    if class_margin is not None and class_margin < 0.15:
        return "contradictory_evidence"
    if distribution_multimodal:
        return "needs_escalation_multimodal"
    return None


def is_label_gate(reason: str | None) -> bool:
    return reason in _LABEL_GATES
