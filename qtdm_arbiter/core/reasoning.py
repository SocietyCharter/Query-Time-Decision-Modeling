"""
reasoning.py — Semantic tilt estimation for QTDM.

Per QTDM build doctrine (see qtdm_arbiter/QTDM_BUILD_DOCTRINE.md):
- The LLM does NOT directly predict the target value.
- The LLM produces a bounded semantic tilt z_sem ∈ [-2, 2].
- z_sem is converted to quantile p_sem = Φ(z_sem), then y_hat = Q_N(p_sem).

Flow:
  1. Distribution pass produces case-built outcome distribution.
  2. call_llm_tilt() sends query + neighbors + distribution summary to vLLM.
  3. LLM returns semantic_tilt ∈ [-2, 2] + supporting/counter reasoning.
  4. Caller converts tilt → quantile → point estimate from distribution.

Policy flags (in request.policy):
  "semantic_tilt": true    — enable this pass (default: false)
  "llm_model": "<model>"  — override model (default: qwen3.6-27b)
  "llm_timeout": 60       — request timeout in seconds (default: 60)
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

VLLM_URL = os.environ.get("QTDM_VLLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("QTDM_REASONING_MODEL", "qwen3.6-27b")
DEFAULT_TIMEOUT = int(os.environ.get("QTDM_REASONING_TIMEOUT", "60"))

# Fields shown to the LLM per neighbor (keep compact — token budget)
NEIGHBOR_FIELDS = [
    "pl_name", "pl_eqt", "pl_rade", "pl_masse", "pl_insol",
    "pl_orbsmax", "pl_orbper", "pl_dens", "st_teff", "st_spectype",
    "sy_dist", "hz_flag", "water_score",
]


def _format_neighbors(cases: List[Dict[str, Any]], limit: int = 12) -> str:
    rows = []
    for i, case in enumerate(cases[:limit]):
        payload = case.get("payload") or case.get("metadata") or case  # flat or nested
        parts = []
        for f in NEIGHBOR_FIELDS:
            v = payload.get(f)
            if v is not None and v != "" and v is not False:
                parts.append(f"{f}={v}")
        rows.append(f"  [{i+1}] " + ", ".join(parts))
    return "\n".join(rows)


def _build_tilt_prompt(
    query: str,
    target_name: str,
    target_type: str,
    neighbors: List[Dict[str, Any]],
    distribution_summary: Dict[str, Any],
    counter_distribution_summary: Optional[Dict[str, Any]] = None,
) -> str:
    # Format distribution summary
    dist_str = ""
    if distribution_summary:
        parts = []
        for k in ["q10", "q25", "median", "q75", "q90", "mean"]:
            v = distribution_summary.get(k)
            if v is not None:
                parts.append(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}")
        if parts:
            dist_str = "\nOutcome distribution of retrieved neighbors: " + ", ".join(parts)

    # Format counter distribution if available
    counter_str = ""
    if counter_distribution_summary:
        parts = []
        for k in ["q10", "q25", "median", "q75", "q90", "mean"]:
            v = counter_distribution_summary.get(k)
            if v is not None:
                parts.append(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}")
        if parts:
            counter_str = "\nCounter-example distribution: " + ", ".join(parts)

    neighbor_text = _format_neighbors(neighbors)

    unit_hint = {
        "pl_eqt": "K (Kelvin)",
        "pl_rade": "Earth radii",
        "pl_masse": "Earth masses",
        "pl_insol": "Earth flux",
        "pl_orbper": "days",
        "pl_orbsmax": "AU",
        "pl_dens": "g/cm³",
        "st_teff": "K (Kelvin)",
        "sy_dist": "parsecs",
        "water_score": "0–1 probability score",
    }.get(target_name, "")

    return f"""You are a planetary science analyst evaluating where a specific case belongs within a distribution of similar past cases.

CRITICAL RULE: You must NOT output a target value. You output ONLY a semantic tilt score.

QUERY: {query}

TARGET FIELD: `{target_name}` ({unit_hint})
{dist_str}{counter_str}

RETRIEVED NEIGHBOR CASES (similar past cases):
{neighbor_text}

Your task: Decide where this specific query belongs WITHIN the outcome distribution above.

Think about:
- Does this query describe a case that is physically LOWER, HIGHER, or TYPICAL compared to the retrieved neighbors?
- What physical factors push the value up or down (stellar type, orbital distance, planet size, insolation)?
- Are there counterexamples in the neighborhood that suggest a different position?
- What context is missing that would change your judgment?

Respond in this exact JSON format with no extra text:
{{
  "semantic_tilt": <number between -2 and 2>,
  "data_says": "<1-2 sentences: what the neighborhood directly supports>",
  "missing_context": ["<missing factor 1>", "<missing factor 2>"],
  "supporting_reason": "<why the query belongs higher/lower/typical in the distribution>",
  "counter_reason": "<counterarguments that suggest a different position>",
  "confidence_rationale": "<high|medium|low and why>"
}}

semantic_tilt guide:
- -2.0 → case is at the very bottom of the distribution (near Q5)
- -1.0 → case is clearly below median (near Q25)
-  0.0 → case is typical / near median (near Q50)
- +1.0 → case is clearly above median (near Q75)
- +2.0 → case is at the very top of the distribution (near Q95)

Do NOT output any numeric target value. Only the tilt score."""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from LLM output, tolerating leading prose."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_llm_tilt(
    query: str,
    target_name: str,
    target_type: str,
    neighbors: List[Dict[str, Any]],
    distribution_summary: Dict[str, Any],
    policy: Dict[str, Any],
    counter_distribution_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any], str]:
    """
    Call vLLM for semantic tilt estimation.

    Returns:
        (semantic_tilt, metadata_dict, model_used)
        semantic_tilt is None if LLM call failed or returned unparseable output.
        metadata_dict contains data_says, missing_context, supporting_reason, etc.
    """
    model = str(policy.get("llm_model", DEFAULT_MODEL))
    timeout = int(policy.get("llm_timeout", DEFAULT_TIMEOUT))

    prompt = _build_tilt_prompt(
        query, target_name, target_type, neighbors,
        distribution_summary, counter_distribution_summary,
    )

    try:
        resp = requests.post(
            VLLM_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return None, {"error": f"LLM call failed: {e}"}, model

    parsed = _extract_json(content)
    if not parsed:
        return None, {"error": f"LLM returned unparseable output: {content[:200]}"}, model

    # Extract and clamp semantic tilt
    raw_tilt = parsed.get("semantic_tilt")
    semantic_tilt = None
    if raw_tilt is not None:
        try:
            semantic_tilt = float(np.clip(float(raw_tilt), -2.0, 2.0))
        except (TypeError, ValueError):
            pass

    metadata = {
        "data_says": parsed.get("data_says", ""),
        "missing_context": parsed.get("missing_context", []),
        "supporting_reason": parsed.get("supporting_reason", ""),
        "counter_reason": parsed.get("counter_reason", ""),
        "confidence_rationale": parsed.get("confidence_rationale", ""),
    }

    return semantic_tilt, metadata, model


def tilt_to_prediction(
    semantic_tilt: float,
    y_values: List[float],
    weights: List[float],
) -> Tuple[float, float]:
    """
    Convert semantic tilt to point estimate using doctrine pipeline:
    p_sem = Φ(z_sem), then y_hat = Q_N(p_sem)

    Returns:
        (prediction, semantic_quantile)
    """
    from qtdm_arbiter.core.distribution import weighted_quantile

    p_sem = normal_cdf(semantic_tilt)
    # Clamp quantile to valid range
    p_sem = float(np.clip(p_sem, 0.01, 0.99))
    y_hat = weighted_quantile(y_values, weights, p_sem)
    return y_hat, p_sem
