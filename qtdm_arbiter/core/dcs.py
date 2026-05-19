"""
dcs.py — Decision Context Synthesizer for QTDM.

Stage 1 of the LLM reasoning pipeline. Runs after the distribution pass.
Converts partial semantic evidence into a structured decision-grade analysis
without inventing facts.

Output schema:
{
  "data_says": str,                    # what the retrieved neighborhood directly supports
  "missing_context": [str],            # important context not present in the data
  "latent_factors": [str],             # implied factors the data hints at but doesn't state
  "decision_claim": str,               # the responsible claim that can be made
  "supporting_distribution": str,      # qualitative description of the supporting case cluster
  "counter_distribution": str,         # qualitative description of the counter case cluster
  "confidence_reason": str,            # why confidence is high/medium/low
  "limits": [str]                      # hard limits on what can be claimed
}

Policy flags (in request.policy):
  "dcs": true            — enable this stage (default: false)
  "llm_model": str       — model to use (shared with reasoning.py)
  "llm_timeout": int     — timeout in seconds (default: 60)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

VLLM_URL = os.environ.get("QTDM_VLLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("QTDM_REASONING_MODEL", "qwen3.6-27b")
DEFAULT_TIMEOUT = int(os.environ.get("QTDM_REASONING_TIMEOUT", "60"))

NEIGHBOR_FIELDS = [
    "pl_name", "pl_eqt", "pl_rade", "pl_masse", "pl_insol",
    "pl_orbsmax", "pl_orbper", "pl_dens", "st_teff", "st_spectype",
    "sy_dist", "hz_flag", "water_score",
]

FIELD_UNITS = {
    "pl_eqt": "K", "pl_rade": "Re", "pl_masse": "Me", "pl_insol": "S⊕",
    "pl_orbsmax": "AU", "pl_orbper": "d", "pl_dens": "g/cm³",
    "st_teff": "K", "sy_dist": "pc", "water_score": "0-1",
}


def _format_neighbors(cases: List[Dict[str, Any]], limit: int = 12) -> str:
    rows = []
    for i, case in enumerate(cases[:limit]):
        payload = case.get("payload", case)
        parts = []
        for f in NEIGHBOR_FIELDS:
            v = payload.get(f)
            if v is not None and v != "" and v is not False:
                unit = FIELD_UNITS.get(f, "")
                parts.append(f"{f}={v}{unit}")
        if parts:
            rows.append(f"  [{i+1}] " + ", ".join(parts))
    return "\n".join(rows) if rows else "  (no neighbor data)"


def _format_distribution(summary: Dict[str, Any], label: str) -> str:
    if not summary:
        return f"{label}: (unavailable)"
    keys = ["mean", "median", "std", "p10", "p90", "min", "max"]
    bits = [f"{k}={summary[k]:.1f}" for k in keys if summary.get(k) is not None]
    return f"{label}: " + ", ".join(bits)


def _build_dcs_prompt(
    query: str,
    target_name: str,
    target_type: str,
    neighbors: List[Dict[str, Any]],
    first_pass: Optional[float],
    distribution_summary: Dict[str, Any],
    counter_distribution_summary: Dict[str, Any],
) -> str:
    unit_hint = {
        "pl_eqt": "K (equilibrium temperature)",
        "pl_rade": "Earth radii (planetary radius)",
        "pl_masse": "Earth masses (planetary mass)",
        "pl_insol": "Earth flux (stellar insolation)",
        "pl_orbper": "days (orbital period)",
        "pl_orbsmax": "AU (semi-major axis)",
        "pl_dens": "g/cm³ (bulk density)",
        "st_teff": "K (stellar effective temperature)",
        "sy_dist": "parsecs (system distance)",
        "water_score": "0–1 (habitability-weighted score)",
    }.get(target_name, "")

    neighbor_text = _format_neighbors(neighbors)
    dist_text = _format_distribution(distribution_summary, "Supporting neighborhood distribution")
    counter_text = _format_distribution(counter_distribution_summary, "Counter neighborhood distribution")

    return f"""You are a planetary science analyst performing a structured evidence synthesis.

QUERY: {query}

TARGET FIELD: `{target_name}` ({unit_hint})
FIRST-PASS ESTIMATE (statistical): {first_pass if first_pass is not None else "unavailable"}

{dist_text}
{counter_text}

RETRIEVED NEIGHBOR CASES (most semantically similar):
{neighbor_text}

Your task is structured evidence synthesis — not prediction. Analyze:
1. What the retrieved data directly and reliably supports
2. What context is absent that would change the answer
3. What physical or astrophysical factors are latent in the data (implied but not stated)
4. What claim about `{target_name}` can be made responsibly given this evidence
5. How the supporting cases cluster vs the counter cases
6. Why confidence should be high, medium, or low
7. Hard limits on what can be claimed
8. Where this case belongs within the observed outcome distribution (semantic tilt)

Rules:
- Do NOT invent values not present in the data
- Do NOT restate the first-pass estimate as your claim unless the data fully supports it
- DO flag when the first-pass estimate is likely biased by an unrepresentative neighborhood
- BE specific about physical mechanisms (e.g. "M-dwarf HZ planets have shorter orbital periods than solar analogs at the same insolation")

Respond in this exact JSON format with no extra text:
{{
  "semantic_tilt": <number between -2 and 2>,
  "data_says": "<1-2 sentences: what the neighborhood directly supports>",
  "missing_context": ["<missing factor 1>", "<missing factor 2>"],
  "latent_factors": ["<implied physical factor 1>", "<implied physical factor 2>"],
  "decision_claim": "<1 sentence: the responsible claim about {target_name} given this evidence>",
  "supporting_distribution": "<qualitative description of supporting case cluster>",
  "counter_distribution": "<qualitative description of counter case cluster>",
  "confidence_reason": "<why confidence is high/medium/low>",
  "limits": ["<limit on claim 1>", "<limit on claim 2>"],
  "supporting_reason": "<why the query belongs higher/lower/typical in the distribution>",
  "counter_reason": "<counterarguments suggesting a different position>"
}}

semantic_tilt guide (CRITICAL: do NOT output the target value, only the tilt):
- -2.0 → case is at the very bottom of the observed distribution (near Q5)
- -1.0 → case is clearly below median (near Q25)
-  0.0 → case is typical / near median (near Q50)
- +1.0 → case is clearly above median (near Q75)
- +2.0 → case is at the very top of the observed distribution (near Q95)"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
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


def run_dcs(
    query: str,
    target_name: str,
    target_type: str,
    neighbors: List[Dict[str, Any]],
    first_pass: Optional[float],
    distribution_summary: Dict[str, Any],
    counter_distribution_summary: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Run the Decision Context Synthesizer.

    Returns:
        (dcs_output_dict, model_used)
        dcs_output_dict is None if the call failed.
    """
    model = str(policy.get("llm_model", DEFAULT_MODEL))
    timeout = int(policy.get("llm_timeout", DEFAULT_TIMEOUT))

    prompt = _build_dcs_prompt(
        query, target_name, target_type, neighbors,
        first_pass, distribution_summary, counter_distribution_summary,
    )

    try:
        resp = requests.post(
            VLLM_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": str(e)}, model

    parsed = _extract_json(content)
    if not parsed:
        return {"error": f"unparseable: {content[:200]}"}, model

    return parsed, model
