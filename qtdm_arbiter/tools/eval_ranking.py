#!/usr/bin/env python3
"""
eval_ranking.py — Blind ranking eval for QTDM.

For each test case:
1. Retrieve each candidate's payload from Qdrant (ground truth water_score known but not given to LLM)
2. Baseline: rank by QTDM water_score prediction per candidate (one arbiter call each)
3. LLM ranking: send query + candidate list to vLLM, ask for ranked order
4. Score both with Spearman rank correlation vs true_scores

Usage:
    python3 eval_ranking.py \
        --test-set qtdm_arbiter/tools/test_sets/water_score_ranking.jsonl \
        --out qtdm_arbiter/output/eval_runs/ranking_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
for p in [str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
VLLM_URL = os.environ.get("QTDM_VLLM_URL", "http://localhost:8000/v1/chat/completions")
RETRIEVAL_URL = os.environ.get("QTDM_RETRIEVAL_URL", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("QTDM_REASONING_MODEL", "qwen3.6-27b")
COLLECTION = "exoplanets"

NEIGHBOR_FIELDS = [
    "pl_name", "pl_eqt", "pl_rade", "pl_masse", "pl_insol",
    "pl_orbsmax", "pl_orbper", "pl_dens", "st_teff", "st_spectype",
    "sy_dist", "hz_flag", "discoverymethod",
]


def fetch_planet_payload(pl_name: str) -> Optional[Dict[str, Any]]:
    """Fetch planet payload from Qdrant by name."""
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={
            "limit": 1,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "pl_name", "match": {"value": pl_name}}]},
        },
        timeout=10,
    )
    resp.raise_for_status()
    pts = resp.json()["result"]["points"]
    return pts[0]["payload"] if pts else None


def format_candidate(name: str, payload: Optional[Dict[str, Any]]) -> str:
    """Format a candidate for the LLM prompt — physical fields only, no water_score."""
    if not payload:
        return f"  {name}: (no data available)"
    parts = []
    for f in NEIGHBOR_FIELDS:
        v = payload.get(f)
        if v is not None and v != "" and v is not False:
            parts.append(f"{f}={v}")
    return f"  {name}: " + ", ".join(parts)


def llm_rank(
    query: str,
    candidates: List[str],
    payloads: Dict[str, Optional[Dict[str, Any]]],
    model: str = DEFAULT_MODEL,
    timeout: int = 90,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Ask LLM to rank candidates. Returns (ranked_list, reasoning)."""
    candidate_block = "\n".join(format_candidate(n, payloads.get(n)) for n in candidates)

    prompt = f"""You are a planetary science analyst. Given the following query and list of exoplanets with their physical parameters, rank the planets from most to least suitable.

QUERY: {query}

CANDIDATE PLANETS (physical parameters — no suitability score provided):
{candidate_block}

Instructions:
1. Consider each planet's physical parameters relative to the query.
2. Apply physical reasoning: equilibrium temperature, insolation, radius, density, stellar host type.
3. Rank ALL candidates from most suitable (rank 1) to least suitable.
4. Do not invent data. Use only what is shown above.

Respond in this exact JSON format with no extra text:
{{
  "ranking": [{{"rank": 1, "name": "<planet name>", "reason": "<one sentence>"}}, ...],
  "summary": "<2-3 sentence overall reasoning>"
}}"""

    try:
        resp = requests.post(
            VLLM_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1600,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return None, f"LLM call failed: {e}"

    # Extract JSON
    parsed = None
    try:
        parsed = json.loads(content.strip())
    except Exception:
        m = re.search(r'\{[\s\S]+\}', content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    if not parsed:
        return None, f"unparseable: {content[:300]}"

    ranking = parsed.get("ranking", [])
    # Extract ordered names
    ordered = [r["name"] for r in sorted(ranking, key=lambda x: x.get("rank", 999)) if r.get("name")]
    summary = parsed.get("summary", "")
    return ordered, summary


def spearman_rho(true_scores: Dict[str, float], predicted_order: List[str]) -> float:
    """Spearman rank correlation between true score order and predicted order."""
    # Only score candidates present in both
    names = [n for n in predicted_order if n in true_scores]
    if len(names) < 2:
        return float("nan")

    # True ranks (1 = highest score)
    sorted_true = sorted(names, key=lambda n: -true_scores[n])
    true_rank = {n: i + 1 for i, n in enumerate(sorted_true)}

    # Predicted ranks
    pred_rank = {n: i + 1 for i, n in enumerate(names)}

    n = len(names)
    d2 = sum((true_rank[n] - pred_rank[n]) ** 2 for n in names)
    rho = 1 - (6 * d2) / (n * (n ** 2 - 1))
    return round(rho, 4)


def top_k_accuracy(true_scores: Dict[str, float], predicted_order: List[str], k: int = 3) -> float:
    """Fraction of true top-k that appear in predicted top-k."""
    true_top = set(sorted(true_scores, key=lambda n: -true_scores[n])[:k])
    pred_top = set(predicted_order[:k])
    present = true_top & pred_top
    return round(len(present) / k, 3)


def run_case(case: Dict[str, Any], model: str) -> Dict[str, Any]:
    candidates = case["candidates"]
    true_scores = case["true_scores"]
    query = case["query"]

    # Fetch payloads (without water_score — we'll strip it)
    payloads = {}
    for name in candidates:
        p = fetch_planet_payload(name)
        if p:
            p = {k: v for k, v in p.items() if k != "water_score"}  # blind
        payloads[name] = p

    # LLM ranking
    t0 = time.time()
    llm_order, llm_summary = llm_rank(query, candidates, payloads, model=model)
    latency = round(time.time() - t0, 2)

    # Baseline: true score order (oracle ceiling)
    oracle_order = sorted(candidates, key=lambda n: -true_scores.get(n, 0))

    # Score LLM ranking
    llm_rho = spearman_rho(true_scores, llm_order) if llm_order else float("nan")
    llm_top3 = top_k_accuracy(true_scores, llm_order, k=3) if llm_order else float("nan")

    # Print per-case
    print(f"\n{'='*60}")
    print(f"Case: {case['request_id']} — {case['description']}")
    print(f"Query: {query[:80]}...")
    print(f"\nTrue order:      {' > '.join(oracle_order[:6])}{'...' if len(oracle_order)>6 else ''}")
    if llm_order:
        print(f"LLM ranked:      {' > '.join(llm_order[:6])}{'...' if len(llm_order)>6 else ''}")
        print(f"Spearman rho:    {llm_rho}")
        print(f"Top-3 accuracy:  {llm_top3}")
        if llm_summary:
            print(f"Summary: {llm_summary[:200]}")
    else:
        print(f"LLM failed: {llm_summary}")

    # Detailed rank comparison
    print("\nRank-by-rank:")
    max_len = max(len(oracle_order), len(llm_order or []))
    for i in range(max_len):
        true_name = oracle_order[i] if i < len(oracle_order) else "—"
        pred_name = (llm_order[i] if i < len(llm_order) else "—") if llm_order else "—"
        true_score = true_scores.get(true_name, "?")
        match = "✓" if true_name == pred_name else " "
        print(f"  [{i+1}] {match} true={true_name:<22}({true_score})  llm={pred_name}")

    return {
        "request_id": case["request_id"],
        "description": case["description"],
        "n_candidates": len(candidates),
        "llm_order": llm_order,
        "oracle_order": oracle_order,
        "spearman_rho": llm_rho,
        "top3_accuracy": llm_top3,
        "latency_s": latency,
        "llm_summary": llm_summary,
        "llm_call_success": llm_order is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--out", default="qtdm_arbiter/output/eval_runs/ranking_results.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    cases = []
    with open(args.test_set) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Running ranking eval on {len(cases)} cases with model {args.model}")

    results = []
    for case in cases:
        r = run_case(case, args.model)
        results.append(r)

    # Summary
    answered = [r for r in results if r["llm_call_success"]]
    mean_rho = sum(r["spearman_rho"] for r in answered if r["spearman_rho"] == r["spearman_rho"]) / max(len(answered), 1)
    mean_top3 = sum(r["top3_accuracy"] for r in answered if r["top3_accuracy"] == r["top3_accuracy"]) / max(len(answered), 1)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"Cases run:        {len(results)}")
    print(f"LLM succeeded:    {len(answered)}")
    print(f"Mean Spearman ρ:  {mean_rho:.4f}")
    print(f"Mean top-3 acc:   {mean_top3:.3f}")

    out = {
        "test_set": args.test_set,
        "model": args.model,
        "n_cases": len(results),
        "n_answered": len(answered),
        "mean_spearman_rho": round(mean_rho, 4),
        "mean_top3_accuracy": round(mean_top3, 3),
        "cases": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
