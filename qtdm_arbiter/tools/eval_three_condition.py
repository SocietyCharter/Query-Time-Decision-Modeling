#!/usr/bin/env python3
"""
eval_three_condition.py — Compare three conditions on obscure planets:

  1. LLM-only:   planet name only → LLM estimates water_score cold
  2. RAG-only:   retrieved payload fields → LLM estimates from params (no neighborhood stats)
  3. QTDM-full:  retrieved payload + neighborhood distribution summary → LLM estimates

Candidates: recent/obscure planets (post-2022, non-famous) with known water_score in Qdrant.
Ground truth: water_score from our formula (not public, not in LLM training data).
Metric: MAE, rank correlation vs true water_score ordering.
"""
from __future__ import annotations

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
RETRIEVAL_URL = os.environ.get("QTDM_RETRIEVAL_URL", "http://localhost:8000")
VLLM_URL = os.environ.get("QTDM_VLLM_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("QTDM_REASONING_MODEL", "qwen3.6-27b")
COLLECTION = "exoplanets"

DISPLAY_FIELDS = [
    "pl_eqt", "pl_rade", "pl_masse", "pl_bmasse", "pl_insol",
    "pl_orbsmax", "pl_orbper", "pl_dens", "st_teff", "st_spectype",
    "sy_dist", "hz_flag", "discoverymethod", "disc_year",
]

CANDIDATES = [
    ("L 98-59 f",   0.997),
    ("GJ 1002 b",   0.6614),
    ("TOI-715 b",   0.6555),
    ("TOI-1266 d",  0.5925),
    ("LP 890-9 c",  0.4812),
    ("TOI-2094 b",  0.4502),
    ("Barnard e",   0.3719),
    ("Gliese 12 b", 0.2793),
    ("TOI-7166 b",  0.2699),
    ("TOI-5799 c",  0.2598),
]


def fetch_payload(name: str) -> Optional[Dict[str, Any]]:
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 1, "with_payload": True, "with_vector": False,
              "filter": {"must": [{"key": "pl_name", "match": {"value": name}}]}},
        timeout=10,
    )
    resp.raise_for_status()
    pts = resp.json()["result"]["points"]
    return pts[0]["payload"] if pts else None


def fetch_neighborhood_stats(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get neighborhood stats using semantic search (content endpoint) then
    compute water_score distribution from the returned payloads directly.
    Also pulls a separate HZ-filtered sample from Qdrant for comparison.
    """
    eqt   = payload.get("pl_eqt")
    rade  = payload.get("pl_rade")
    insol = payload.get("pl_insol")
    st_teff = payload.get("st_teff")

    # Build a focused semantic query
    query_parts = []
    if rade and rade < 2.5:
        query_parts.append("rocky or super-Earth planet")
    else:
        query_parts.append("sub-Neptune planet")
    if st_teff and st_teff < 4000:
        query_parts.append("M-dwarf host star")
    if eqt:
        query_parts.append(f"equilibrium temperature {eqt:.0f} K")
    if insol:
        query_parts.append(f"insolation {insol:.2f} Earth flux")
    if rade:
        query_parts.append(f"radius {rade:.2f} Earth radii")
    query_parts.append("habitable zone candidate")
    query = ", ".join(query_parts)

    # Semantic search via retrieval service
    try:
        resp = requests.post(
            f"{RETRIEVAL_URL}/search/content",
            json={"query": query, "filters": {}, "limit": 25, "data_types": ["exoplanet_record"]},
            timeout=15,
        )
        resp.raise_for_status()
        sem_results = resp.json().get("results", [])
        # Filter out the planet itself
        sem_results = [r for r in sem_results if r.get("pl_name") != name]
    except Exception:
        sem_results = []

    # Compute stats from semantic neighbors
    sem_scores = [r["water_score"] for r in sem_results if r.get("water_score") is not None]
    sem_eqts   = [r["pl_eqt"] for r in sem_results if r.get("pl_eqt") is not None]
    sem_hz     = sum(1 for r in sem_results if r.get("hz_flag"))
    sem_raders = [r["pl_rade"] for r in sem_results if r.get("pl_rade") is not None]

    # Separately: pull physically similar HZ planets from Qdrant by eqt range
    hz_stats: Dict[str, Any] = {}
    if eqt is not None:
        eqt_lo, eqt_hi = eqt - 60, eqt + 60
        try:
            resp2 = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                json={
                    "limit": 50, "with_payload": True, "with_vector": False,
                    "filter": {"must": [
                        {"key": "hz_flag", "match": {"value": True}},
                        {"key": "pl_eqt", "range": {"gte": eqt_lo, "lte": eqt_hi}},
                    ]}
                },
                timeout=10,
            )
            resp2.raise_for_status()
            hz_pts = [p["payload"] for p in resp2.json()["result"]["points"]
                      if p["payload"].get("pl_name") != name]
            hz_ws = sorted([p["water_score"] for p in hz_pts if p.get("water_score") is not None])
            if hz_ws:
                n = len(hz_ws)
                hz_stats = {
                    "hz_eqt_match_count": n,
                    "hz_eqt_match_water_median": round(hz_ws[n // 2], 4),
                    "hz_eqt_match_water_mean":   round(sum(hz_ws) / n, 4),
                    "hz_eqt_match_water_p25":    round(hz_ws[max(0, n//4)], 4),
                    "hz_eqt_match_water_p75":    round(hz_ws[min(n-1, 3*n//4)], 4),
                    "hz_eqt_match_above_0.5":    sum(1 for s in hz_ws if s > 0.5),
                }
        except Exception:
            pass

    # Filter to HZ neighbors only for water_score stats — avoids zero-inflation from non-HZ majority
    hz_sem = [r for r in sem_results if r.get("hz_flag")]
    hz_sem_scores = sorted([r["water_score"] for r in hz_sem if r.get("water_score") is not None])

    stats: Dict[str, Any] = {
        "semantic_neighbors_retrieved": len(sem_results),
        "semantic_hz_neighbors": sem_hz,
    }
    if hz_sem_scores:
        n = len(hz_sem_scores)
        stats.update({
            "hz_neighbor_water_median": round(hz_sem_scores[n // 2], 4),
            "hz_neighbor_water_mean":   round(sum(hz_sem_scores) / n, 4),
            "hz_neighbor_water_p25":    round(hz_sem_scores[max(0, n//4)], 4),
            "hz_neighbor_water_p75":    round(hz_sem_scores[min(n-1, 3*n//4)], 4),
            "hz_neighbors_above_0.5":   sum(1 for s in hz_sem_scores if s > 0.5),
            "hz_neighbors_count":       n,
        })
    elif sem_hz > 0:
        stats["hz_neighbor_note"] = f"{sem_hz} HZ neighbors found but water_score not computed for them"

    if sem_eqts:
        es = sorted(sem_eqts)
        stats["semantic_eqt_median"] = round(es[len(es)//2], 1)
    stats.update(hz_stats)
    return stats


def llm_call(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 90) -> Optional[str]:
    try:
        resp = requests.post(
            VLLM_URL,
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300, "temperature": 0.1},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def extract_float(text: str) -> Optional[float]:
    m = re.search(r'"estimate"\s*:\s*([0-9.]+)', text)
    if m:
        try: return float(m.group(1))
        except: pass
    m = re.search(r'\b0\.[0-9]+\b', text)
    if m:
        try: return float(m.group(0))
        except: pass
    return None


def prompt_llm_only(name: str) -> str:
    return f"""You are a planetary science analyst. Estimate the water habitability score (0.0 to 1.0) for the exoplanet "{name}".

The score reflects likelihood of surface liquid water: 1.0 = highly suitable, 0.0 = not suitable.

Respond with only this JSON:
{{"estimate": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def prompt_rag(name: str, payload: Dict[str, Any]) -> str:
    fields = {k: payload[k] for k in DISPLAY_FIELDS if payload.get(k) is not None}
    params = ", ".join(f"{k}={v}" for k, v in fields.items())
    return f"""You are a planetary science analyst. Estimate the water habitability score (0.0 to 1.0) for exoplanet "{name}".

Physical parameters: {params}

The score reflects likelihood of surface liquid water: 1.0 = highly suitable, 0.0 = not suitable.

Respond with only this JSON:
{{"estimate": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def prompt_qtdm(name: str, payload: Dict[str, Any], neighborhood: Dict[str, Any]) -> str:
    fields = {k: payload[k] for k in DISPLAY_FIELDS if payload.get(k) is not None}
    params = ", ".join(f"{k}={v}" for k, v in fields.items())
    nbr_lines = []
    for k, v in neighborhood.items():
        if v is not None:
            nbr_lines.append(f"  {k}: {v}")
    nbr_block = "\n".join(nbr_lines) if nbr_lines else "  (unavailable)"
    return f"""You are a planetary science analyst. Estimate the water habitability score (0.0 to 1.0) for exoplanet "{name}".

Physical parameters: {params}

Neighborhood context (from 20 most physically similar known planets):
{nbr_block}

The score reflects likelihood of surface liquid water: 1.0 = highly suitable, 0.0 = not suitable.
Use both the planet's own parameters AND the neighborhood distribution as evidence.

Respond with only this JSON:
{{"estimate": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def spearman(true_scores: Dict[str, float], pred_scores: Dict[str, float]) -> float:
    names = [n for n in true_scores if n in pred_scores and pred_scores[n] is not None]
    if len(names) < 2:
        return float("nan")
    true_rank = {n: i+1 for i, n in enumerate(sorted(names, key=lambda n: -true_scores[n]))}
    pred_rank = {n: i+1 for i, n in enumerate(sorted(names, key=lambda n: -pred_scores[n]))}
    n = len(names)
    d2 = sum((true_rank[n] - pred_rank[n])**2 for n in names)
    return round(1 - 6*d2 / (n*(n**2-1)), 4)


def mae(true_scores: Dict[str, float], pred_scores: Dict[str, float]) -> float:
    pairs = [(true_scores[n], pred_scores[n]) for n in true_scores if n in pred_scores and pred_scores[n] is not None]
    if not pairs: return float("nan")
    return round(sum(abs(t-p) for t,p in pairs) / len(pairs), 4)


def main():
    print(f"Three-condition eval — {len(CANDIDATES)} candidates\n")
    print(f"{'Planet':<22} {'True':>6} | {'LLM-only':>9} {'RAG':>9} {'QTDM':>9}")
    print("-" * 65)

    results = []
    true_scores = dict(CANDIDATES)
    llm_preds, rag_preds, qtdm_preds = {}, {}, {}
    reasonings = {}

    for name, true_ws in CANDIDATES:
        payload = fetch_payload(name)
        if not payload:
            print(f"{name:<22} {'?':>6} | MISSING PAYLOAD")
            continue

        # Condition 1: LLM only
        t0 = time.time()
        out1 = llm_call(prompt_llm_only(name))
        llm_est = extract_float(out1 or "")
        t1 = time.time()

        # Condition 2: RAG only
        out2 = llm_call(prompt_rag(name, payload))
        rag_est = extract_float(out2 or "")
        t2 = time.time()

        # Condition 3: QTDM (RAG + neighborhood stats)
        nbr = fetch_neighborhood_stats(name, payload)
        out3 = llm_call(prompt_qtdm(name, payload, nbr))
        qtdm_est = extract_float(out3 or "")
        t3 = time.time()

        llm_preds[name]  = llm_est
        rag_preds[name]  = rag_est
        qtdm_preds[name] = qtdm_est
        reasonings[name] = {"llm": out1, "rag": out2, "qtdm": out3, "neighborhood": nbr}

        def fmt(v): return f"{v:.3f}" if v is not None else "  err"
        print(f"{name:<22} {true_ws:>6.4f} | {fmt(llm_est):>9} {fmt(rag_est):>9} {fmt(qtdm_est):>9}")

    print()
    print(f"{'MAE':<22} {'':>6} | {mae(true_scores, llm_preds):>9.4f} {mae(true_scores, rag_preds):>9.4f} {mae(true_scores, qtdm_preds):>9.4f}")
    print(f"{'Spearman rho':<22} {'':>6} | {spearman(true_scores, llm_preds):>9.4f} {spearman(true_scores, rag_preds):>9.4f} {spearman(true_scores, qtdm_preds):>9.4f}")

    # Difference test — if conditions are too similar, flag it
    print()
    print("Divergence check (per planet, |QTDM - LLM-only|):")
    diffs = []
    for name, _ in CANDIDATES:
        lv = llm_preds.get(name)
        qv = qtdm_preds.get(name)
        if lv is not None and qv is not None:
            diff = abs(qv - lv)
            diffs.append(diff)
            print(f"  {name:<22} diff={diff:.3f}")
    if diffs:
        print(f"  Mean divergence QTDM vs LLM-only: {sum(diffs)/len(diffs):.3f}")
        if sum(diffs)/len(diffs) < 0.05:
            print("  ⚠ LOW DIVERGENCE — conditions are nearly identical, QTDM adding no signal")

    out_path = Path(__file__).parent.parent / "output" / "eval_runs" / "three_condition_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "candidates": CANDIDATES,
            "true_scores": true_scores,
            "llm_only": llm_preds,
            "rag_only": rag_preds,
            "qtdm_full": qtdm_preds,
            "mae": {"llm_only": mae(true_scores, llm_preds), "rag_only": mae(true_scores, rag_preds), "qtdm_full": mae(true_scores, qtdm_preds)},
            "spearman": {"llm_only": spearman(true_scores, llm_preds), "rag_only": spearman(true_scores, rag_preds), "qtdm_full": spearman(true_scores, qtdm_preds)},
            "reasonings": reasonings,
        }, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
