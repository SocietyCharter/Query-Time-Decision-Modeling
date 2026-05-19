"""Exoplanet-specific field hints used by the public demo domain.

The QTDM core is domain-neutral. This adapter keeps planetary-science field
names, unit hints, and compact neighbor formatting outside the generic
decision machinery.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


PHYSICAL_FIELDS = [
    "pl_eqt",
    "pl_insol",
    "pl_rade",
    "pl_dens",
    "pl_masse",
    "pl_orbper",
    "pl_orbsmax",
    "st_teff",
    "st_mass",
    "sy_dist",
    "water_score",
]

QUERY_FEATURES = list(PHYSICAL_FIELDS)

DEFAULT_NEIGHBOR_FIELDS = [
    "pl_name",
    "pl_eqt",
    "pl_rade",
    "pl_masse",
    "pl_insol",
    "pl_orbsmax",
    "pl_orbper",
    "pl_dens",
    "st_teff",
    "st_spectype",
    "sy_dist",
    "hz_flag",
    "water_score",
]

UNIT_HINTS = {
    "pl_eqt": "K (Kelvin)",
    "pl_rade": "Earth radii",
    "pl_masse": "Earth masses",
    "pl_insol": "Earth flux",
    "pl_orbper": "days",
    "pl_orbsmax": "AU",
    "pl_dens": "g/cm^3",
    "st_teff": "K (Kelvin)",
    "st_mass": "solar masses",
    "sy_dist": "parsecs",
    "water_score": "0-1 probability score",
}


def domain_hint(target_name: str) -> str:
    unit = UNIT_HINTS.get(target_name, "")
    return f"exoplanet target; units: {unit}" if unit else "exoplanet target"


def format_neighbor(case: Dict[str, Any], fields: Iterable[str] | None = None) -> str:
    payload = case.get("payload") or case.get("metadata") or case
    parts: List[str] = []
    for field in fields or DEFAULT_NEIGHBOR_FIELDS:
        value = payload.get(field)
        if value is not None and value != "" and value is not False:
            parts.append(f"{field}={value}")
    if not parts:
        case_id = payload.get("finding_id") or payload.get("chunk_id") or payload.get("source_id")
        text = payload.get("chunk_text") or payload.get("summary") or payload.get("content")
        if case_id:
            parts.append(f"id={case_id}")
        if text:
            parts.append(f"text={str(text)[:120]}")
    return ", ".join(parts)

