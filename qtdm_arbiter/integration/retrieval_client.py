from __future__ import annotations

from typing import Any, Dict, List, Sequence

import requests


class RetrievalClient:
    def __init__(self, base_url: str, timeout: float = 15.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def search_findings(self, query: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/search/findings",
            json={"query": query, "filters": filters, "limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("results", []))

    def get_labels(self, finding_ids: Sequence[str], target_name: str) -> Dict[str, Dict[str, Any]]:
        if not finding_ids:
            return {}
        response = self.session.get(
            f"{self.base_url}/labels",
            params=[("target_name", target_name), *[("finding_ids", finding_id) for finding_id in finding_ids]],
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("labels", [])
        return {str(row["finding_id"]): dict(row) for row in rows}

    def search_content(
        self,
        query: str,
        filters: Dict[str, Any],
        limit: int,
        data_types: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"query": query, "filters": filters, "limit": limit}
        if data_types:
            body["data_types"] = data_types
        response = self.session.post(
            f"{self.base_url}/search/content",
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("results", []))

    def label(self, finding_id: str, target_name: str, value: float, labeled_by: str = "human", notes: str | None = None) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/label",
            json={
                "finding_id": finding_id,
                "target_name": target_name,
                "value": value,
                "labeled_by": labeled_by,
                "notes": notes,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return dict(response.json())
