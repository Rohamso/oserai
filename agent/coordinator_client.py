"""HTTP client for the coordinator API.

Registration and result submission retry with exponential backoff (capped)
since those must not silently get lost. Heartbeats and job polling are on
their own cadence already, so a single failure there just waits for the next
tick rather than blocking the agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger("agent.coordinator_client")


class CoordinatorClient:
    def __init__(self, base_url: str, backoff_cap_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.backoff_cap_s = backoff_cap_s
        self.node_id: Optional[str] = None
        self.token: Optional[str] = None
        self._client = httpx.Client(timeout=15.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        delay = 1.0
        while True:
            try:
                resp = self._client.request(method, f"{self.base_url}{path}", headers=self._headers(), **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPError as exc:
                logger.warning(f"coordinator request failed ({exc}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, self.backoff_cap_s)

    def register(self, payload: dict[str, Any]) -> tuple[str, str]:
        resp = self._request_with_backoff("POST", "/nodes/register", json=payload)
        data = resp.json()
        self.node_id, self.token = data["node_id"], data["token"]
        return self.node_id, self.token

    def heartbeat(self, status: str, cpu_pct: float, ram_free_mb: float) -> Optional[int]:
        """Returns the HTTP status code, or None if the coordinator was unreachable."""
        try:
            resp = self._client.request(
                "POST", f"{self.base_url}/nodes/{self.node_id}/heartbeat",
                headers=self._headers(),
                json={"status": status, "cpu_pct": cpu_pct, "ram_free_mb": ram_free_mb},
            )
            return resp.status_code
        except httpx.HTTPError:
            return None

    def next_job(self) -> Optional[dict[str, Any]]:
        """Returns a job payload, or None if none is queued. Raises httpx.HTTPError
        if the coordinator is unreachable/erroring, so callers can back off."""
        resp = self._client.request(
            "GET", f"{self.base_url}/jobs/next", headers=self._headers(),
            params={"node_id": self.node_id},
        )
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json()

    def post_result(self, job_id: str, payload: dict[str, Any]) -> None:
        self._request_with_backoff("POST", f"/jobs/{job_id}/result", json=payload)

    def deregister(self) -> None:
        try:
            self._client.request(
                "POST", f"{self.base_url}/nodes/{self.node_id}/deregister", headers=self._headers()
            )
        except httpx.HTTPError:
            pass

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ #
    # training (DiLoCo) -- heartbeat()/deregister() above are reused as-is,
    # they have no inference-specific assumptions.
    # ------------------------------------------------------------------ #

    def register_training(self, payload: dict[str, Any]) -> tuple[str, str]:
        resp = self._request_with_backoff("POST", "/training/nodes/register", json=payload)
        data = resp.json()
        self.node_id, self.token = data["node_id"], data["token"]
        return self.node_id, self.token

    def get_run(self, run_id: str) -> dict[str, Any]:
        resp = self._request_with_backoff("GET", f"/training/runs/{run_id}")
        return resp.json()

    def get_current_round(self, run_id: str) -> Optional[dict[str, Any]]:
        """Returns the current round payload, or None if the run hasn't
        started yet. Raises httpx.HTTPError if the coordinator is unreachable,
        so callers can back off the same way next_job() does."""
        resp = self._client.request(
            "GET", f"{self.base_url}/training/runs/{run_id}/round/current",
            headers=self._headers(), params={"node_id": self.node_id},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def download_checkpoint(self, checkpoint_id: str) -> bytes:
        resp = self._client.request(
            "GET", f"{self.base_url}/training/checkpoints/{checkpoint_id}",
            headers=self._headers(), params={"node_id": self.node_id},
        )
        resp.raise_for_status()
        return resp.content

    def upload_delta(
        self, round_id: str, blob: bytes,
        local_loss: float, local_steps_completed: int, wall_time_s: float,
    ) -> Optional[dict[str, Any]]:
        """Retries transient failures with backoff, like post_result() --
        a completed round of local training must not silently get lost.
        A 409 means the round already closed server-side (deadline passed
        while this node was still training); that's not transient, so it
        returns None instead of retrying forever."""
        delay = 1.0
        while True:
            try:
                resp = self._client.request(
                    "POST", f"{self.base_url}/training/rounds/{round_id}/delta",
                    headers=self._headers(),
                    params={
                        "node_id": self.node_id, "local_loss": local_loss,
                        "local_steps_completed": local_steps_completed, "wall_time_s": wall_time_s,
                    },
                    content=blob,
                )
                if resp.status_code == 409:
                    logger.warning(f"round {round_id} already closed; discarding this delta")
                    return None
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.warning(f"delta upload failed ({exc}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, self.backoff_cap_s)
