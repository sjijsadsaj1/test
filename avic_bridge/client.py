"""Small HTTP client used from the dependency-sensitive ApexNav process."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import validate_response


class AvicClientError(RuntimeError):
    pass


class AvicHTTPClient:
    def __init__(self, base_url: str, timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _json_request(self, path: str, payload: Optional[Dict[str, Any]] = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise AvicClientError(str(exc)) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise AvicClientError("AVIC service returned invalid JSON") from exc

    def health(self) -> Dict[str, Any]:
        return self._json_request("/health")

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._json_request("/v1/decide", payload)
        return validate_response(response, payload["request_id"])
