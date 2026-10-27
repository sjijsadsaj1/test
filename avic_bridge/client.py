"""Small HTTP client used from the dependency-sensitive ApexNav process."""

from __future__ import annotations

import json
import os
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
        archive=os.environ.get('APEXNAV_SCORE_RESPONSE_ARCHIVE')
        replay=os.environ.get('APEXNAV_SCORE_RESPONSE_REPLAY')
        store=None
        if archive or replay:
            from .response_replay import ResponseArchive
            if not hasattr(self,'_response_archive'):
                self._response_archive=ResponseArchive(archive,replay)
            store=self._response_archive
            frozen=store.lookup(payload)
            if frozen is not None:return frozen
        response = self._json_request("/v1/decide", payload)
        response=validate_response(response, payload["request_id"])
        if store:
            try:store.save(payload,response)
            except (OSError,ValueError,TypeError) as exc:
                store.log('ARCHIVE_ERROR',reason=type(exc).__name__,request_id=payload['request_id'])
        return response
