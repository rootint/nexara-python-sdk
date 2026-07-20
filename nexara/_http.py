"""The real transport: httpx against api.nexara.ru.

This is the other side of the seam described in `_transport.py`. It turns the
resource layer's `request(method, path, *, form, file)` calls into actual HTTP
and hands back a `Response` in exactly the shape the mock produced, so nothing
above this file changes.

Retry policy is deliberately narrow. We retry 429 (rate limit) and genuine
connection/timeout failures, but NOT 500: on the sync `create()` path the LLM
step is billed *before* a 500 can be raised, so a blind retry pays twice. See
docs/design.md.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import IO, Any

import httpx

from ._exceptions import APIConnectionError, APITimeoutError
from ._transport import FileInput, Response

# Statuses worth another attempt. 500 is intentionally absent (see module docs).
_RETRY_STATUSES = frozenset({429})

_DEFAULT_BACKOFF = 0.5
_MAX_BACKOFF = 8.0


class HttpxTransport:
    """Implements the Transport protocol over a real network.

    `http_client` is an injection point for tests: pass an `httpx.Client` backed
    by `httpx.MockTransport` and no socket is opened. The Authorization header is
    attached per-request, not on the client, so an injected bare client still
    carries auth — and tests can assert on it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 600.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = http_client or httpx.Client(timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response:
        # Build the URL ourselves rather than lean on httpx base_url joining:
        # RFC-3986 join drops the "/v1" prefix for an absolute-path reference.
        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        attempt = 0
        while True:
            # Re-seek a stream so a retry re-reads from the start; bytes need no
            # help, and paths are reopened fresh inside _send on every attempt.
            if attempt and hasattr(file, "seek"):
                file.seek(0)  # type: ignore[union-attr]
            try:
                resp = self._send(method, url, headers, form, file)
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    time.sleep(_backoff(attempt))
                    continue
                raise APITimeoutError(f"request to {url} timed out") from exc
            except httpx.HTTPError as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    time.sleep(_backoff(attempt))
                    continue
                raise APIConnectionError(f"could not reach {url}: {exc}") from exc

            if resp.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                attempt += 1
                time.sleep(_retry_after(resp) or _backoff(attempt))
                continue

            return Response(resp.status_code, _body(resp))

    def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        form: dict[str, Any] | None,
        file: FileInput | None,
    ) -> httpx.Response:
        if method == "GET":
            return self._client.get(url, headers=headers)

        data = _wire_form(form or {})
        if file is None:
            return self._client.post(url, headers=headers, data=data)
        if isinstance(file, (str, Path)):
            # Open here, not in the resource layer: httpx streams a file object
            # chunk by chunk, so an hours-long recording — the whole point of
            # create_job() — is never held in memory. Opening per attempt also
            # makes retries restart from byte 0 with no seek bookkeeping.
            with Path(file).open("rb") as fh:
                return self._client.post(
                    url, headers=headers, data=data, files=_file_part(fh)
                )
        return self._client.post(url, headers=headers, data=data, files=_file_part(file))

    def close(self) -> None:
        self._client.close()


class AsyncHttpxTransport:
    """The async twin of HttpxTransport. Same wire behavior, awaited.

    Shares every serialization/parsing/backoff helper with the sync transport;
    only the I/O and the sleep are awaited. `http_client` is the test injection
    point (an httpx.AsyncClient backed by httpx.MockTransport).
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 600.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response:
        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        attempt = 0
        while True:
            if attempt and hasattr(file, "seek"):
                file.seek(0)  # type: ignore[union-attr]
            try:
                resp = await self._send(method, url, headers, form, file)
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise APITimeoutError(f"request to {url} timed out") from exc
            except httpx.HTTPError as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise APIConnectionError(f"could not reach {url}: {exc}") from exc

            if resp.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                attempt += 1
                await asyncio.sleep(_retry_after(resp) or _backoff(attempt))
                continue

            return Response(resp.status_code, _body(resp))

    async def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        form: dict[str, Any] | None,
        file: FileInput | None,
    ) -> httpx.Response:
        if method == "GET":
            return await self._client.get(url, headers=headers)

        data = _wire_form(form or {})
        if file is None:
            return await self._client.post(url, headers=headers, data=data)
        if isinstance(file, (str, Path)):
            # See the sync twin: streamed from disk, reopened per attempt.
            with Path(file).open("rb") as fh:
                return await self._client.post(
                    url, headers=headers, data=data, files=_file_part(fh)
                )
        return await self._client.post(url, headers=headers, data=data, files=_file_part(file))

    async def aclose(self) -> None:
        await self._client.aclose()


def _file_part(content: bytes | IO[bytes]) -> dict[str, Any]:
    """The multipart `file` field. The filename is a constant "audio": the
    server detects the container from the bytes, not the name."""
    return {"file": ("audio", content, "application/octet-stream")}


def _wire_form(form: dict[str, Any]) -> dict[str, Any]:
    """Serialize native form values to what multipart expects.

    bool -> "true"/"false" (Python's "True"/"False" would confuse the server),
    lists stay lists (httpx emits one field per element, which is how
    `timestamp_granularities[]` is meant to arrive), everything else -> str.
    """
    wire: dict[str, Any] = {}
    for key, value in form.items():
        if isinstance(value, bool):
            wire[key] = "true" if value else "false"
        elif isinstance(value, list):
            wire[key] = [str(v) for v in value]
        else:
            wire[key] = str(value)
    return wire


def _body(resp: httpx.Response) -> Any:
    """Parsed JSON for json-ish responses, a plain str for text/srt/vtt."""
    if "application/json" in resp.headers.get("content-type", ""):
        return resp.json()
    return resp.text


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    delay = _DEFAULT_BACKOFF * (2 ** (attempt - 1))
    return float(min(delay, _MAX_BACKOFF))
