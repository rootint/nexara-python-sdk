"""Client constructors."""

from __future__ import annotations

import os

from ._http import AsyncHttpxTransport, HttpxTransport
from ._transport import AsyncTransport, Transport
from .resources.realtime import Realtime
from .resources.transcriptions import AsyncTranscriptions, Transcriptions

DEFAULT_BASE_URL = "https://api.nexara.ru/v1"
"""The server also serves /api/v1 as an alias. /v1 is shorter and matches the
shape OpenAI users expect. Override per-call with base_url= or globally with the
NEXARA_BASE_URL env var — handy for pointing at a locally-run instance."""


class Nexara:
    """Synchronous client for the Nexara speech-to-text API.

        client = Nexara(api_key="...")
        text = client.transcriptions.create(file="audio.mp3").text

    For asyncio, use AsyncNexara — the same interface under ``await``.
    Realtime is not yet available (see resources/realtime.py).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        """
        Args:
            api_key: Defaults to the NEXARA_API_KEY environment variable.
            base_url: API root. Defaults to the NEXARA_BASE_URL env var, then to
                the public endpoint. Point it at your local instance to test.
            timeout: Per-request timeout in seconds.
            max_retries: Retries on 429 and on connection/timeout failures.
                500 is deliberately NOT retried: on the sync path the LLM step is
                billed before a 500 can be raised, so a retry pays twice. Deferred
                jobs bill after the LLM, so a failed job costs nothing and
                resubmitting is free.
            transport: Internal seam. Leave unset in normal use; the test suite
                injects a mock here. Setting NEXARA_USE_MOCK=1 selects the
                in-memory mock transport instead (used to run examples offline).
        """
        key = api_key or os.environ.get("NEXARA_API_KEY")
        if not key:
            raise ValueError(
                "No API key. Pass api_key= or set the NEXARA_API_KEY environment variable."
            )

        self.api_key = key
        self.base_url = base_url or os.environ.get("NEXARA_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries

        self._transport: Transport = transport or _default_transport(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.transcriptions = Transcriptions(self._transport)
        self.realtime = Realtime(self.api_key)

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        close = getattr(self._transport, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> Nexara:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncNexara:
    """Asynchronous client — the same interface as Nexara, under ``await``.

        async with AsyncNexara(api_key="...") as client:
            result = await client.transcriptions.create(file="audio.mp3")
            print(result.text)

    Realtime is not yet available (see resources/realtime.py).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        transport: AsyncTransport | None = None,
    ) -> None:
        """See Nexara for the argument semantics — they are identical."""
        key = api_key or os.environ.get("NEXARA_API_KEY")
        if not key:
            raise ValueError(
                "No API key. Pass api_key= or set the NEXARA_API_KEY environment variable."
            )

        self.api_key = key
        self.base_url = base_url or os.environ.get("NEXARA_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries

        self._transport: AsyncTransport = transport or _default_async_transport(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.transcriptions = AsyncTranscriptions(self._transport)
        self.realtime = Realtime(self.api_key)

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        aclose = getattr(self._transport, "aclose", None)
        if aclose is not None:
            await aclose()

    async def __aenter__(self) -> AsyncNexara:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _default_transport(
    *, api_key: str, base_url: str, timeout: float, max_retries: int
) -> Transport:
    if os.environ.get("NEXARA_USE_MOCK"):
        # Test/demo only: the in-memory mock, so examples run without a network
        # or a real key. Never the default in production use.
        from ._mock.transport import MockTransport

        return MockTransport()
    return HttpxTransport(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def _default_async_transport(
    *, api_key: str, base_url: str, timeout: float, max_retries: int
) -> AsyncTransport:
    if os.environ.get("NEXARA_USE_MOCK"):
        from ._mock.transport import AsyncMockTransport

        return AsyncMockTransport()
    return AsyncHttpxTransport(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
