"""The billing resource: balance and itemized usage.

Both endpoints are authenticated with the same API key as transcription and are
scoped to the *account*, not the key: `usage()` returns calls made with every
key the account owns, including keys that have since been deleted.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from .._exceptions import NexaraValidationError, error_for_status
from .._transport import AsyncTransport, Transport
from ..types.billing import Balance, UsageItem, UsagePage

DEFAULT_USAGE_LIMIT = 50
MAX_USAGE_LIMIT = 100
"""The server's own bounds (limit in [1, 100], cursor >= 1). Out-of-range values
come back as a 422 with a FastAPI validation body, which the error mapping would
surface as a bare APIError — so we check first and fail with a clear message."""


def _usage_query(cursor: int | None, limit: int) -> dict[str, Any]:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise NexaraValidationError("limit must be an int.")
    if not 1 <= limit <= MAX_USAGE_LIMIT:
        raise NexaraValidationError(
            f"limit must be between 1 and {MAX_USAGE_LIMIT}; got {limit}."
        )
    query: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise NexaraValidationError("cursor must be an int.")
        if cursor < 1:
            raise NexaraValidationError(f"cursor must be >= 1; got {cursor}.")
        query["cursor"] = cursor
    return query


class Billing:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def balance(self) -> Balance:
        """Current balance, per-minute rate and currency for this account.

        Raises NotFoundError (404) if the API key is unknown to the server.
        """
        response = self._transport.request("GET", "/billing/balance")
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return Balance.model_validate(response.body)

    def usage(self, *, cursor: int | None = None, limit: int = DEFAULT_USAGE_LIMIT) -> UsagePage:
        """One page of billed requests, newest first.

        Keyset-paginated rather than offset-paginated: pass the previous page's
        `next_cursor` to get the next (older) page. New calls landing between
        requests therefore cannot shift rows across the page boundary the way an
        offset would — they simply appear above the first page.

            page = client.billing.usage(limit=20)
            while page.has_more:
                page = client.billing.usage(cursor=page.next_cursor, limit=20)

        `iter_usage()` does that loop for you.

        Args:
            cursor: Return rows older than this one. None starts at the newest.
            limit: Page size, 1..100.
        """
        query = _usage_query(cursor, limit)
        response = self._transport.request("GET", "/billing/usage", params=query)
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return UsagePage.model_validate(response.body)

    def iter_usage(
        self, *, limit: int = DEFAULT_USAGE_LIMIT, max_items: int | None = None
    ) -> Iterator[UsageItem]:
        """Iterate billed requests across pages, newest first.

        Fetches lazily, one page at a time. History is unbounded, so pass
        `max_items` unless you really do want to walk the whole account.
        """
        yielded = 0
        cursor: int | None = None
        while True:
            page = self.usage(cursor=cursor, limit=limit)
            for item in page.items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            # next_cursor is only set alongside has_more; both are checked so a
            # server that ever sends has_more without a cursor stops the loop
            # instead of re-requesting page one forever.
            if not page.has_more or page.next_cursor is None:
                return
            cursor = page.next_cursor


class AsyncBilling:
    """The async twin of Billing. Same interface, awaited."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def balance(self) -> Balance:
        """Current balance for this account. See Billing.balance."""
        response = await self._transport.request("GET", "/billing/balance")
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return Balance.model_validate(response.body)

    async def usage(
        self, *, cursor: int | None = None, limit: int = DEFAULT_USAGE_LIMIT
    ) -> UsagePage:
        """One page of billed requests. See Billing.usage."""
        query = _usage_query(cursor, limit)
        response = await self._transport.request("GET", "/billing/usage", params=query)
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return UsagePage.model_validate(response.body)

    async def iter_usage(
        self, *, limit: int = DEFAULT_USAGE_LIMIT, max_items: int | None = None
    ) -> AsyncIterator[UsageItem]:
        """Async-iterate billed requests across pages. See Billing.iter_usage.

            async for item in client.billing.iter_usage(max_items=100):
                ...
        """
        yielded = 0
        cursor: int | None = None
        while True:
            page = await self.usage(cursor=cursor, limit=limit)
            for item in page.items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            if not page.has_more or page.next_cursor is None:
                return
            cursor = page.next_cursor


def _detail(body: Any) -> str:
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
