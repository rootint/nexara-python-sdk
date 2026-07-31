"""The billing resource: balance, usage paging, and what goes on the wire.

Model-level assertions run against the in-memory mock; the wire shape (query
string, method, auth) is pinned with httpx.MockTransport so a regression in the
transport's query handling cannot hide behind the mock.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nexara import (
    AsyncNexara,
    Balance,
    Nexara,
    NexaraValidationError,
    NotFoundError,
    UsagePage,
)
from nexara._http import AsyncHttpxTransport, HttpxTransport
from nexara._mock.transport import AsyncMockTransport, MockTransport

BASE = "https://api.nexara.ru/v1"


@pytest.fixture
def client() -> Nexara:
    return Nexara(api_key="k", transport=MockTransport())


@pytest.fixture
def async_client() -> AsyncNexara:
    return AsyncNexara(api_key="k", transport=AsyncMockTransport())


# -- balance -----------------------------------------------------------------


def test_balance_parses(client):
    balance = client.billing.balance()
    assert isinstance(balance, Balance)
    assert balance.balance == 1240.5
    assert balance.rate_per_min == 0.36
    assert balance.currency == "RUB"


def test_async_balance_parses(async_client):
    balance = asyncio.run(async_client.billing.balance())
    assert balance.rate_per_min == 0.36


def test_balance_maps_404_to_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "API key not found."})

    nx = Nexara(
        api_key="k",
        transport=HttpxTransport(
            api_key="k", base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    with pytest.raises(NotFoundError) as exc:
        nx.billing.balance()
    assert exc.value.detail == "API key not found."


# -- usage -------------------------------------------------------------------


def test_usage_page_shape(client):
    page = client.billing.usage(limit=2)
    assert isinstance(page, UsagePage)
    assert len(page.items) == 2
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.currency == "RUB"

    first = page.items[0]
    assert first.task == "diarize"
    assert first.role_tagging is True
    assert first.api_key.name == "production"
    assert first.api_key.deleted is False
    assert first.timestamp.year == 2026


def test_usage_is_newest_first_and_pages_backwards(client):
    first = client.billing.usage(limit=2)
    second = client.billing.usage(cursor=first.next_cursor, limit=2)

    # Strictly older, no overlap.
    assert second.items[0].timestamp < first.items[-1].timestamp
    ids = [i.request_id for i in first.items + second.items]
    assert len(set(ids)) == len(ids)


def test_usage_last_page_has_no_cursor(client):
    page = client.billing.usage(limit=100)
    assert page.has_more is False
    assert page.next_cursor is None


def test_usage_keeps_deleted_keys_and_null_cost(client):
    items = list(client.billing.iter_usage(limit=2))

    deleted = [i for i in items if i.api_key.deleted]
    assert deleted and deleted[0].api_key.name == "Key #9"

    # cost is None — "not recorded", not zero. A model that coerced it to 0.0
    # would silently under-report a bill.
    uncosted = [i for i in items if i.cost is None]
    assert uncosted and uncosted[0].request_id is None


def test_iter_usage_walks_every_page(client):
    one_page = client.billing.usage(limit=100).items
    paged = list(client.billing.iter_usage(limit=2))
    assert [i.request_id for i in paged] == [i.request_id for i in one_page]


def test_iter_usage_stops_at_max_items(client):
    assert len(list(client.billing.iter_usage(limit=2, max_items=3))) == 3


def test_async_iter_usage_walks_every_page(async_client):
    async def run():
        return [item async for item in async_client.billing.iter_usage(limit=2)]

    assert len(asyncio.run(run())) == 5


# -- client-side bounds ------------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_usage_rejects_out_of_range_limit(client, limit):
    with pytest.raises(NexaraValidationError):
        client.billing.usage(limit=limit)


def test_usage_rejects_zero_cursor(client):
    with pytest.raises(NexaraValidationError):
        client.billing.usage(cursor=0)


def test_usage_bounds_checked_before_sending():
    """The bound check must fire without a request — the server's own answer is
    a 422 that maps to a bare APIError, which says much less."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should have been sent")

    nx = Nexara(
        api_key="k",
        transport=HttpxTransport(
            api_key="k", base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    with pytest.raises(NexaraValidationError):
        nx.billing.usage(limit=500)


# -- wire shape --------------------------------------------------------------


def test_usage_sends_query_params_and_auth():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"items": [], "next_cursor": None, "has_more": False, "currency": "EUR"}
        )

    nx = Nexara(
        api_key="secret-key",
        transport=HttpxTransport(
            api_key="secret-key",
            base_url=BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )
    page = nx.billing.usage(cursor=207, limit=25)

    assert seen["method"] == "GET"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["url"] == f"{BASE}/billing/usage?limit=25&cursor=207"
    assert page.currency == "EUR"


def test_usage_omits_cursor_when_not_given():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200, json={"items": [], "next_cursor": None, "has_more": False, "currency": "RUB"}
        )

    nx = Nexara(
        api_key="k",
        transport=HttpxTransport(
            api_key="k", base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    nx.billing.usage()
    assert seen["url"] == f"{BASE}/billing/usage?limit=50"


def test_balance_sends_no_query():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"balance": 1.0, "rate_per_min": 0.36, "currency": "RUB"})

    nx = Nexara(
        api_key="k",
        transport=HttpxTransport(
            api_key="k", base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    nx.billing.balance()
    assert seen["url"] == f"{BASE}/billing/balance"


def test_async_usage_sends_query_params():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200, json={"items": [], "next_cursor": None, "has_more": False, "currency": "RUB"}
        )

    async def run():
        nx = AsyncNexara(
            api_key="k",
            transport=AsyncHttpxTransport(
                api_key="k",
                base_url=BASE,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ),
        )
        async with nx:
            await nx.billing.usage(cursor=10, limit=5)

    asyncio.run(run())
    assert seen["url"] == f"{BASE}/billing/usage?limit=5&cursor=10"


# -- the mock's own bounds ---------------------------------------------------


def test_mock_transport_rejects_bad_bounds_like_the_server():
    """Bypassing the resource layer reaches the mock's 422 — the same answer the
    real server gives, so the mock stays honest about what it is imitating."""
    assert MockTransport().request("GET", "/billing/usage", params={"limit": 0}).status_code == 422
