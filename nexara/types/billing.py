"""Response models for the billing endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Currency = Literal["RUB", "EUR"]
"""Fixed per account by the account's location (RU -> RUB, rest of world -> EUR).

Not a per-request choice and not settable from the API: it follows the `location`
on the account, so every number in these models is in the same currency.
"""


class Balance(BaseModel):
    """Result of `billing.balance()`."""

    balance: float
    """What is left on the account. Can be negative: the server allows an
    overdraft (a per-account limit) before it starts refusing requests, so a
    balance at or below zero does not by itself mean the next call fails."""

    rate_per_min: float
    """Price of one minute of transcription for *this* account — pricing is
    per-account, not a public list price.

    The server stores a per-second price and multiplies by 60 for this field.
    It covers plain transcription only: `profanity_filter`, `role_tagging` and
    `prompt` (LLM) each add a surcharge that is not reflected here.
    """

    currency: Currency


class UsageApiKey(BaseModel):
    """Which key made the call."""

    id: int

    name: str
    """Never empty: the server substitutes "Key #<id>" for an unnamed key."""

    deleted: bool
    """True if the key has since been (soft-)deleted. Calls made with a deleted
    key stay in the history — that is why this flag exists instead of the row
    simply disappearing."""


class UsageItem(BaseModel):
    """One billed request."""

    timestamp: datetime
    seconds: float
    """Audio duration billed for this call."""

    bytes: int
    task: str
    model: str | None = None
    language: str | None = None

    cost: float | None = None
    """What this call was charged, in the page's `currency`.

    None for rows written before per-request costs were recorded — not zero.
    Treat None as "unknown", not "free"."""

    profanity_filter: bool = False
    role_tagging: bool = False
    """Both are billed as surcharges on top of the per-minute rate."""

    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    """Set only when the call carried a `prompt`. Already folded into `cost`."""

    request_id: str | None = None
    """Matches the `request_id` in the server logs — quote it in support
    requests about a specific call."""

    api_key: UsageApiKey


class UsagePage(BaseModel):
    """One keyset-paginated page of billed requests, newest first."""

    items: list[UsageItem]

    next_cursor: int | None = None
    """Pass as `cursor=` to fetch the next (older) page. Set only when
    `has_more` is True."""

    has_more: bool = False

    currency: Currency | None = None
    """None only for an account with no wallet yet, which also has no items."""
