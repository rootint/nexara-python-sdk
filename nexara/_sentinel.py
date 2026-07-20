"""The NOT_GIVEN sentinel.

Needed because the SDK must distinguish "user did not pass response_format" from
"user passed the default value". The server silently rewrites response_format and
timestamp_granularities when `prompt` is set; we raise on that conflict, but only
when the user actually asked for something else. Comparing against the default
value cannot tell the two apart, so `prompt` would never work.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar, Union

_T = TypeVar("_T")


class NotGiven:
    """Sentinel for "argument omitted", distinct from None (an explicit value)."""

    _instance: NotGiven | None = None

    def __new__(cls) -> NotGiven:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()

NotGivenOr = Union[_T, NotGiven]


def given(value: Any) -> bool:
    """True if the caller actually passed this argument."""
    return not isinstance(value, NotGiven)


def resolve(value: NotGivenOr[_T], default: _T) -> _T:
    return default if isinstance(value, NotGiven) else value
