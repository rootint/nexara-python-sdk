"""Realtime event model.

HYPOTHESIS — see nexara/resources/realtime.py. The real service may name these
fields differently or split partial/final into separate message types the way
Speechmatics does.
"""

from __future__ import annotations

from pydantic import BaseModel


class RealtimeEvent(BaseModel):
    text: str

    is_final: bool
    """False while the text may still change, True once it is settled.

    Not to be confused with the `is_final` in the billing contract, which marks
    the last *charge tick* of a session and has nothing to do with transcripts.
    """

    start: float | None = None
    end: float | None = None
