"""Realtime event model.

The streaming service emits tokens incrementally as audio is processed, then one
final frame when the stream ends. There is no server-side endpointing yet, so
`is_final` marks only that terminal frame — not per-utterance boundaries the way
AssemblyAI's `end_of_turn` or Deepgram's `speech_final` do. When endpointing
lands, per-segment finals slot in here without a breaking change.
"""

from __future__ import annotations

from pydantic import BaseModel


class RealtimeToken(BaseModel):
    """One decoder token as sent on the wire. `text` is a subword piece; join the
    pieces to reconstruct words. `pos` is a token index, not a timestamp."""

    token_id: int
    text: str
    pos: int


class RealtimeEvent(BaseModel):
    text: str
    """The running transcript so far (accumulated token text), or the final text
    on the terminal event."""

    is_final: bool
    """False for incremental token updates; True on the single final frame the
    server sends after end-of-stream.

    Not to be confused with the `is_final` in the billing contract, which marks
    the last *charge tick* of a session and has nothing to do with transcripts.
    """

    tokens: list[RealtimeToken] = []
    """The raw tokens delivered in this frame (empty on the final event)."""
