"""Response models for task="diarize".

The diarization schema is not the transcription schema with a `speaker` field
bolted on: its segments carry *fewer* fields (no id/seek/tokens/avg_logprob/
compression_ratio/no_speech_prob), because a different pipeline builds them.
Modelling them as one type would promise fields that never arrive.
"""

from __future__ import annotations

from pydantic import BaseModel

from .transcription import Word

EMOTION_LABELS: tuple[str, ...] = ("angry", "sad", "neutral", "positive")
"""The labels the server currently emits. It filters anything else out before
the response is built, so in practice `Emotion.label` is one of these."""


class DiarizedWord(Word):
    speaker: str = "speaker_0"


class Emotion(BaseModel):
    """Per-segment emotion, present only when `emotions=True` was requested.

    The server strips its scoring internals (windows, scored_seconds, group_size,
    unit_id, out_of_distribution) before sending, so what arrives is exactly
    these three fields.
    """

    label: str
    """One of EMOTION_LABELS. Deliberately not a Literal: pydantic validates at
    runtime, so pinning the set would turn a *new* server label into a hard parse
    failure for the whole diarization — losing the transcript over an advisory
    field. Compare against EMOTION_LABELS if you need exhaustiveness."""

    confidence: float
    """How sure the model is of `label`, 0..1."""

    probs: dict[str, float] | None = None
    """Per-label probabilities, keyed by EMOTION_LABELS. Absent when the backend
    did not send them — `label` and `confidence` are always there."""


class DiarizedSegment(BaseModel):
    """Note the absence of everything Segment has beyond these four fields."""

    start: float
    end: float
    text: str
    speaker: str

    emotion: Emotion | None = None
    """Set only when the request passed `emotions=True`, and only on segments the
    model could actually score — a scored response can still have segments
    without it, so check per segment rather than per response.

    Never present on `words`: emotion is scored over a segment's audio."""


class Diarization(BaseModel):
    """Result of task="diarize".

    Returned for both response_format="json" and "verbose_json": unlike the
    transcribe path, `json` does not collapse to {"text": ...} here.
    """

    task: str
    language: str
    duration: float
    text: str

    segments: list[DiarizedSegment]

    words: list[DiarizedWord] | None = None
    """Present or absent depending on which endpoint produced this.

    Diarization always asks the backend for word timestamps, but the sync
    handler strips them when granularity is "segment" and the async worker does
    not. Same parameters, different shape. Do not assume.
    """
