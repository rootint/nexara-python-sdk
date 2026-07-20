"""Response models for task="diarize".

The diarization schema is not the transcription schema with a `speaker` field
bolted on: its segments carry *fewer* fields (no id/seek/tokens/avg_logprob/
compression_ratio/no_speech_prob), because a different pipeline builds them.
Modelling them as one type would promise fields that never arrive.
"""

from __future__ import annotations

from pydantic import BaseModel

from .transcription import Word


class DiarizedWord(Word):
    speaker: str = "speaker_0"


class DiarizedSegment(BaseModel):
    """Note the absence of everything Segment has beyond these four fields."""

    start: float
    end: float
    text: str
    speaker: str


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
