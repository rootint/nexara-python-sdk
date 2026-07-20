"""Response models for task="transcribe"."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Word(BaseModel):
    word: str
    """Already stripped by the server; no leading whisper-style space."""

    start: float
    end: float

    prob: float
    """Note the name: the server renames `probability` to `prob` when building
    the response. It is `prob` on the wire and `prob` here."""


class Segment(BaseModel):
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Sentence(BaseModel):
    start: float
    end: float
    text: str


class Transcription(BaseModel):
    """Result of response_format="json" — the server sends only the text."""

    text: str


class VerboseTranscription(BaseModel):
    """Result of response_format="verbose_json"."""

    task: str
    language: str
    duration: float
    text: str

    segments: list[Segment] | None = None
    """Absent when timestamp_granularities=["sentence"] — the server drops
    `segments` and sends `sentences` instead."""

    words: list[Word] | None = None
    """Absent unless word- or sentence-level timestamps were requested.

    With the default granularity ("segment") this is None: the server strips
    words from the response. It is also None-vs-present *inconsistently between
    create() and create_job()* — the async worker does not strip. See
    docs/design.md, "words: sync и async возвращают разный JSON".
    """

    sentences: list[Sentence] | None = None
    """Only when timestamp_granularities=["sentence"]."""


class LLMResult(BaseModel):
    """Result when `prompt` is set: the transcription is wrapped."""

    transcription: VerboseTranscription
    """Always verbose_json — `prompt` forces it server-side."""

    llm_output: str | dict[str, Any]
    """A string when json_schema was not given, an object shaped by the schema
    when it was.

    The server has a `usage` field (tokens, cost) commented out, so it never
    arrives and is deliberately not modelled here.
    """
