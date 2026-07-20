"""Canned payloads, shaped exactly the way the server shapes them.

Not invented: these mirror what apigateway actually returns, read off
`utils/transcription.py` (build_transcribe_response, build_diarize_response,
format_response) and `routes/api/v1/inference.py` on branch `develop`.

The shaping functions below are not decoration — they reproduce the server's
real quirks, including the ones we consider bugs. A mock that returned the
*tidy* shape would validate the interface against an API that does not exist.
"""

from __future__ import annotations

from typing import Any

TEXT = "Привет, это тестовая запись."
DURATION = 4.42

_WORDS: list[dict[str, Any]] = [
    {"word": "Привет,", "start": 0.0, "end": 0.62, "prob": 0.991},
    {"word": "это", "start": 0.62, "end": 0.94, "prob": 0.998},
    {"word": "тестовая", "start": 0.94, "end": 1.58, "prob": 0.987},
    {"word": "запись.", "start": 1.58, "end": 2.11, "prob": 0.994},
]

_SEGMENTS: list[dict[str, Any]] = [
    {
        "id": 0,
        "seek": 0.0,
        "start": 0.0,
        "end": DURATION,
        "text": TEXT,
        "tokens": [50364, 3763, 11, 50564],
        "temperature": 0.0,
        "avg_logprob": -0.28,
        "compression_ratio": 1.21,
        "no_speech_prob": 0.008,
    }
]

_SENTENCES: list[dict[str, Any]] = [
    {"start": 0.0, "end": 2.11, "text": TEXT},
    {"start": 2.30, "end": DURATION, "text": "Проверяем предложения."},
]

DIARIZE_TEXT = "Здравствуйте, чем могу помочь? Да, у меня вопрос по заказу."
DIARIZE_DURATION = 8.14

_DIARIZE_SEGMENTS: list[dict[str, Any]] = [
    {"start": 0.0, "end": 3.21, "text": "Здравствуйте, чем могу помочь?", "speaker": "speaker_0"},
    {"start": 3.60, "end": 8.14, "text": "Да, у меня вопрос по заказу.", "speaker": "speaker_1"},
]

_DIARIZE_WORDS: list[dict[str, Any]] = [
    {"word": "Здравствуйте,", "start": 0.0, "end": 0.88, "prob": 0.996, "speaker": "speaker_0"},
    {"word": "чем", "start": 0.88, "end": 1.12, "prob": 0.993, "speaker": "speaker_0"},
    {"word": "могу", "start": 1.12, "end": 1.44, "prob": 0.995, "speaker": "speaker_0"},
    {"word": "помочь?", "start": 1.44, "end": 3.21, "prob": 0.989, "speaker": "speaker_0"},
    {"word": "Да,", "start": 3.60, "end": 3.92, "prob": 0.997, "speaker": "speaker_1"},
    {"word": "у", "start": 3.92, "end": 4.05, "prob": 0.999, "speaker": "speaker_1"},
    {"word": "меня", "start": 4.05, "end": 4.38, "prob": 0.996, "speaker": "speaker_1"},
    {"word": "вопрос", "start": 4.38, "end": 4.91, "prob": 0.994, "speaker": "speaker_1"},
    {"word": "по", "start": 4.91, "end": 5.08, "prob": 0.998, "speaker": "speaker_1"},
    {"word": "заказу.", "start": 5.08, "end": 8.14, "prob": 0.992, "speaker": "speaker_1"},
]


def build_transcribe(granularity: str) -> dict[str, Any]:
    """verbose_json for task=transcribe."""
    payload: dict[str, Any] = {
        "task": "transcribe",
        "language": "ru",
        "duration": DURATION,
        "text": TEXT,
        "segments": [dict(s) for s in _SEGMENTS],
    }
    if granularity == "sentence":
        # The server drops segments entirely and replaces them with sentences.
        payload.pop("segments", None)
        payload["sentences"] = [dict(s) for s in _SENTENCES]
        payload["words"] = [dict(w) for w in _WORDS]
        payload["text"] = TEXT + " Проверяем предложения."
    elif granularity == "word":
        payload["words"] = [dict(w) for w in _WORDS]
    return payload


def build_diarize() -> dict[str, Any]:
    """The full diarize payload, words included.

    Diarization always requests word timestamps from the backend
    (word_timestamps=True is hardcoded), so words are always built here. Whether
    they survive into the response is decided later, per-endpoint.
    """
    return {
        "task": "diarize",
        "language": "ru",
        "duration": DIARIZE_DURATION,
        "text": DIARIZE_TEXT,
        "segments": [dict(s) for s in _DIARIZE_SEGMENTS],
        "words": [dict(w) for w in _DIARIZE_WORDS],
    }


def strip_words_if_not_requested(payload: dict[str, Any], granularity: str) -> dict[str, Any]:
    """The sync handler's word-stripping — and only the sync handler's.

    This is the divergence the SDK cannot paper over: the Celery worker has no
    equivalent, so the async path keeps words that the sync path removes. The
    mock reproduces it deliberately. When the server fixes this (by moving the
    function next to format_response, where it belongs), delete this call from
    the sync branch and the two paths agree.
    """
    if granularity not in ("word", "sentence"):
        payload.pop("words", None)
    return payload


def format_response(payload: dict[str, Any], response_format: str, task: str) -> Any:
    """Mirror of the server's format_response."""
    if response_format == "json":
        # For transcribe, json collapses to just the text. For diarize it does
        # not — the full object comes back.
        return {"text": payload["text"]} if task != "diarize" else payload
    if response_format == "verbose_json":
        return payload
    if response_format == "text":
        if task == "diarize":
            return "\n".join(
                f"{s['speaker'].replace('_', ' ').title()}: {s['text']}" for s in payload["segments"]
            )
        return payload["text"]
    if response_format == "srt":
        return _to_srt(payload["segments"])
    if response_format == "vtt":
        return _to_vtt(payload["segments"])
    raise AssertionError(f"unreachable: response_format={response_format!r}")


def wrap_llm(payload: dict[str, Any], has_schema: bool) -> dict[str, Any]:
    """The LLM wrapper. `usage` is commented out server-side, so it is absent."""
    return {
        "transcription": payload,
        "llm_output": (
            {"summary": "Клиент спрашивает про заказ.", "sentiment": "neutral"}
            if has_schema
            else "Клиент спрашивает про заказ."
        ),
    }


def _ts(seconds: float, sep: str) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def _to_srt(segments: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n"
        for i, s in enumerate(segments, 1)
    )


def _to_vtt(segments: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n" for s in segments
    )
    return f"WEBVTT\n\n{body}"
