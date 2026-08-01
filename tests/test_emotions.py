"""The `emotions` parameter: its guardrails, its wire form, and its result shape.

Emotion recognition is billed per second, so every rule here exists to stop a
request being charged for output it could never receive. The server 400s each of
these combinations; failing client-side just makes it free.
"""

from __future__ import annotations

import asyncio

import pytest

from nexara import AsyncNexara, Emotion, Nexara, NexaraValidationError
from nexara._mock.transport import AsyncMockTransport, MockTransport
from nexara._validation import validate_and_build_form

AUDIO = "https://example.com/call.mp3"


@pytest.fixture
def client() -> Nexara:
    return Nexara(api_key="k", transport=MockTransport())


def _diarize(**kwargs):
    params = {"url": AUDIO, "task": "diarize", "model": "nexara-ru", "emotions": True}
    params.update(kwargs)
    return validate_and_build_form(**params)


# -- guardrails --------------------------------------------------------------


def test_emotions_requires_diarize():
    with pytest.raises(NexaraValidationError):
        _diarize(task="transcribe")


def test_emotions_requires_nexara_ru():
    for model in ("whisper-1", "nexara-1"):
        with pytest.raises(NexaraValidationError):
            _diarize(model=model)


def test_emotions_rejects_the_default_model():
    """The SDK default is whisper-1, so `emotions=True` alone must not sail
    through and be rejected only after the audio was uploaded."""
    with pytest.raises(NexaraValidationError):
        validate_and_build_form(url=AUDIO, task="diarize", emotions=True)


@pytest.mark.parametrize("fmt", ["text", "srt", "vtt"])
def test_emotions_rejects_subtitle_formats(fmt):
    # The emotion object hangs off a segment; srt/vtt/text have nowhere to put it.
    with pytest.raises(NexaraValidationError):
        _diarize(response_format=fmt)


@pytest.mark.parametrize("fmt", ["json", "verbose_json"])
def test_emotions_allows_json_formats(fmt):
    assert _diarize(response_format=fmt)["emotions"] is True


def test_emotions_false_never_conflicts():
    """A caller who did not ask for emotion must not be rejected for using
    whisper-1, transcribe, or srt."""
    form = validate_and_build_form(url=AUDIO, task="transcribe", emotions=False, response_format="srt")
    assert "emotions" not in form


def test_emotions_survives_a_prompt():
    """A prompt rewrites response_format to verbose_json, which is allowed. The
    check has to run *after* that rewrite — as it does on the server."""
    form = _diarize(response_format="verbose_json", prompt="Summarise.")
    assert form["response_format"] == "verbose_json"
    assert form["emotions"] is True


# -- wire form ---------------------------------------------------------------


def test_emotions_omitted_when_not_requested():
    # The server defaults the field to False; sending it says nothing extra.
    assert "emotions" not in validate_and_build_form(url=AUDIO, task="diarize")


def test_emotions_reaches_the_wire():
    seen: dict[str, object] = {}

    class RecordingTransport(MockTransport):
        def request(self, method, path, *, form=None, file=None, params=None):
            seen["form"] = form
            return super().request(method, path, form=form, file=file, params=params)

    nx = Nexara(api_key="k", transport=RecordingTransport())
    nx.transcriptions.create(url=AUDIO, task="diarize", model="nexara-ru", emotions=True)
    assert seen["form"]["emotions"] is True


def test_create_job_carries_emotions():
    seen: dict[str, object] = {}

    class RecordingTransport(MockTransport):
        def request(self, method, path, *, form=None, file=None, params=None):
            if path.endswith("/async"):
                seen["form"] = form
            return super().request(method, path, form=form, file=file, params=params)

    nx = Nexara(api_key="k", transport=RecordingTransport())
    nx.transcriptions.create_job(url=AUDIO, task="diarize", model="nexara-ru", emotions=True)
    assert seen["form"]["emotions"] is True


# -- result shape ------------------------------------------------------------


def test_emotion_parses_onto_segments(client):
    result = client.transcriptions.create(
        url=AUDIO, task="diarize", model="nexara-ru", emotions=True
    )
    emotion = result.segments[0].emotion
    assert isinstance(emotion, Emotion)
    assert emotion.label == "neutral"
    assert emotion.confidence == 0.87
    assert set(emotion.probs) == {"angry", "sad", "neutral", "positive"}


def test_unscored_segment_has_no_emotion(client):
    """A scored response can still contain segments the model could not score —
    the attribute is None there, not a zero-confidence emotion."""
    result = client.transcriptions.create(
        url=AUDIO, task="diarize", model="nexara-ru", emotions=True
    )
    assert result.segments[1].emotion is None


def test_no_emotion_without_the_flag(client):
    result = client.transcriptions.create(url=AUDIO, task="diarize")
    assert all(segment.emotion is None for segment in result.segments)


def test_words_never_carry_emotion(client):
    result = client.transcriptions.create(
        url=AUDIO,
        task="diarize",
        model="nexara-ru",
        emotions=True,
        timestamp_granularities=["word"],
    )
    assert result.words
    assert not any(hasattr(word, "emotion") for word in result.words)


def test_unknown_label_still_parses():
    """`label` is a str, not a Literal, on purpose: a label the server adds later
    must not turn the whole diarization into a parse error over one advisory
    field."""
    assert Emotion.model_validate({"label": "surprised", "confidence": 0.4}).label == "surprised"


def test_probs_are_optional():
    assert Emotion.model_validate({"label": "sad", "confidence": 0.6}).probs is None


def test_async_emotions_round_trip():
    async def run():
        nx = AsyncNexara(api_key="k", transport=AsyncMockTransport())
        return await nx.transcriptions.create(
            url=AUDIO, task="diarize", model="nexara-ru", emotions=True
        )

    assert asyncio.run(run()).segments[0].emotion.label == "neutral"


# -- billing -----------------------------------------------------------------


def test_usage_reports_emotions(client):
    items = list(client.billing.iter_usage())
    scored = [item for item in items if item.emotions]
    assert scored and scored[0].model == "nexara-ru" and scored[0].task == "diarize"
