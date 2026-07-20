"""Offline tests for the real transport, driven by httpx.MockTransport.

No socket is opened: httpx.MockTransport intercepts every request and lets us
assert on exactly what the SDK put on the wire, and control what comes back.
"""

from __future__ import annotations

import httpx
import pytest

from nexara import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    Nexara,
    Transcription,
)
from nexara._http import HttpxTransport

BASE = "https://api.nexara.ru/v1"


def _transport(handler, **kwargs) -> HttpxTransport:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HttpxTransport(api_key="secret-key", base_url=BASE, http_client=client, **kwargs)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retries back off with time.sleep; make it instant for the tests."""
    monkeypatch.setattr("nexara._http.time.sleep", lambda _s: None)


# -- request shaping ---------------------------------------------------------


def test_post_sends_auth_url_and_multipart_fields():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "привет"})

    t = _transport(handler)
    resp = t.request(
        "POST",
        "/audio/transcriptions",
        form={
            "task": "diarize",
            "response_format": "json",
            "timestamp_granularities[]": ["segment"],
            "profanity_filter": True,
            "model": "whisper-1",
        },
        file=b"RIFFfakeaudio",
    )

    assert resp.status_code == 200
    assert resp.body == {"text": "привет"}
    assert seen["auth"] == "Bearer secret-key"
    # Full /v1 prefix preserved, not dropped by RFC-3986 path joining.
    assert seen["url"] == f"{BASE}/audio/transcriptions"
    assert seen["method"] == "POST"

    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="file"' in body and b"RIFFfakeaudio" in body
    assert b'name="task"' in body and b"diarize" in body
    # bool serialized lowercase, not Python's "True".
    assert b'name="profanity_filter"' in body and b"true" in body
    assert b"True" not in body
    assert b'name="timestamp_granularities[]"' in body


def test_url_mode_sends_no_file_part():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        seen["ctype"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"text": "ok"})

    t = _transport(handler)
    t.request(
        "POST",
        "/audio/transcriptions",
        form={"task": "transcribe", "url": "https://example.com/a.mp3"},
        file=None,
    )

    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="file"' not in body
    assert b"example.com" in body


def test_roles_json_reaches_the_wire():
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"task": "diarize", "language": "ru",
                                         "duration": 1.0, "text": "x", "segments": []})

    t = _transport(handler)
    # The resource layer JSON-encodes list roles; here we hand the wire value in.
    t.request(
        "POST",
        "/audio/transcriptions",
        form={"task": "diarize", "roles": '["operator", "client"]'},
        file=b"a",
    )
    assert b'name="roles"' in seen["body"]
    assert b"operator" in seen["body"]


def test_path_upload_streams_and_reopens_on_retry(tmp_path):
    """A path is opened by the transport itself (streamed, not preloaded) and
    reopened from byte 0 on retry."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3longrecording")
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        if len(bodies) == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "ok"})

    resp = _transport(handler, max_retries=1).request("POST", "/x", file=audio)
    assert resp.status_code == 200
    assert len(bodies) == 2
    for body in bodies:  # both attempts carry the full file
        assert b'name="file"' in body and b"ID3longrecording" in body


def test_io_upload_reseeked_on_retry():
    """A caller-provided stream is rewound before the retry resends it."""
    import io

    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        if len(bodies) == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "ok"})

    resp = _transport(handler, max_retries=1).request(
        "POST", "/x", file=io.BytesIO(b"RIFFwavedata")
    )
    assert resp.status_code == 200
    assert len(bodies) == 2
    assert b"RIFFwavedata" in bodies[0]
    assert b"RIFFwavedata" in bodies[1]  # not empty from an exhausted stream


# -- response parsing --------------------------------------------------------


def test_json_response_parsed_text_response_left_as_string():
    def json_handler(_r):
        return httpx.Response(200, json={"text": "hi"})

    def text_handler(_r):
        return httpx.Response(200, text="1\n00:00:00,000 --> 00:00:01,000\nhi\n",
                              headers={"content-type": "application/x-subrip"})

    assert _transport(json_handler).request("POST", "/x", file=b"a").body == {"text": "hi"}
    body = _transport(text_handler).request("POST", "/x", file=b"a").body
    assert isinstance(body, str) and body.startswith("1\n")


# -- retry policy ------------------------------------------------------------


def test_429_is_retried_then_succeeds():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "ok"})

    resp = _transport(handler, max_retries=2).request("POST", "/x", file=b"a")
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_500_is_not_retried():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    resp = _transport(handler, max_retries=3).request("POST", "/x", file=b"a")
    assert resp.status_code == 500
    assert calls["n"] == 1  # billed-before-500 hazard: never retried


def test_connection_error_retried_then_raises():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    with pytest.raises(APIConnectionError):
        _transport(handler, max_retries=2).request("POST", "/x", file=b"a")
    assert calls["n"] == 3  # initial + 2 retries


def test_timeout_raises_api_timeout():
    def handler(_r):
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(APITimeoutError):
        _transport(handler, max_retries=1).request("POST", "/x", file=b"a")


# -- job polling path --------------------------------------------------------


def test_get_builds_job_path():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"job_id": "abc", "status": "complete",
                                         "created_at": "2026-07-18T00:00:00Z"})

    _transport(handler).request("GET", "/audio/transcriptions/async/abc")
    assert seen["method"] == "GET"
    assert seen["url"] == f"{BASE}/audio/transcriptions/async/abc"


# -- end to end through the client -------------------------------------------


def test_client_create_end_to_end():
    def handler(_r):
        return httpx.Response(200, json={"text": "распознанный текст"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpxTransport(api_key="k", base_url=BASE, http_client=client)
    nx = Nexara(api_key="k", transport=transport)

    result = nx.transcriptions.create(file=b"audio")
    assert isinstance(result, Transcription)
    assert result.text == "распознанный текст"


def test_500_surfaces_as_internal_server_error():
    def handler(_r):
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpxTransport(api_key="k", base_url=BASE, http_client=client)
    nx = Nexara(api_key="k", transport=transport)

    with pytest.raises(InternalServerError):
        nx.transcriptions.create(file=b"audio")
