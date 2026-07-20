"""Offline tests for the async transport and AsyncNexara.

Driven by httpx.MockTransport over an httpx.AsyncClient — no socket opened. The
tests are plain functions that drive the coroutine with asyncio.run, so no
pytest-asyncio plugin is needed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nexara import (
    APIConnectionError,
    APITimeoutError,
    AsyncNexara,
    InternalServerError,
    Transcription,
)
from nexara._http import AsyncHttpxTransport

BASE = "https://api.nexara.ru/v1"


def _transport(handler, **kwargs) -> AsyncHttpxTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncHttpxTransport(api_key="secret-key", base_url=BASE, http_client=client, **kwargs)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("nexara._http.asyncio.sleep", _instant_sleep)


async def _instant_sleep(_seconds):
    return None


# -- request shaping ---------------------------------------------------------


def test_post_sends_auth_and_multipart():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "привет"})

    async def go():
        return await _transport(handler).request(
            "POST",
            "/audio/transcriptions",
            form={"task": "transcribe", "profanity_filter": True},
            file=b"RIFFaudio",
        )

    resp = asyncio.run(go())
    assert resp.status_code == 200 and resp.body == {"text": "привет"}
    assert seen["auth"] == "Bearer secret-key"
    assert seen["url"] == f"{BASE}/audio/transcriptions"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="file"' in body and b"RIFFaudio" in body
    assert b"true" in body and b"True" not in body


def test_text_response_stays_string():
    def handler(_r):
        return httpx.Response(200, text="WEBVTT\n\nhi", headers={"content-type": "text/vtt"})

    async def go():
        return await _transport(handler).request("POST", "/x", file=b"a")

    body = asyncio.run(go()).body
    assert isinstance(body, str) and body.startswith("WEBVTT")


def test_path_upload_streams_and_reopens_on_retry(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3longrecording")
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        if len(bodies) == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "ok"})

    async def go():
        return await _transport(handler, max_retries=1).request("POST", "/x", file=audio)

    assert asyncio.run(go()).status_code == 200
    assert len(bodies) == 2
    for body in bodies:
        assert b'name="file"' in body and b"ID3longrecording" in body


# -- retry policy ------------------------------------------------------------


def test_429_retried_then_succeeds():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "ok"})

    async def go():
        return await _transport(handler, max_retries=2).request("POST", "/x", file=b"a")

    assert asyncio.run(go()).status_code == 200
    assert calls["n"] == 2


def test_500_not_retried():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    async def go():
        return await _transport(handler, max_retries=3).request("POST", "/x", file=b"a")

    assert asyncio.run(go()).status_code == 500
    assert calls["n"] == 1


def test_connection_error_retried_then_raises():
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    async def go():
        await _transport(handler, max_retries=2).request("POST", "/x", file=b"a")

    with pytest.raises(APIConnectionError):
        asyncio.run(go())
    assert calls["n"] == 3


def test_timeout_raises_api_timeout():
    def handler(_r):
        raise httpx.ReadTimeout("slow")

    async def go():
        await _transport(handler, max_retries=1).request("POST", "/x", file=b"a")

    with pytest.raises(APITimeoutError):
        asyncio.run(go())


# -- end to end through AsyncNexara ------------------------------------------


def test_async_client_create_end_to_end():
    def handler(_r):
        return httpx.Response(200, json={"text": "распознанный текст"})

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AsyncHttpxTransport(api_key="k", base_url=BASE, http_client=client)
        async with AsyncNexara(api_key="k", transport=transport) as nx:
            return await nx.transcriptions.create(file=b"audio")

    result = asyncio.run(go())
    assert isinstance(result, Transcription)
    assert result.text == "распознанный текст"


def test_async_job_wait_polls_to_completion():
    """create_job -> in_progress twice -> complete, driven through AsyncJob.wait."""
    state = {"polls": 0}
    created = "2026-07-20T00:00:00Z"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/async") and request.method == "POST":
            return httpx.Response(200, json={"job_id": "j1", "status": "in_progress",
                                             "created_at": created})
        # GET poll
        state["polls"] += 1
        if state["polls"] < 2:
            return httpx.Response(200, json={"job_id": "j1", "status": "in_progress",
                                             "created_at": created, "result": None})
        return httpx.Response(200, json={"job_id": "j1", "status": "complete",
                                         "created_at": created, "result": {"text": "done"}})

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AsyncHttpxTransport(api_key="k", base_url=BASE, http_client=client)
        async with AsyncNexara(api_key="k", transport=transport) as nx:
            job = await nx.transcriptions.create_job(file=b"a")
            assert job.status == "in_progress"
            return await job.wait(poll_interval=0.01)

    result = asyncio.run(go())
    assert isinstance(result, Transcription)
    assert result.text == "done"


def test_500_surfaces_as_internal_server_error():
    def handler(_r):
        return httpx.Response(500, text="boom")

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AsyncHttpxTransport(api_key="k", base_url=BASE, http_client=client)
        async with AsyncNexara(api_key="k", transport=transport) as nx:
            await nx.transcriptions.create(file=b"audio")

    with pytest.raises(InternalServerError):
        asyncio.run(go())
