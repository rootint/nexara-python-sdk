"""Realtime WebSocket client tests.

`websockets.connect` is monkeypatched to return a FakeWS that scripts the
server's frames, so we exercise the real message parsing / event mapping / error
translation without a network or the streaming service.
"""

import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.frames import Close

import nexara.resources.realtime as rt
from nexara._exceptions import (
    APIConnectionError,
    AuthenticationError,
    InsufficientBalanceError,
)
from nexara.resources.realtime import Realtime


class FakeWS:
    """Scripts server frames. `recv()` (used for the welcome) and async iteration
    (used for the result stream) both draw from the same queue. When the queue is
    empty, iteration ends normally unless `close_exc` is set."""

    def __init__(self, incoming, *, post_finish=(), gate=False,
                 close_code=1000, close_exc=None):
        self._q = [json.dumps(m) if isinstance(m, dict) else m for m in incoming]
        # Frames the real server only sends after end-of-stream (the empty
        # binary frame). With gate=True, iteration blocks until that arrives.
        self._post_finish = [json.dumps(m) if isinstance(m, dict) else m
                             for m in post_finish]
        self._gate = gate
        self._finished = asyncio.Event()
        self.sent = []
        self.close_code = close_code
        self.closed = False
        self._close_exc = close_exc

    async def send(self, data):
        self.sent.append(data)
        if data == b"":  # end-of-stream: release the post-finish frames
            self._q.extend(self._post_finish)
            self._finished.set()

    async def recv(self):
        return self._q.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._gate:
            while not self._q and not self._finished.is_set():
                await self._finished.wait()
        if self._q:
            return self._q.pop(0)
        if self._close_exc is not None:
            raise self._close_exc
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


def _realtime(monkeypatch, fake):
    async def fake_connect(url, **kw):
        return fake
    monkeypatch.setattr(rt.websockets, "connect", fake_connect)
    return Realtime("key", url="wss://x/ws")


def test_welcome_tokens_final(monkeypatch):
    fake = FakeWS([
        {"type": "welcome", "version": 1, "session_id": "sess-9", "conn_id": "c1"},
        {"type": "tokens", "tokens": [{"token_id": 1, "text": "Hel", "pos": 0}]},
        {"type": "tokens", "tokens": [{"token_id": 2, "text": "lo", "pos": 1}]},
        {"type": "final", "final_text": "Hello", "total_chunks": 3, "wall_s": 1.2},
    ])
    realtime = _realtime(monkeypatch, fake)

    async def go():
        async with realtime.connect() as s:
            assert s.session_id == "sess-9"
            assert s.conn_id == "c1"
            return [e async for e in s]

    events = asyncio.run(go())
    assert [e.is_final for e in events] == [False, False, True]
    assert events[1].text == "Hello"   # accumulated running text
    assert events[-1].text == "Hello"  # authoritative final_text
    # config frame declared the int16 default
    assert json.loads(fake.sent[0]) == {
        "type": "config", "encoding": "s16le", "sample_rate": 16000,
    }


def test_insufficient_balance_frame_raises(monkeypatch):
    fake = FakeWS([
        {"type": "welcome", "session_id": "s", "conn_id": "c"},
        {"type": "error", "code": "insufficient_balance", "balance_remaining": 2.4},
    ])
    realtime = _realtime(monkeypatch, fake)

    async def go():
        async with realtime.connect() as s:
            async for _ in s:
                pass

    with pytest.raises(InsufficientBalanceError):
        asyncio.run(go())


def test_server_full_frame_raises_and_closes(monkeypatch):
    fake = FakeWS([{"type": "error", "code": "server_full"}])
    realtime = _realtime(monkeypatch, fake)

    async def go():
        async with realtime.connect():
            pass

    with pytest.raises(APIConnectionError):
        asyncio.run(go())
    assert fake.closed  # _connect must not leak the socket on failure


def test_close_1011_raises(monkeypatch):
    fake = FakeWS(
        [{"type": "welcome", "session_id": "s", "conn_id": "c"}],
        close_exc=ConnectionClosed(Close(1011, "session closed"), None),
    )
    realtime = _realtime(monkeypatch, fake)

    async def go():
        async with realtime.connect() as s:
            async for _ in s:
                pass

    with pytest.raises(APIConnectionError):
        asyncio.run(go())


def test_drain_1001_raises(monkeypatch):
    fake = FakeWS(
        [{"type": "welcome", "session_id": "s", "conn_id": "c"}],
        close_code=1001,   # websockets ends iteration silently; client checks code
    )
    realtime = _realtime(monkeypatch, fake)

    async def go():
        async with realtime.connect() as s:
            async for _ in s:
                pass

    with pytest.raises(APIConnectionError):
        asyncio.run(go())


def test_handshake_402_maps_to_insufficient_balance(monkeypatch):
    class FakeResponse:
        status_code = 402
        body = b"insufficient balance: 2.4"

    async def fake_connect(url, **kw):
        raise InvalidStatus(FakeResponse())  # type: ignore[arg-type]  # test double

    monkeypatch.setattr(rt.websockets, "connect", fake_connect)
    realtime = Realtime("key", url="wss://x/ws")

    async def go():
        async with realtime.connect():
            pass

    with pytest.raises(InsufficientBalanceError):
        asyncio.run(go())


def test_handshake_403_maps_to_auth_error(monkeypatch):
    class FakeResponse:
        status_code = 403
        body = b"forbidden"

    async def fake_connect(url, **kw):
        raise InvalidStatus(FakeResponse())  # type: ignore[arg-type]  # test double

    monkeypatch.setattr(rt.websockets, "connect", fake_connect)
    realtime = Realtime("key", url="wss://x/ws")

    async def go():
        async with realtime.connect():
            pass

    with pytest.raises(AuthenticationError):
        asyncio.run(go())


def test_stream_pumps_audio_then_finalizes(monkeypatch):
    # Realistic: the server sends `final` only after the end-of-stream frame.
    fake = FakeWS(
        [{"type": "welcome", "session_id": "s", "conn_id": "c"}],
        post_finish=[{"type": "final", "final_text": "ok",
                      "total_chunks": 2, "wall_s": 0.1}],
        gate=True,
    )
    realtime = _realtime(monkeypatch, fake)

    async def audio():
        yield b"\x00\x00" * 160
        yield b"\x01\x00" * 160

    async def go():
        async with realtime.connect() as s:
            return [e async for e in s.stream(audio())]

    events = asyncio.run(go())
    assert events[-1].is_final and events[-1].text == "ok"
    # config frame declared s16le; then 2 audio frames + the empty end-of-stream
    assert fake.sent[0].startswith('{"type": "config"')
    binary = [m for m in fake.sent if isinstance(m, (bytes, bytearray))]
    assert len(binary) == 3
    assert binary[-1] == b""   # end-of-stream marker sent last
