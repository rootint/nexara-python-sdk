"""Realtime transcription over WebSocket.

Wire protocol (streaming.nexara.ru, protocol version 1)
-------------------------------------------------------
Connect: WebSocket upgrade carrying the API key in the ``X-API-Key`` header.
The key is authorized *before* the upgrade, so a rejection arrives as an HTTP
status on the handshake (translated to the usual exceptions):

    401 missing key · 402 no funds · 403 bad key · 429 too many sessions · 503 down

Client -> server:
  * optional first text frame:
        {"type":"config","encoding":"s16le"|"f32le","sample_rate":16000}
    ``s16le`` (int16 LE) is the default — half the bytes of float32 for the same
    audio. Send audio matching the declared encoding.
  * binary frames: raw PCM, 16 kHz mono, in the declared encoding.
  * empty binary frame (0 bytes): end of stream.

Server -> client (every frame carries a ``type``):
  * {"type":"welcome","version":1,"session_id":…,"conn_id":…}
  * {"type":"config_applied","encoding":…,"sample_rate":…}
  * {"type":"tokens","tokens":[{"token_id","text","pos"}]}   (incremental)
  * {"type":"final","final_text":…,"total_chunks":…,"wall_s":…}
  * {"type":"error","code":…,"message"|"balance_remaining":…}

Live-socket close codes: 1008 balance exhausted · 1011 session closed · 1001
server draining · 1000 normal (after ``final``).

Note the billing consequence: charges follow *received* audio, and the backend
reaps a session after 120 s with no charge. Sending silence keeps it alive
(silence is bytes, and bytes are billed); sending nothing for two minutes loses
it. There is no earlier idle timeout on the streaming side.
"""

from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Any, AsyncIterator

from .._exceptions import (
    APIConnectionError,
    APIError,
    InsufficientBalanceError,
    NexaraValidationError,
    error_for_status,
)
from ..types.realtime import RealtimeEvent, RealtimeToken

try:
    import websockets as websockets  # explicit re-export (tests patch rt.websockets)
    from websockets.exceptions import ConnectionClosed, InvalidStatus
except ImportError:  # pragma: no cover - exercised only without the extra
    websockets = None  # type: ignore[assignment]


_MAX_MESSAGE = 2**22


class RealtimeSession:
    """A live transcription session. Obtain via ``client.realtime.connect()`` and
    enter it as an async context manager.

    Sending audio and reading results are concurrent: either drive both with
    ``stream()``, or send from your own task and iterate this object directly.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        api_key_header: str,
        encoding: str,
        sample_rate: int,
    ) -> None:
        if websockets is None:
            raise RuntimeError(
                "Realtime needs the 'websockets' package. "
                "Install it with:  pip install nexara[realtime]"
            )
        self._url = url
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._ws: "websockets.ClientConnection | None" = None
        self._audio_done = False
        self._text_parts: list[str] = []

        #: Backend session id from the ``welcome`` frame (None in dev/no-billing).
        self.session_id: str | None = None
        #: Per-connection id for log correlation.
        self.conn_id: str | None = None

    # -- lifecycle ---------------------------------------------------------
    async def __aenter__(self) -> RealtimeSession:
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                self._url,
                additional_headers={self._api_key_header: self._api_key},
                max_size=_MAX_MESSAGE,
                ping_timeout=300,
            )
        except InvalidStatus as e:  # handshake rejected before the upgrade
            status = e.response.status_code
            detail = _body_text(e) or f"streaming authorize rejected ({status})"
            raise error_for_status(status, detail) from None

        # From here the socket is open; close it if anything below fails so a
        # rejected handshake-after-upgrade (e.g. server_full) can't leak it.
        try:
            # The server greets with `welcome` (or an error, e.g. server_full).
            first = await self._recv_json()
            if first.get("type") == "error":
                self._raise_error_frame(first)
            if first.get("type") == "welcome":
                self.session_id = first.get("session_id")
                self.conn_id = first.get("conn_id")

            # Declare the audio encoding; the server applies it before any audio.
            await self._ws.send(json.dumps({
                "type": "config",
                "encoding": self._encoding,
                "sample_rate": self._sample_rate,
            }))
        except BaseException:
            await self._ws.close()
            raise

    # -- sending -----------------------------------------------------------
    async def send_audio(self, chunk: bytes) -> None:
        """Send one binary audio frame. ``chunk`` must be raw PCM matching the
        session's encoding (int16 LE by default), 16 kHz mono."""
        if self._ws is None:
            raise RuntimeError("session is not connected")
        await self._ws.send(chunk)

    async def finish(self) -> None:
        """Signal end of audio (empty binary frame) and let the server flush its
        final result. Idempotent."""
        if self._audio_done or self._ws is None:
            return
        self._audio_done = True
        try:
            await self._ws.send(b"")
        except ConnectionClosed:
            pass

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()

    # -- receiving ---------------------------------------------------------
    async def stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[RealtimeEvent]:
        """Pump ``chunks`` in and yield events out — the convenience path, so
        callers don't hand-roll the concurrent send task."""
        import asyncio

        async def pump() -> None:
            try:
                async for chunk in chunks:
                    await self.send_audio(chunk)
            finally:
                await self.finish()

        task = asyncio.create_task(pump())
        try:
            async for event in self:
                yield event
        finally:
            task.cancel()

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        if self._ws is None:
            raise RuntimeError("session is not connected")
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # server speaks JSON text; ignore stray binary
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "tokens":
                    toks = [RealtimeToken(**t) for t in msg.get("tokens", [])]
                    self._text_parts.extend(t.text for t in toks)
                    yield RealtimeEvent(
                        text="".join(self._text_parts), is_final=False, tokens=toks
                    )
                elif kind == "final":
                    yield RealtimeEvent(
                        text=msg.get("final_text", ""), is_final=True
                    )
                    return
                elif kind == "error":
                    self._raise_error_frame(msg)
                # welcome / config_applied / unknown: ignore
        except ConnectionClosed as e:
            self._raise_close(e)
            return
        # websockets ends iteration silently on close codes 1000/1001 (no raise).
        # If the server drained us (1001) before a `final`, surface it.
        if getattr(self._ws, "close_code", None) == 1001:
            raise APIConnectionError("streaming server is shutting down")

    # -- helpers -----------------------------------------------------------
    async def _recv_json(self) -> dict[str, Any]:
        assert self._ws is not None
        raw = await self._ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            return {}
        data: dict[str, Any] = json.loads(raw)
        return data

    def _raise_error_frame(self, msg: dict[str, Any]) -> None:
        code = msg.get("code")
        if code == "insufficient_balance":
            raise InsufficientBalanceError(
                402,
                f"balance exhausted mid-session "
                f"(remaining {msg.get('balance_remaining')})",
            )
        if code == "server_full":
            raise APIConnectionError(
                "streaming server is at capacity; retry shortly"
            )
        if code in ("bad_config", "config_after_audio"):
            raise NexaraValidationError(msg.get("message", code))
        raise APIError(500, msg.get("message") or code or "streaming error")

    def _raise_close(self, exc: ConnectionClosed) -> None:
        close = exc.rcvd or exc.sent
        code = close.code if close is not None else 1006
        if code == 1008:  # balance exhausted (usually preceded by an error frame)
            raise InsufficientBalanceError(402, "balance exhausted mid-session")
        if code == 1011:
            raise APIConnectionError("streaming session was closed by the server")
        if code == 1001:
            raise APIConnectionError("streaming server is shutting down")
        if code in (1000, 1005):
            return  # normal closure
        raise APIConnectionError(f"streaming connection closed abnormally ({code})")


def _body_text(exc: "InvalidStatus") -> str:
    try:
        body = exc.response.body
        return body.decode(errors="replace").strip() if body else ""
    except Exception:
        return ""


class Realtime:
    def __init__(self, api_key: str, *, url: str, api_key_header: str = "X-API-Key") -> None:
        self._api_key = api_key
        self._url = url
        self._api_key_header = api_key_header

    def connect(
        self,
        *,
        encoding: str = "s16le",
        sample_rate: int = 16000,
    ) -> RealtimeSession:
        """Open a realtime session.

        Use as ``async with client.realtime.connect() as session: ...``.

        Args:
            encoding: Wire encoding of the audio you will send. ``s16le`` (int16
                little-endian, the default) or ``f32le`` (float32). int16 is half
                the bytes for the same audio — prefer it unless your source is
                already float32.
            sample_rate: Must be 16000 (the model's rate); other values are
                rejected by the server.

        Realtime takes none of the transcription flags (language, diarization,
        response_format …); it is audio in, text out.
        """
        return RealtimeSession(
            url=self._url,
            api_key=self._api_key,
            api_key_header=self._api_key_header,
            encoding=encoding,
            sample_rate=sample_rate,
        )
