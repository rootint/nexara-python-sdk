"""Realtime transcription over WebSocket.

╔══════════════════════════════════════════════════════════════════════════╗
║ THE PROTOCOL IN THIS MODULE IS A GUESS.                                   ║
║                                                                           ║
║ Streaming lives in a separate service (streaming.nexara.ru) whose code we ║
║ do not have. The codec, sample rate, chunk size, message types and the    ║
║ partial/final distinction below are INVENTED as a plausible strawman.     ║
║ The *shape* of this API (async with + async for) is what we are testing;  ║
║ the wire format will not survive contact with the real service.           ║
╚══════════════════════════════════════════════════════════════════════════╝

What is actually known, from the billing contract between streaming and
apigateway (the only place the two services touch that we can see):

  * it is a WebSocket at streaming.nexara.ru;
  * the client presents its API key at connect, before the upgrade;
  * charges tick every 10 seconds of *received audio*;
  * a key may hold 5 concurrent sessions; a 6th is refused.

What must come from the streaming team before this module is real:

  1. Connect: full URL, and HOW the key is passed — header, query param, or
     first message? Three different clients.
  2. Audio: codec (raw PCM? bit depth? endianness?), sample rate, channel
     count, chunk size, binary frames vs base64.
  3. Server messages: the full list; how partial differs from final (separate
     message types, as Speechmatics does, or a field?); whether timestamps are
     absolute or relative.
  4. End of stream: how the client signals it, whether the server sends a final
     message worth awaiting, whether keepalive is required.
  5. Disconnects — a request, not a question: please give these DIFFERENT close
     codes, or the SDK can only ever say "the connection closed":

       - invalid API key            (authorize -> 403)
       - no funds for the first tick (authorize -> 402)
       - 5 sessions already open    (authorize -> 429)
       - balance ran out mid-session (charge -> insufficient_balance)
       - reaped after 120s of silence
       - internal error             (500)

     The first three happen before the upgrade; if they surface as an HTTP
     status on the handshake, that is enough and no close codes are needed.
     The rest arrive on a live socket, and there they are indistinguishable.

  6. Silence kills the session, and it looks unintentional — please confirm.
     The reaper expires sessions with no charges for 120s, and charges follow
     received bytes. So a client that connects and sends no audio for two
     minutes loses the session via billing, not via any socket timeout.
     Sending silence keeps it alive (silence is bytes, and bytes are billed);
     sending nothing does not. Is there an earlier idle timeout on the
     streaming side that users would hit first?
"""

from __future__ import annotations

import asyncio
import os
from types import TracebackType
from typing import AsyncIterator

from ..types.realtime import RealtimeEvent


class RealtimeSession:
    """A live transcription session. Obtain via `client.realtime.connect()`.

    Sending audio and reading results are two concurrent activities. Either
    drive both with `stream()`, or send from your own task and iterate this
    object directly.
    """

    def __init__(self, *, sample_rate: int, encoding: str) -> None:
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._events: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
        self._closed = False
        self._mock_words: list[str] = []

    async def __aenter__(self) -> RealtimeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def send_audio(self, chunk: bytes) -> None:
        """Send one chunk of audio.

        Note a real consequence of the billing design: a session with no audio
        for 120 seconds is reaped, because charges are what proves it alive.
        Sending silence keeps it open (silence is bytes, and bytes are billed);
        sending nothing does not. Pausing for two minutes loses the session.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        # MOCK: pretend each chunk yields a word, emitting a partial per chunk
        # and a final every fourth one.
        await self._mock_recognize(chunk)

    async def finish(self) -> None:
        """Signal end of audio and let the server flush its last result."""
        if self._closed:
            return
        if self._mock_words:
            await self._events.put(
                RealtimeEvent(text=" ".join(self._mock_words), is_final=True, start=0.0, end=None)
            )
        await self._events.put(None)

    async def close(self) -> None:
        self._closed = True

    async def stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[RealtimeEvent]:
        """Pump `chunks` in and yield events out.

        The convenience path. Without it every caller would write the same
        asyncio.create_task boilerplate, because you cannot send and receive
        from one sequential loop.
        """

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
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    # -- mock recognition -------------------------------------------------

    _MOCK_TRANSCRIPT = ["Привет,", "это", "потоковое", "распознавание."]

    async def _mock_recognize(self, chunk: bytes) -> None:
        index = len(self._mock_words)
        if index >= len(self._MOCK_TRANSCRIPT):
            return
        self._mock_words.append(self._MOCK_TRANSCRIPT[index])
        text = " ".join(self._mock_words)
        is_final = len(self._mock_words) == len(self._MOCK_TRANSCRIPT)
        await self._events.put(
            RealtimeEvent(text=text, is_final=is_final, start=0.0, end=index * 0.5 + 0.5)
        )
        if is_final:
            self._mock_words = []


class Realtime:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def connect(
        self,
        *,
        sample_rate: int = 16000,
        encoding: str = "pcm_s16le",
    ) -> RealtimeSession:
        """Open a session.

        `sample_rate` and `encoding` are GUESSES — see the module docstring.
        Realtime takes none of the transcription flags (language, diarization,
        response_format …); it is audio in, text out.

        Until the streaming protocol is public this raises NotImplementedError:
        the alternative would be silently returning canned mock transcripts to a
        paying caller. Set NEXARA_USE_MOCK=1 to explore the intended interface
        against the offline mock.
        """
        if not os.environ.get("NEXARA_USE_MOCK"):
            raise NotImplementedError(
                "Realtime transcription is not available in this release — the "
                "streaming protocol is not yet public, and this SDK will not "
                "pretend to transcribe. Set NEXARA_USE_MOCK=1 to try the "
                "intended interface against an offline mock."
            )
        return RealtimeSession(sample_rate=sample_rate, encoding=encoding)
