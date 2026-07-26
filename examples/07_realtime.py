"""Realtime transcription over WebSocket (streaming.nexara.ru).

Audio goes out as int16 LE PCM (s16le), 16 kHz mono — the default encoding, and
what mics and WAV files already produce. Results come back as incremental token
updates (is_final=False) followed by one final event (is_final=True).
"""

import asyncio
from typing import AsyncIterator

from nexara import AsyncNexara
from nexara._exceptions import InsufficientBalanceError


async def fake_microphone() -> AsyncIterator[bytes]:
    """Whatever produces audio chunks — a mic, a file, a phone bridge. Yields
    int16 LE PCM at 16 kHz. 1600 samples = 100 ms per chunk."""
    for _ in range(20):
        await asyncio.sleep(0.1)
        yield b"\x00\x00" * 1600  # silence (still billed — bytes are what count)


async def main() -> None:
    client = AsyncNexara(api_key="...")  # or set NEXARA_API_KEY

    # The convenient path: hand it an audio iterator, read events out.
    async with client.realtime.connect() as session:
        print("session:", session.session_id)
        try:
            async for event in session.stream(fake_microphone()):
                marker = "FINAL" if event.is_final else "  ..."
                print(f"{marker}  {event.text}")
        except InsufficientBalanceError as e:
            print("stopped — out of funds:", e)

    # The manual path, for when audio does not arrive as a neat iterator.
    # Sending and receiving are concurrent, so the send lives in its own task.
    async with client.realtime.connect(encoding="s16le") as session:

        async def send() -> None:
            async for chunk in fake_microphone():
                await session.send_audio(chunk)
            await session.finish()  # empty frame -> server flushes the final

        asyncio.create_task(send())
        async for event in session:
            if event.is_final:
                print("FINAL ", event.text)


asyncio.run(main())
