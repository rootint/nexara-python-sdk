"""Realtime transcription.

⚠ The protocol underneath is a GUESS — streaming.nexara.ru's wire format is not
documented anywhere we can see. The shape below is what we want to review; the
sample rate, encoding and event fields are placeholders. The open questions are
listed at the top of nexara/resources/realtime.py.
"""

import asyncio
from typing import AsyncIterator

from nexara import Nexara


async def fake_microphone() -> AsyncIterator[bytes]:
    """Whatever produces audio chunks — a mic, a file, a phone bridge."""
    for _ in range(4):
        await asyncio.sleep(0.05)
        yield b"\x00\x01" * 1600


async def main() -> None:
    client = Nexara(api_key="mock-key")

    # The convenient path: hand it an audio iterator, read events out.
    async with client.realtime.connect(sample_rate=16000) as session:
        async for event in session.stream(fake_microphone()):
            marker = "FINAL" if event.is_final else "  ..."
            print(f"{marker}  {event.text}")

    # The manual path, for when audio does not arrive as a neat iterator.
    # Sending and receiving are concurrent, so the send lives in its own task.
    async with client.realtime.connect() as session:

        async def send() -> None:
            async for chunk in fake_microphone():
                await session.send_audio(chunk)
            await session.finish()

        asyncio.create_task(send())
        async for event in session:
            if event.is_final:
                print("FINAL ", event.text)

    # Worth knowing: the session stays alive only as long as audio keeps
    # arriving. Two minutes without any and it is reaped, because billing ticks
    # are what proves it alive. Silence is fine (it is bytes, and it is billed);
    # sending nothing is not.


asyncio.run(main())
