"""The asyncio client: the same interface as Nexara, under await."""

import asyncio

from nexara import AsyncNexara


async def main() -> None:
    # `async with` closes the HTTP connection pool on exit.
    async with AsyncNexara(api_key="mock-key") as client:
        result = await client.transcriptions.create(url="https://example.com/audio.mp3")
        print(result.text)

        # Deferred jobs poll with `await asyncio.sleep`, so the event loop stays
        # free while you wait.
        job = await client.transcriptions.create_job(url="https://example.com/long.mp3")
        print("job:", job.job_id, job.status)
        print((await job.wait()).text)


asyncio.run(main())
