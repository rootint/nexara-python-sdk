"""Deferred transcription for long audio.

`create()` waits for the result. `create_job()` hands you a Job and returns.
The word "async" is not used for either: it belongs to asyncio.
"""

from nexara import Nexara

client = Nexara(api_key="mock-key")

job = client.transcriptions.create_job(url="https://example.com/long.mp3", task="diarize")
print(f"submitted {job.job_id}, status={job.status}")

result = job.wait(poll_interval=0.1)
print(result.text)

# The same request through create() and create_job() returns DIFFERENT shapes.
# Diarization always computes word timestamps; the sync handler strips them at
# the default granularity, and the Celery worker does not. This is a server-side
# inconsistency, not an SDK choice — which is why `words` is Optional everywhere
# and you should not assume either way.
sync_result = client.transcriptions.create(url="https://example.com/long.mp3", task="diarize")
print(f"words via create():     {'present' if sync_result.words else 'stripped'}")
print(f"words via create_job(): {'present' if result.words else 'stripped'}")
