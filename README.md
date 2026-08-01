# Nexara Python SDK

Python SDK for the [Nexara](https://nexara.ru) speech-to-text API: transcription,
speaker diarization, speaker role tagging, structured LLM post-processing, and
account billing.
Full API documentation lives at [docs.nexara.ru](https://docs.nexara.ru).

Requires Python 3.10+.

```bash
pip install nexara
```

## Quickstart

```python
from nexara import Nexara

client = Nexara(api_key="...")  # or set NEXARA_API_KEY

text = client.transcriptions.create(file="audio.mp3").text
```

Pass exactly one of `file=` (path, bytes, or a binary file object — paths are
streamed from disk, not loaded into memory) or `url=`.

## Diarization

```python
call = client.transcriptions.create(file="call.mp3", task="diarize")
for segment in call.segments:
    print(f"{segment.speaker}: {segment.text}")
```

Add meaningful speaker labels with `roles` — `"auto"` lets the model invent
labels, a list restricts them, a dict adds descriptions:

```python
call = client.transcriptions.create(
    file="call.mp3",
    task="diarize",
    roles=["client", "agent"],
)
```

### Emotions

`emotions=True` attaches an emotion to each diarized segment — `label` (one of
`angry`, `sad`, `neutral`, `positive`), `confidence`, and the full `probs`
distribution when the server sends it:

```python
call = client.transcriptions.create(
    file="call.mp3",
    task="diarize",
    model="nexara-ru",
    emotions=True,
)
for segment in call.segments:
    if segment.emotion:
        print(segment.speaker, segment.emotion.label, segment.emotion.confidence)
```

The scoring runs inside the ASR model, so it requires `task="diarize"`,
`model="nexara-ru"` and a JSON response format; anything else raises
`NexaraValidationError` before the upload. Not every segment can be scored, so
check `segment.emotion` rather than assuming it is there. It carries a
per-second surcharge, charged only when emotion was actually returned.

## Long audio: deferred jobs

`create_job()` submits the audio and returns immediately; the result is fetched
by polling. A failed job is never billed, so resubmitting is free.

```python
job = client.transcriptions.create_job(file="long_recording.mp3")
result = job.wait()  # polls; default timeout 1800s

# ...or pick it up later, even from another process:
job = client.transcriptions.retrieve_job(job_id)
```

Job results live for 12 hours from creation; up to 200 jobs may be in progress
per API key. In this SDK "async" always means asyncio — the deferred mode is
`create_job()`, not `AsyncNexara`.

## LLM post-processing

Pass `prompt=` to run an LLM over the transcript, and optionally `json_schema=`
to force structured output:

```python
result = client.transcriptions.create(
    file="meeting.mp3",
    prompt="Summarize the key decisions",
    json_schema={"type": "object", "properties": {"decisions": {"type": "array"}}},
)
print(result.llm_output)          # dict, validated against your schema
print(result.transcription.text)  # the transcript it was derived from
```

## Balance and usage

`client.billing` reports what is on the account and what it has been spent on.
Both endpoints cover the whole account, not just the key you authenticate with:

```python
balance = client.billing.balance()
print(balance.balance, balance.currency, balance.rate_per_min)

# One page of billed calls, newest first.
page = client.billing.usage(limit=20)
for item in page.items:
    print(item.timestamp, item.task, item.cost, item.api_key.name)

# ...or let the SDK walk the pages. History is unbounded — bound it.
for item in client.billing.iter_usage(max_items=200):
    print(item.request_id, item.seconds, item.cost)
```

Paging is keyset-based, not offset-based: pass a page's `next_cursor` as
`cursor=` to get the next (older) page, so calls arriving mid-walk cannot shift
rows across a page boundary. `item.cost` is `None` — not `0` — for rows written
before per-request costs were recorded, and `rate_per_min` covers plain
transcription only (`profanity_filter`, `roles`, `emotions` and `prompt` are
surcharges on top of it).

## asyncio

`AsyncNexara` is the same interface under `await`:

```python
from nexara import AsyncNexara

async with AsyncNexara() as client:
    result = await client.transcriptions.create(file="audio.mp3")
    print(result.text)
```

## Errors and validation

Requests that the server would reject — or, worse, accept, charge for, and
silently do something else with — fail client-side with `NexaraValidationError`
before any network call. Server errors map to typed exceptions by status code:

```python
from nexara import NexaraValidationError, InsufficientBalanceError, RateLimitError

try:
    result = client.transcriptions.create(file="audio.mp3")
except InsufficientBalanceError as e:  # 402
    print(e.detail)
```

429 and connection/timeout failures are retried with exponential backoff
(honoring `Retry-After`). 500 is deliberately **not** retried: on the
synchronous path the request may already have been billed, so a blind retry
could pay twice. Deferred jobs bill only on success, which makes `create_job()`
the safe path for retry-heavy workloads.

## Not yet available

- **Realtime streaming** — the protocol is not yet public; `client.realtime`
  raises `NotImplementedError` for now.
- **Webhooks** — job results are fetched by polling.

## Development

The package is fully typed (`py.typed`, mypy strict). Offline tests
(`pytest`, no network needed) and runnable examples live in the repository;
`NEXARA_USE_MOCK=1` runs everything against an in-memory mock transport.

## License

MIT
