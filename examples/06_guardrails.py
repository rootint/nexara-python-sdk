"""What the SDK refuses to send, and why.

Every call below is accepted by the raw API. Every one is also billed, and every
one quietly does something other than what was asked. Failing here costs nothing.

Two layers catch these, and it is worth knowing which is which:

  * **mypy** rejects some of them before the code ever runs — no overload accepts
    json_schema without prompt, or num_speakers without task="diarize". Those
    lines carry a `type: ignore` here precisely *because* the type checker is
    doing its job; without it this file would not type-check.
  * **Runtime validation** catches all of them, for callers who do not run a type
    checker — which is most callers.

Run `mypy examples/06_guardrails.py` with the ignores removed to see the first
layer fire.
"""

from typing import Callable

from nexara import Nexara, NexaraValidationError

client = Nexara(api_key="mock-key")
AUDIO = "https://example.com/audio.mp3"


def show(label: str, call: Callable[[], object]) -> None:
    try:
        call()
    except NexaraValidationError as exc:
        print(f"{label}:\n  {exc}\n")
    else:
        print(f"{label}: no error (unexpected)\n")


# --- caught by mypy AND at runtime ---------------------------------------

show(
    "json_schema without prompt",
    lambda: client.transcriptions.create(  # type: ignore[call-overload]
        url=AUDIO, json_schema={"type": "object"}
    ),
)

show(
    "prompt + response_format='srt'",
    lambda: client.transcriptions.create(  # type: ignore[call-overload]
        url=AUDIO, prompt="Summarise.", response_format="srt"
    ),
)

show(
    "num_speakers without diarize",
    lambda: client.transcriptions.create(url=AUDIO, num_speakers=2),  # type: ignore[call-overload]
)

# --- caught at runtime only ----------------------------------------------
# The type system cannot express "a list of exactly one" or "these two fields
# are mutually exclusive", so these need the validator.

show(
    "two granularities",
    lambda: client.transcriptions.create(
        url=AUDIO, response_format="verbose_json", timestamp_granularities=["word", "segment"]
    ),
)

show(
    "sentence granularity + diarize",
    lambda: client.transcriptions.create(
        url=AUDIO, task="diarize", timestamp_granularities=["sentence"]
    ),
)

show("file and url together", lambda: client.transcriptions.create(url=AUDIO, file=b"..."))

show(
    "invalid json_schema",
    lambda: client.transcriptions.create(
        url=AUDIO, prompt="Summarise.", json_schema={"type": "nonsense"}
    ),
)
