"""Python SDK for the Nexara speech-to-text API.

Synchronous (`Nexara`) and asyncio (`AsyncNexara`) clients talk to the API over
httpx. Realtime is not yet available. See docs/design.md.
"""

import sys

if sys.version_info < (3, 10):  # pragma: no cover
    # pip refuses to install on <3.10 (requires-python), but a source checkout
    # on PYTHONPATH bypasses that and would die later inside pydantic with a
    # baffling "unsupported operand type(s) for |" instead of this message.
    raise ImportError(
        "nexara requires Python 3.10 or newer; this is Python "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

__version__ = "0.2.0"

from ._client import AsyncNexara, Nexara
from ._exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    InsufficientBalanceError,
    InternalServerError,
    JobFailedError,
    JobTimeoutError,
    NexaraError,
    NexaraValidationError,
    NotFoundError,
    RateLimitError,
    SyncLLMTimeoutError,
)
from ._sentinel import NOT_GIVEN, NotGiven
from .types.diarization import Diarization, DiarizedSegment, DiarizedWord
from .types.job import AsyncJob, Job, JobStatus
from .types.realtime import RealtimeEvent
from .types.transcription import (
    LLMResult,
    Segment,
    Sentence,
    Transcription,
    VerboseTranscription,
    Word,
)

__all__ = [
    "NOT_GIVEN",
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AsyncJob",
    "AsyncNexara",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "Diarization",
    "DiarizedSegment",
    "DiarizedWord",
    "InsufficientBalanceError",
    "InternalServerError",
    "Job",
    "JobFailedError",
    "JobStatus",
    "JobTimeoutError",
    "LLMResult",
    "Nexara",
    "NexaraError",
    "NexaraValidationError",
    "NotFoundError",
    "NotGiven",
    "RateLimitError",
    "RealtimeEvent",
    "Segment",
    "Sentence",
    "SyncLLMTimeoutError",
    "Transcription",
    "VerboseTranscription",
    "Word",
]
