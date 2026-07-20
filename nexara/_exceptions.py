"""Exception hierarchy.

Mapped from the HTTP status alone. The API has no machine-readable error code —
only a status and a human `detail` string whose *language depends on the
account* (RU accounts get Russian, ROW gets English). So `detail` is carried
verbatim into the message and never parsed.
"""

from __future__ import annotations


class NexaraError(Exception):
    """Base for everything this SDK raises."""


class NexaraValidationError(NexaraError, ValueError):
    """Raised before the request is sent.

    The server would either reject this with a 400 or — worse — accept it,
    charge for it, and silently do something other than what was asked.
    """


class APIError(NexaraError):
    """The server returned an error status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class BadRequestError(APIError):
    """400."""


class InsufficientBalanceError(APIError):
    """402 — estimated cost exceeds balance plus overdraft."""


class AuthenticationError(APIError):
    """403 — missing or unknown API key.

    The API answers 403 here, not 401, for both "no Authorization header" and
    "key not found".
    """


class NotFoundError(APIError):
    """404."""


class RateLimitError(APIError):
    """429 — 10 req/sec per endpoint, or more than 200 in-progress jobs per key."""


class InternalServerError(APIError):
    """500.

    The sync handler wraps everything in `except Exception` and returns a bare
    500 with no detail, so a transient GPU-provider blip and a permanent failure
    are indistinguishable from the response body.
    """


class APIConnectionError(NexaraError):
    """The request never reached the server."""


class APITimeoutError(APIConnectionError):
    """The request timed out."""


class JobFailedError(NexaraError):
    """An async job finished with status="error"."""

    def __init__(self, job_id: str, error: str) -> None:
        self.job_id = job_id
        self.error = error
        super().__init__(f"job {job_id} failed: {error}")


class JobTimeoutError(NexaraError):
    """`Job.wait()` gave up.

    The job is not cancelled — there is no cancel endpoint. It may still finish;
    retrieve_job() will find it until the 12-hour cleanup removes it.
    """

    def __init__(self, job_id: str, timeout: float) -> None:
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(
            f"job {job_id} still in_progress after {timeout}s. "
            f"It was not cancelled; retrieve_job({job_id!r}) can pick it up later."
        )


_STATUS_MAP: dict[int, type[APIError]] = {
    400: BadRequestError,
    402: InsufficientBalanceError,
    403: AuthenticationError,
    404: NotFoundError,
    429: RateLimitError,
    500: InternalServerError,
}


def error_for_status(status_code: int, detail: str) -> APIError:
    return _STATUS_MAP.get(status_code, APIError)(status_code, detail)
