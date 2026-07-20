"""The async job model — sync and asyncio flavors."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, PrivateAttr

from .._exceptions import JobFailedError, JobTimeoutError

if TYPE_CHECKING:
    from ..resources.transcriptions import AsyncTranscriptions, Transcriptions

JobStatus = Literal["in_progress", "complete", "error"]
"""Three values, and it is "complete" — not "completed".

The Celery task returns {"status": "completed"} into its own result backend, but
the API serves the status from Postgres, where the enum value is "complete".
"""

DEFAULT_WAIT_TIMEOUT = 1800.0


class _JobFields(BaseModel):
    """A deferred transcription job's data — no behavior.

    Lifetime: the row is deleted 12 hours after *creation* (not completion).
    After that a lookup returns 404 — indistinguishable from a job_id that never
    existed. `NotFoundError` means one or the other, and the server does not say
    which.

    There is no cancel endpoint, so there is no `cancel()`.
    """

    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: datetime | None = None

    result: Any = None
    """Raw payload — what the sync endpoint would have returned for the same
    flags, except that the async path does not strip `words`. Use `wait()` to
    get it parsed. None while in_progress and on error."""

    error: str | None = None
    """Set when status="error"."""


class Job(_JobFields):
    """A deferred job bound to a synchronous client."""

    _resource: Transcriptions | None = PrivateAttr(default=None)

    def _bind(self, resource: Transcriptions) -> None:
        self._resource = resource

    def refresh(self) -> Job:
        """Re-fetch this job and return the fresh copy."""
        if self._resource is None:
            raise RuntimeError("this Job is not bound to a client")
        return self._resource.retrieve_job(self.job_id)

    def wait(
        self,
        *,
        timeout: float | None = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = 2.0,
    ) -> Any:
        """Poll until the job finishes, then return the parsed result.

        The default timeout is finite on purpose. A job can stay `in_progress`
        forever: the worker's finalization is wrapped in a try/except that only
        logs, so if that write fails the work is done and billed but the status
        never changes. Waiting forever would hang the caller until the 12-hour
        cleanup turned the job into a 404. `timeout=None` waits indefinitely if
        that is genuinely what you want.

        Raises JobFailedError on status="error" — note that a failed job is not
        billed at all, so submitting it again is free.
        """
        from ..resources.transcriptions import parse_result

        deadline = None if timeout is None else time.monotonic() + timeout
        job: Job = self

        while job.status == "in_progress":
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(self.job_id, timeout)  # type: ignore[arg-type]
            time.sleep(poll_interval)
            job = job.refresh()

        if job.status == "error":
            raise JobFailedError(job.job_id, job.error or "unknown error")

        return parse_result(job.result)


class AsyncJob(_JobFields):
    """A deferred job bound to an asynchronous client. Same semantics, awaited."""

    _resource: AsyncTranscriptions | None = PrivateAttr(default=None)

    def _bind(self, resource: AsyncTranscriptions) -> None:
        self._resource = resource

    async def refresh(self) -> AsyncJob:
        """Re-fetch this job and return the fresh copy."""
        if self._resource is None:
            raise RuntimeError("this AsyncJob is not bound to a client")
        return await self._resource.retrieve_job(self.job_id)

    async def wait(
        self,
        *,
        timeout: float | None = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = 2.0,
    ) -> Any:
        """Poll until the job finishes, then return the parsed result.

        Identical to `Job.wait()` but non-blocking: the poll interval is an
        `await asyncio.sleep`, so the event loop stays free. See `Job.wait` for
        why the default timeout is finite.
        """
        from ..resources.transcriptions import parse_result

        deadline = None if timeout is None else time.monotonic() + timeout
        job: AsyncJob = self

        while job.status == "in_progress":
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(self.job_id, timeout)  # type: ignore[arg-type]
            await asyncio.sleep(poll_interval)
            job = await job.refresh()

        if job.status == "error":
            raise JobFailedError(job.job_id, job.error or "unknown error")

        return parse_result(job.result)
