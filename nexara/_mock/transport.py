"""In-memory transport. No sockets, no network.

Its job is to be wrong in exactly the ways the real server is wrong, so that the
interface above it is validated against reality rather than against a tidier API
we wish existed.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .._transport import FileInput, Response
from . import fixtures

_JOB_PATH = re.compile(r"^/audio/transcriptions/async/(?P<job_id>[^/]+)$")

# How many polls a mock job stays in_progress. Non-zero on purpose: a job that
# is complete on the first poll would let a broken wait() loop pass.
_POLLS_BEFORE_COMPLETE = 2


class MockTransport:
    """Implements the Transport protocol against canned fixtures."""

    def __init__(
        self,
        *,
        polls_before_complete: int = _POLLS_BEFORE_COMPLETE,
        fail_jobs: bool = False,
    ) -> None:
        """
        Args:
            polls_before_complete: How long a job stays in_progress.
            fail_jobs: Finish every job with status="error" instead of
                "complete". Without this the JobFailedError path is unreachable
                and would never be exercised.
        """
        self._jobs: dict[str, dict[str, Any]] = {}
        self._polls_before_complete = polls_before_complete
        self._fail_jobs = fail_jobs

    def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response:
        form = form or {}

        if method == "POST" and path == "/audio/transcriptions":
            return Response(200, self._transcribe(form, sync=True))

        if method == "POST" and path == "/audio/transcriptions/async":
            return self._create_job(form)

        match = _JOB_PATH.match(path)
        if method == "GET" and match:
            return self._poll_job(match.group("job_id"))

        return Response(404, {"detail": f"Not Found: {method} {path}"})

    # -- transcription ---------------------------------------------------

    def _transcribe(self, form: dict[str, Any], *, sync: bool) -> Any:
        task = form.get("task", "transcribe")
        fmt = form.get("response_format", "json")
        granularity = form.get("timestamp_granularities[]", ["segment"])[0]

        if task == "diarize":
            payload = fixtures.build_diarize()
            if sync:
                # Only the sync handler strips. The Celery worker does not, so
                # the same request returns words via the async path. This one
                # line is the whole divergence.
                payload = fixtures.strip_words_if_not_requested(payload, granularity)
        else:
            payload = fixtures.build_transcribe(granularity)

        if form.get("prompt"):
            return fixtures.wrap_llm(payload, has_schema=bool(form.get("json_schema")))

        return fixtures.format_response(payload, fmt, task)

    # -- jobs -------------------------------------------------------------

    def _create_job(self, form: dict[str, Any]) -> Response:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "form": form,
            "polls": 0,
            "created_at": datetime.now(timezone.utc),
        }
        return Response(
            200,
            {
                "job_id": job_id,
                "status": "in_progress",
                "created_at": self._jobs[job_id]["created_at"].isoformat(),
            },
        )

    def _poll_job(self, job_id: str) -> Response:
        try:
            uuid.UUID(job_id)
        except ValueError:
            return Response(400, {"detail": f"Invalid job_id: {job_id}"})

        job = self._jobs.get(job_id)
        if job is None:
            # Indistinguishable from "created more than 12 hours ago and swept
            # by cleanup_old_async_jobs". The server cannot tell these apart
            # either, which is exactly why NotFoundError documents both.
            return Response(404, {"detail": "Job not found"})

        job["polls"] += 1
        if job["polls"] <= self._polls_before_complete:
            return Response(
                200,
                {
                    "job_id": job_id,
                    "status": "in_progress",
                    "created_at": job["created_at"].isoformat(),
                    "result": None,
                    "error": None,
                },
            )

        completed_at = (job["created_at"] + timedelta(seconds=44)).isoformat()

        if self._fail_jobs:
            # A failed job is not billed at all: the async path charges after the
            # LLM step, so a failure anywhere before that costs nothing.
            return Response(
                200,
                {
                    "job_id": job_id,
                    "status": "error",
                    "created_at": job["created_at"].isoformat(),
                    "completed_at": completed_at,
                    "result": None,
                    "error": "pipeline failed: backend returned no audio stream",
                },
            )

        return Response(
            200,
            {
                "job_id": job_id,
                "status": "complete",  # not "completed" — the API serves the enum
                "created_at": job["created_at"].isoformat(),
                "completed_at": completed_at,
                "result": self._transcribe(job["form"], sync=False),
                "error": None,
            },
        )


class AsyncMockTransport:
    """The async twin of MockTransport, satisfying the AsyncTransport protocol.

    A thin wrapper: the mock does no real I/O, so `request` just returns the same
    canned Response. Wrapping the sync mock keeps the fixtures and job-state
    machine in exactly one place.
    """

    def __init__(
        self,
        *,
        polls_before_complete: int = _POLLS_BEFORE_COMPLETE,
        fail_jobs: bool = False,
    ) -> None:
        self._inner = MockTransport(
            polls_before_complete=polls_before_complete, fail_jobs=fail_jobs
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response:
        return self._inner.request(method, path, form=form, file=file)
