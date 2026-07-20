"""The transcriptions resource."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Literal, Sequence, overload

from .._exceptions import error_for_status
from .._sentinel import NOT_GIVEN, NotGivenOr, given
from .._transport import AsyncTransport, Transport
from .._validation import Granularity, ResponseFormat, Task, validate_and_build_form
from ..types.diarization import Diarization
from ..types.job import AsyncJob, Job
from ..types.transcription import LLMResult, Transcription, VerboseTranscription

FileTypes = str | Path | bytes | IO[bytes]


def parse_result(body: Any) -> Any:
    """Turn a raw payload into the model the overloads promised.

    The shape is read off the payload itself rather than from the flags that
    produced it. That is not cleverness for its own sake: a Job fetched by id in
    another process has no memory of its flags, and defaulting to "probably a
    transcription" would parse a diarization into a Transcription — silently
    dropping every speaker, because pydantic ignores extra fields. Wrong data is
    worse than an error, and the payload already says what it is.
    """
    if not isinstance(body, dict):
        return body  # text / srt / vtt come back as plain strings
    if "llm_output" in body:
        return LLMResult.model_validate(body)
    if body.get("task") == "diarize":
        return Diarization.model_validate(body)
    if "task" in body:
        return VerboseTranscription.model_validate(body)
    return Transcription.model_validate(body)


class Transcriptions:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    # -- create ----------------------------------------------------------

    @overload
    def create(
        self,
        *,
        prompt: str,
        file: FileTypes | None = ...,
        url: str | None = ...,
        task: Literal["transcribe"] = ...,
        language: NotGivenOr[str | None] = ...,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> LLMResult: ...

    @overload
    def create(
        self,
        *,
        task: Literal["diarize"],
        response_format: Literal["json", "verbose_json"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        num_speakers: NotGivenOr[int | None] = ...,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = ...,
        diarization_setting: NotGivenOr[str] = ...,
        model: NotGivenOr[str] = ...,
    ) -> Diarization: ...

    @overload
    def create(
        self,
        *,
        task: Literal["diarize"],
        response_format: Literal["text", "srt", "vtt"],
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        num_speakers: NotGivenOr[int | None] = ...,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = ...,
        diarization_setting: NotGivenOr[str] = ...,
        model: NotGivenOr[str] = ...,
    ) -> str: ...

    @overload
    def create(
        self,
        *,
        response_format: Literal["verbose_json"],
        task: Literal["transcribe"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> VerboseTranscription: ...

    @overload
    def create(
        self,
        *,
        response_format: Literal["text", "srt", "vtt"],
        task: Literal["transcribe"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> str: ...

    @overload
    def create(
        self,
        *,
        task: Literal["transcribe"] = ...,
        response_format: Literal["json"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> Transcription: ...

    def create(
        self,
        *,
        task: Task = "transcribe",
        file: FileTypes | None = None,
        url: str | None = None,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        response_format: NotGivenOr[ResponseFormat] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        dictionary: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        num_speakers: NotGivenOr[int | None] = NOT_GIVEN,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = NOT_GIVEN,
        diarization_setting: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> Any:
        """Transcribe audio and wait for the result.

        Pass exactly one of `file` or `url`. A str/Path `file` is streamed from
        disk by the transport, so long recordings are never held in memory.

        `profanity_filter=True` masks Russian profanity server-side. The masking
        is morphological (lemma-based, not substring) and length-preserving, so
        word timings stay aligned; it carries a per-second billing surcharge.

        `roles` turns on speaker role_tagging and requires task="diarize". Pass
        "auto" to let the model invent labels, a list to restrict them, or a dict
        to add descriptions; lists and dicts are JSON-encoded for you.

        Not in the public docs (docs.nexara.ru) — supported by the server but
        undocumented, so treat them as unstable and subject to change:
          * `dictionary="medical"` — domain vocabulary correction;
          * `timestamp_granularities=["sentence"]` — sentence-level timestamps
            (the server drops `segments` and returns `sentences` instead).

        Note on retries: a 500 raised here may arrive *after* your transcription
        was billed (the LLM step runs after the charge on the sync path), so a
        retry can pay twice. `create_job()` bills after the LLM instead, which
        makes a failed job free. See docs/design.md.
        """
        form = validate_and_build_form(
            task=task,
            file=file,
            url=url,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            profanity_filter=profanity_filter,
            dictionary=dictionary,
            prompt=prompt,
            json_schema=json_schema,
            num_speakers=num_speakers,
            roles=roles,
            diarization_setting=diarization_setting,
            model=model,
        )
        response = self._transport.request(
            "POST", "/audio/transcriptions", form=form, file=file
        )
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return parse_result(response.body)

    # -- jobs -------------------------------------------------------------

    def create_job(
        self,
        *,
        task: Task = "transcribe",
        file: FileTypes | None = None,
        url: str | None = None,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        response_format: NotGivenOr[ResponseFormat] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        dictionary: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        num_speakers: NotGivenOr[int | None] = NOT_GIVEN,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = NOT_GIVEN,
        diarization_setting: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> Job:
        """Submit audio for deferred transcription and return immediately.

        The result is fetched by polling — there are no webhooks. (The API has a
        `callback_url` field; it is accepted and then dropped on the floor, so
        this SDK does not expose it.)

        Results live for 12 hours from creation, then vanish. Up to 200 jobs may
        be in_progress per key.

        Takes the same flags as `create()` — including the undocumented
        `dictionary` and `timestamp_granularities=["sentence"]` noted there.
        """
        form = validate_and_build_form(
            task=task,
            file=file,
            url=url,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            profanity_filter=profanity_filter,
            dictionary=dictionary,
            prompt=prompt,
            json_schema=json_schema,
            num_speakers=num_speakers,
            roles=roles,
            diarization_setting=diarization_setting,
            model=model,
        )
        response = self._transport.request(
            "POST", "/audio/transcriptions/async", form=form, file=file
        )
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))

        job = Job.model_validate(response.body)
        job._bind(self)
        return job

    def retrieve_job(self, job_id: str) -> Job:
        """Fetch a job by id — including from a process that never submitted it.

        No flags needed: `wait()` reads the result's shape off the payload.

        Raises NotFoundError both for an unknown job_id and for one that existed
        and was swept after 12 hours. The server does not distinguish them.
        """
        response = self._transport.request("GET", f"/audio/transcriptions/async/{job_id}")
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        job = Job.model_validate(response.body)
        job._bind(self)
        return job


class AsyncTranscriptions:
    """The async twin of Transcriptions. Same interface, awaited.

    Validation, form-building and payload parsing are shared verbatim with the
    sync resource (they are pure functions); only the transport call is awaited.
    """

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    # -- create ----------------------------------------------------------

    @overload
    async def create(
        self,
        *,
        prompt: str,
        file: FileTypes | None = ...,
        url: str | None = ...,
        task: Literal["transcribe"] = ...,
        language: NotGivenOr[str | None] = ...,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> LLMResult: ...

    @overload
    async def create(
        self,
        *,
        task: Literal["diarize"],
        response_format: Literal["json", "verbose_json"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        num_speakers: NotGivenOr[int | None] = ...,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = ...,
        diarization_setting: NotGivenOr[str] = ...,
        model: NotGivenOr[str] = ...,
    ) -> Diarization: ...

    @overload
    async def create(
        self,
        *,
        task: Literal["diarize"],
        response_format: Literal["text", "srt", "vtt"],
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        num_speakers: NotGivenOr[int | None] = ...,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = ...,
        diarization_setting: NotGivenOr[str] = ...,
        model: NotGivenOr[str] = ...,
    ) -> str: ...

    @overload
    async def create(
        self,
        *,
        response_format: Literal["verbose_json"],
        task: Literal["transcribe"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> VerboseTranscription: ...

    @overload
    async def create(
        self,
        *,
        response_format: Literal["text", "srt", "vtt"],
        task: Literal["transcribe"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> str: ...

    @overload
    async def create(
        self,
        *,
        task: Literal["transcribe"] = ...,
        response_format: Literal["json"] = ...,
        file: FileTypes | None = ...,
        url: str | None = ...,
        language: NotGivenOr[str | None] = ...,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = ...,
        profanity_filter: NotGivenOr[bool] = ...,
        dictionary: NotGivenOr[str | None] = ...,
        model: NotGivenOr[str] = ...,
    ) -> Transcription: ...

    async def create(
        self,
        *,
        task: Task = "transcribe",
        file: FileTypes | None = None,
        url: str | None = None,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        response_format: NotGivenOr[ResponseFormat] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        dictionary: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        num_speakers: NotGivenOr[int | None] = NOT_GIVEN,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = NOT_GIVEN,
        diarization_setting: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> Any:
        """Transcribe audio and await the result. See Transcriptions.create."""
        form = validate_and_build_form(
            task=task,
            file=file,
            url=url,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            profanity_filter=profanity_filter,
            dictionary=dictionary,
            prompt=prompt,
            json_schema=json_schema,
            num_speakers=num_speakers,
            roles=roles,
            diarization_setting=diarization_setting,
            model=model,
        )
        response = await self._transport.request(
            "POST", "/audio/transcriptions", form=form, file=file
        )
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        return parse_result(response.body)

    # -- jobs -------------------------------------------------------------

    async def create_job(
        self,
        *,
        task: Task = "transcribe",
        file: FileTypes | None = None,
        url: str | None = None,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        response_format: NotGivenOr[ResponseFormat] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[Sequence[Granularity]] = NOT_GIVEN,
        profanity_filter: NotGivenOr[bool] = NOT_GIVEN,
        dictionary: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
        json_schema: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        num_speakers: NotGivenOr[int | None] = NOT_GIVEN,
        roles: NotGivenOr[str | list[str] | dict[str, str] | None] = NOT_GIVEN,
        diarization_setting: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
    ) -> AsyncJob:
        """Submit deferred transcription and return at once. See Transcriptions.create_job."""
        form = validate_and_build_form(
            task=task,
            file=file,
            url=url,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            profanity_filter=profanity_filter,
            dictionary=dictionary,
            prompt=prompt,
            json_schema=json_schema,
            num_speakers=num_speakers,
            roles=roles,
            diarization_setting=diarization_setting,
            model=model,
        )
        response = await self._transport.request(
            "POST", "/audio/transcriptions/async", form=form, file=file
        )
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))

        job = AsyncJob.model_validate(response.body)
        job._bind(self)
        return job

    async def retrieve_job(self, job_id: str) -> AsyncJob:
        """Fetch a job by id. See Transcriptions.retrieve_job."""
        response = await self._transport.request(
            "GET", f"/audio/transcriptions/async/{job_id}"
        )
        if response.status_code >= 400:
            raise error_for_status(response.status_code, _detail(response.body))
        job = AsyncJob.model_validate(response.body)
        job._bind(self)
        return job


def _detail(body: Any) -> str:
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
