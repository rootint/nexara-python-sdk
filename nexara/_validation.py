"""Client-side validation.

Everything here fires before the request is sent. Two kinds of rules live
together:

  * things the server rejects with a 400 — we just fail faster and cheaper;
  * things the server accepts and then silently does differently — those are the
    reason this module exists. A request that is charged for and quietly does
    the wrong thing is worse than one that fails.

Where the server is silently permissive, we are deliberately stricter than it
is. That is a real trade-off: code that worked against raw curl can raise here.
Each such rule says why in its error message.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import IO, Any, Literal, Sequence

from ._exceptions import NexaraValidationError
from ._sentinel import NOT_GIVEN, NotGivenOr, given, resolve

Task = Literal["transcribe", "diarize"]
ResponseFormat = Literal["json", "verbose_json", "text", "srt", "vtt"]
Granularity = Literal["word", "segment", "sentence"]

TASKS: frozenset[str] = frozenset({"transcribe", "diarize"})
RESPONSE_FORMATS: frozenset[str] = frozenset({"json", "verbose_json", "text", "srt", "vtt"})
GRANULARITIES: frozenset[str] = frozenset({"word", "segment", "sentence"})
DICTIONARIES: frozenset[str] = frozenset({"medical"})
DIARIZATION_SETTINGS: frozenset[str] = frozenset({"general", "telephonic"})
AUTO_LANGUAGE: frozenset[str] = frozenset({"auto", "automatic", ""})

# Limits the server enforces on `roles` (role_tagging), published at
# docs.nexara.ru/speakers-roles. Documented numbers, so we check them here and
# fail faster and cheaper than a 400 — unlike the language list, which is not
# published and therefore not mirrored.
MAX_ROLES = 10
MAX_ROLE_NAME_LEN = 64
MAX_ROLE_DESC_LEN = 500

_DIARIZE_ONLY = ("num_speakers", "roles", "diarization_setting")

# The server checks `language` against a fixed list of ~100 ISO-639-1 codes in
# utils/language.py. We deliberately do NOT mirror that list: we do not have it,
# and a guessed copy that is missing a code would reject a language that
# actually works — a worse failure than the server's own 400. Format check only.
_LANG_RE = re.compile(r"^[a-z]{2}$")


def _fail(msg: str) -> None:
    raise NexaraValidationError(msg)


def validate_and_build_form(
    *,
    task: Task = "transcribe",
    file: str | Path | bytes | IO[bytes] | None = None,
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
) -> dict[str, Any]:
    """Validate, then return the multipart form fields (minus `file` itself).

    Validation and serialization live in one function on purpose: they read the
    same set of resolved values, and splitting them would mean deciding twice
    what "the user asked for X" means.
    """
    if task not in TASKS:
        _fail(f"task must be one of {sorted(TASKS)}, got {task!r}")

    # --- audio source ---------------------------------------------------
    if file is not None and url is not None:
        _fail("pass either file= or url=, not both — the server accepts exactly one")
    if file is None and url is None:
        _fail("pass either file= or url=")

    # --- diarize-only parameters ---------------------------------------
    if task != "diarize":
        for name, value in zip(_DIARIZE_ONLY, (num_speakers, roles, diarization_setting)):
            if given(value) and value is not None:
                _fail(
                    f"{name}= requires task='diarize'. "
                    f"With task='transcribe' the server "
                    + (
                        "rejects it with a 400"
                        if name == "roles"
                        else "accepts it and silently ignores it"
                    )
                    + "."
                )

    # --- granularity ----------------------------------------------------
    gran = list(resolve(timestamp_granularities, ["segment"]))
    for g in gran:
        if g not in GRANULARITIES:
            _fail(f"timestamp_granularities values must be in {sorted(GRANULARITIES)}, got {g!r}")
    if len(gran) != 1:
        _fail(
            f"timestamp_granularities takes exactly one value, got {gran!r}. "
            "The field is a list for OpenAI compatibility, but the server reads "
            "only the first element and silently ignores the rest."
        )
    if task == "diarize" and gran == ["sentence"]:
        _fail("timestamp_granularities=['sentence'] is not supported with task='diarize'")

    # --- response_format ------------------------------------------------
    fmt = resolve(response_format, "json")
    if fmt not in RESPONSE_FORMATS:
        _fail(f"response_format must be one of {sorted(RESPONSE_FORMATS)}, got {fmt!r}")

    # --- prompt / json_schema -------------------------------------------
    has_prompt = given(prompt) and prompt is not None
    has_schema = given(json_schema) and json_schema is not None

    if has_schema and not has_prompt:
        _fail(
            "json_schema= requires prompt=. Without a prompt the server ignores "
            "the schema entirely — no error, no structured output, and the "
            "request is still billed."
        )

    if has_prompt:
        # The server force-sets both of these when a prompt is present. Rather
        # than let it silently overwrite what was asked for, refuse. Only an
        # *explicit* value conflicts: the defaults must stay compatible or
        # prompt= could never be used at all.
        if given(response_format) and fmt != "verbose_json":
            _fail(
                f"prompt= cannot be combined with response_format={fmt!r}: the "
                "server forces verbose_json whenever a prompt is set, so you "
                f"would be charged for a request and not get {fmt}."
            )
        if given(timestamp_granularities) and gran != ["segment"]:
            _fail(
                f"prompt= cannot be combined with timestamp_granularities={gran!r}: "
                "the server forces ['segment'] whenever a prompt is set."
            )
        fmt = "verbose_json"
        gran = ["segment"]

    schema_str: str | None = None
    if has_schema:
        schema_str = _validate_json_schema(json_schema)

    # --- misc ------------------------------------------------------------
    dic = resolve(dictionary, None)
    if dic is not None and dic not in DICTIONARIES:
        _fail(f"dictionary must be one of {sorted(DICTIONARIES)} or None, got {dic!r}")

    # Only two presets are documented (docs.nexara.ru/diarization); an unknown
    # value is a 400. The default "general" is in the set, so checking the
    # resolved value is harmless on the transcribe path.
    diar_setting = resolve(diarization_setting, "general")
    if diar_setting not in DIARIZATION_SETTINGS:
        _fail(
            f"diarization_setting must be one of {sorted(DIARIZATION_SETTINGS)}, "
            f"got {diar_setting!r}"
        )

    lang = resolve(language, None)
    if lang is not None and lang not in AUTO_LANGUAGE and not _LANG_RE.match(lang):
        _fail(
            f"language must be a lowercase ISO-639-1 code (e.g. 'ru', 'en'), "
            f"'auto', or None for auto-detection — got {lang!r}"
        )

    n_speakers = resolve(num_speakers, None)
    if n_speakers is not None and n_speakers < 1:
        _fail(f"num_speakers must be >= 1, got {n_speakers}")

    roles_form = _validate_roles(roles) if given(roles) and roles is not None else None

    # --- build form -------------------------------------------------------
    form: dict[str, Any] = {
        "task": task,
        "response_format": fmt,
        # The alias carries the brackets: a field named without them is not seen
        # by the server at all.
        "timestamp_granularities[]": gran,
        "profanity_filter": resolve(profanity_filter, False),
        "model": resolve(model, "whisper-1"),
    }
    if url is not None:
        form["url"] = url
    if lang is not None:
        form["language"] = lang
    if dic is not None:
        form["dictionary"] = dic
    if has_prompt:
        form["prompt"] = prompt
    if schema_str is not None:
        form["json_schema"] = schema_str
    if task == "diarize":
        form["diarization_setting"] = diar_setting
        if n_speakers is not None:
            form["num_speakers"] = n_speakers
        if roles_form is not None:
            form["roles"] = roles_form

    return form


def _validate_roles(roles: str | list[str] | dict[str, str] | Any) -> str:
    """Validate `roles` (which turns on role_tagging) and return the wire string.

    Three modes, mirroring parse_roles_param server-side:
      * "auto"           — the LLM invents short labels in the dialogue's language;
      * ["client", ...]  — roles are limited to this set (+ "unknown");
      * {"client": "…"}  — the same, with descriptions handed to the LLM.

    list/dict are serialized to JSON here — the server field is a single string —
    exactly as json_schema is. A plain str (e.g. "auto", or a pre-built JSON
    string) is passed through untouched.

    The role-count and name/description length limits below are the ones the
    server documents (docs.nexara.ru/speakers-roles): a 400 otherwise, so we
    catch it client-side.
    """
    if isinstance(roles, str):
        if not roles.strip():
            _fail("roles= must be a non-empty string (e.g. 'auto'), a list, or a dict")
        return roles

    if isinstance(roles, list):
        if not roles:
            _fail("roles=[] is empty; pass 'auto', a non-empty list, or a dict")
        if len(roles) > MAX_ROLES:
            _fail(f"roles takes at most {MAX_ROLES} roles, got {len(roles)}")
        for item in roles:
            if not isinstance(item, str) or not item.strip():
                _fail(f"roles list entries must be non-empty strings, got {item!r}")
            if len(item) > MAX_ROLE_NAME_LEN:
                _fail(f"role name exceeds {MAX_ROLE_NAME_LEN} chars: {item!r}")
        if len(set(roles)) != len(roles):
            _fail(f"roles must not contain duplicates, got {roles!r}")
        return json.dumps(roles, ensure_ascii=False)

    if isinstance(roles, dict):
        if not roles:
            _fail("roles={} is empty; pass 'auto', a list, or a non-empty dict")
        if len(roles) > MAX_ROLES:
            _fail(f"roles takes at most {MAX_ROLES} roles, got {len(roles)}")
        for key, value in roles.items():
            if not isinstance(key, str) or not key.strip():
                _fail(f"roles dict keys must be non-empty strings, got {key!r}")
            if len(key) > MAX_ROLE_NAME_LEN:
                _fail(f"role name exceeds {MAX_ROLE_NAME_LEN} chars: {key!r}")
            if not isinstance(value, str) or not value.strip():
                _fail(f"roles[{key!r}] must be a non-empty string description, got {value!r}")
            if len(value) > MAX_ROLE_DESC_LEN:
                _fail(f"role description for {key!r} exceeds {MAX_ROLE_DESC_LEN} chars")
        return json.dumps(roles, ensure_ascii=False)

    _fail(f"roles must be a str, list[str], or dict[str, str], got {type(roles).__name__}")
    raise AssertionError("unreachable")  # _fail always raises; satisfies the type checker


def _validate_json_schema(schema: str | dict[str, Any] | Any) -> str:
    """Check the schema here so a bad one costs nothing.

    On `develop` the server validates this and returns 400. On the
    `steaming_billing` branch it does not — the schema reaches Fireworks and
    fails only *after* a paid transcription. Validating client-side makes the
    SDK behave the same either way.
    """
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise NexaraValidationError(f"json_schema is not valid JSON: {exc}") from exc
    elif isinstance(schema, dict):
        parsed = schema
    else:
        _fail(f"json_schema must be a dict or a JSON string, got {type(schema).__name__}")

    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover
        return json.dumps(parsed, ensure_ascii=False)

    try:
        Draft202012Validator.check_schema(parsed)
    except Exception as exc:
        raise NexaraValidationError(f"json_schema is not a valid JSON Schema Draft 2020-12: {exc}") from exc

    return json.dumps(parsed, ensure_ascii=False)
