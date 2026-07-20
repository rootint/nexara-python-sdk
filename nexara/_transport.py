"""The transport seam.

Everything above this line (resources, validation, models) is the real SDK.
Below it sit the production httpx transports (`_http.py`) and the in-memory
mock (`_mock/`). The point of the seam is that swapping them changes no
public API — which also means validation runs for real even on the mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol, Union


@dataclass
class Response:
    status_code: int
    body: Any
    """Parsed JSON for json-ish formats, a str for text/srt/vtt."""


FileInput = Union[str, Path, bytes, IO[bytes]]
"""What `file=` may be by the time it reaches a transport. Paths (str/Path) are
opened by the transport itself, so uploads stream from disk instead of being
loaded into memory first."""


class Transport(Protocol):
    """The only thing that touches the network — or pretends to."""

    def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response: ...


class AsyncTransport(Protocol):
    """The async twin of Transport: same call, awaited."""

    async def request(
        self,
        method: str,
        path: str,
        *,
        form: dict[str, Any] | None = None,
        file: FileInput | None = None,
    ) -> Response: ...
