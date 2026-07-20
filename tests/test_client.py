"""Client construction: transport selection and env handling."""

from __future__ import annotations

import pytest

from nexara import AsyncNexara, Nexara
from nexara._http import AsyncHttpxTransport, HttpxTransport
from nexara._mock.transport import AsyncMockTransport, MockTransport


def test_default_transport_is_real(monkeypatch):
    monkeypatch.delenv("NEXARA_USE_MOCK", raising=False)
    nx = Nexara(api_key="k")
    assert isinstance(nx._transport, HttpxTransport)


def test_use_mock_env_selects_mock(monkeypatch):
    monkeypatch.setenv("NEXARA_USE_MOCK", "1")
    nx = Nexara(api_key="k")
    assert isinstance(nx._transport, MockTransport)


def test_async_default_transport_is_real(monkeypatch):
    monkeypatch.delenv("NEXARA_USE_MOCK", raising=False)
    nx = AsyncNexara(api_key="k")
    assert isinstance(nx._transport, AsyncHttpxTransport)


def test_async_use_mock_env_selects_mock(monkeypatch):
    monkeypatch.setenv("NEXARA_USE_MOCK", "1")
    nx = AsyncNexara(api_key="k")
    assert isinstance(nx._transport, AsyncMockTransport)


def test_base_url_precedence(monkeypatch):
    monkeypatch.setenv("NEXARA_BASE_URL", "http://localhost:8000/v1")
    assert Nexara(api_key="k").base_url == "http://localhost:8000/v1"
    # Explicit arg wins over the env var.
    assert Nexara(api_key="k", base_url="http://other/v1").base_url == "http://other/v1"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("NEXARA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        Nexara()


def test_realtime_unavailable_without_mock(monkeypatch):
    """The streaming protocol is not public yet; a real client must get a clear
    error, never fabricated mock transcripts."""
    monkeypatch.delenv("NEXARA_USE_MOCK", raising=False)
    with pytest.raises(NotImplementedError):
        Nexara(api_key="k").realtime.connect()


def test_realtime_mock_opt_in(monkeypatch):
    monkeypatch.setenv("NEXARA_USE_MOCK", "1")
    session = Nexara(api_key="k").realtime.connect()
    assert session is not None
