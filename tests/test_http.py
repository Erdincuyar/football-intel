"""Ortak HTTP oturumu: ulasilamayan kaynak hatti durdurmaz (SourceUnavailable)."""

from __future__ import annotations

import pytest
import requests

from footintel.ingest import http as H


def test_connection_timeout_becomes_source_unavailable(monkeypatch):
    calls = []

    def boom(url, **kw):
        calls.append(url)
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    s = H.PoliteSession(delay=0)
    monkeypatch.setattr(s.session, "get", boom)
    with pytest.raises(H.SourceUnavailable):
        s.get("http://api.clubelo.com/2022-01-15", retries=2)
    assert len(calls) == 3                          # ilk deneme + 2 tekrar


def test_recovers_after_one_timeout(monkeypatch):
    ok = requests.Response()
    ok.status_code = 200
    seq = [requests.exceptions.ReadTimeout("slow"), ok]

    def flaky(url, **kw):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    s = H.PoliteSession(delay=0)
    monkeypatch.setattr(s.session, "get", flaky)
    assert s.get("http://x").status_code == 200
