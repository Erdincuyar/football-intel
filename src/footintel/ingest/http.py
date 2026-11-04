"""Tum kaynaklar icin ortak, nazik HTTP oturumu."""

from __future__ import annotations

import time

import requests
import truststore

# Bkz. ingest/football_data.py: isletim sisteminin sertifika deposunu kullan.
truststore.inject_into_ssl()

USER_AGENT = "footintel-portfolio-project/0.1 (personal research)"


class PoliteSession:
    """Istekler arasinda en az `delay` saniye bekleyen requests oturumu."""

    def __init__(self, delay: float = 1.5, headers: dict[str, str] | None = None) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, **(headers or {})})
        self._last = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        response = self.session.get(url, timeout=30, **kwargs)
        self._last = time.monotonic()
        return response
