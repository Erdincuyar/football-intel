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

    def get(self, url: str, retries: int = 3, **kwargs) -> requests.Response:
        """5xx (sunucu tarafi) hatalarinda artan beklemeyle (5, 10, 20 sn) yeniden dener.

        4xx hatalari yeniden denenmez: onlar bizim istegimizin yanlis oldugunu gosterir.
        """
        for attempt in range(retries + 1):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            response = self.session.get(url, timeout=30, **kwargs)
            self._last = time.monotonic()
            if response.status_code < 500 or attempt == retries:
                return response
            time.sleep(5 * 2**attempt)
        return response


class SourceUnavailable(RuntimeError):
    """Kaynak su an veri vermiyor (bakim sayfasi, 5xx hatasi, engel vb.)."""
