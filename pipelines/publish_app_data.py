"""Faz 8: uygulamanin okudugu tablolari Neon PostgreSQL'e yayinlar.

Baglanti adresi FOOTINTEL_DATABASE_URL ortam degiskeninden gelir (GitHub'da repo secret).
Tanimli degilse adim sessizce atlanir; boylece secret eklenmeden once CI kirmizi olmaz.

Kullanim:
    FOOTINTEL_DATABASE_URL=postgresql://... uv run python pipelines/publish_app_data.py
"""

from __future__ import annotations

import logging

from footintel import store

log = logging.getLogger("publish_app_data")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    url = store.database_url()
    if not url:
        log.info("%s tanimli degil; yayinlama atlandi.", store.ENV_VAR)
        return
    out = store.publish(url)
    for name, n in out["tables"].items():
        log.info("tablo %-28s %7d satir", name, n)
    log.info("json: %s", ", ".join(out["json"]))


if __name__ == "__main__":
    main()
