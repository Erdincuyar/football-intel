"""Faz 11: FotMob'dan xG, kadro, hoca ve eksik oyuncu verisini artimli olarak toplar.

Biten maclar data/reference/fotmob/ altina bir kez yazilir (git'e girer, yeniden
indirilmez). Yaklasan maclarin eksik oyunculari her calismada yenilenir:
data/clean/fotmob_upcoming.parquet.

Istek butcesi: resmi olmayan bir kaynagi yormamak icin calisma basina en fazla
--budget istek atilir; geri kalan maclar sonraki calismalarda tamamlanir.

Kullanim:
    uv run python pipelines/ingest_fotmob.py                     # bu sezon + yaklasanlar
    uv run python pipelines/ingest_fotmob.py --seasons 2223 2324 --budget 1500
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from footintel.config import CLEAN_DIR, CURRENT_SEASON, REFERENCE_DIR
from footintel.ingest import fotmob as FM
from footintel.ingest.http import SourceUnavailable

log = logging.getLogger("ingest_fotmob")
OUT = REFERENCE_DIR / "fotmob"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=[CURRENT_SEASON])
    ap.add_argument("--budget", type=int, default=400)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    s = FM.session()
    left = args.budget
    upcoming = []
    for season in args.seasons:
        for league in FM.LEAGUES:
            if left <= 1:
                log.info("Istek butcesi bitti; kalanlar sonraki calismada")
                break
            try:
                used, up = FM.update_season(s, league, season, OUT, left)
            except SourceUnavailable as exc:
                log.warning("FotMob %s %s atlandi: %s", league, season, exc)
                continue
            left -= used
            if season == CURRENT_SEASON and not up.empty:
                upcoming.append(up.assign(league=league))
    if upcoming:
        pd.concat(upcoming, ignore_index=True).to_parquet(
            CLEAN_DIR / "fotmob_upcoming.parquet", index=False)
    log.info("FotMob: %d istek kullanildi", args.budget - left)


if __name__ == "__main__":
    main()
