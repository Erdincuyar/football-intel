"""ClubElo'dan takim Elo puanlarini indirir.

Elo nedir: Satrancta kullanilan, her mactan sonra guncellenen bir guc puani.
Beklenenden iyi sonuc alan takim puan kazanir, kotu sonuc alan kaybeder. Tek bir
sayiyla takimin o anki gucunu ozetler ve tahmin modellerinde guclu bir baseline'dir.

ClubElo iki CSV adresi sunar (ucretsiz, kayit gerektirmez):
- /YYYY-MM-DD   -> o tarihteki tum takimlarin puani (lig ve ulke bilgisiyle)
- /TakimAdi     -> bir takimin tum puan gecmisi (From-To tarih araliklariyla)

Once her sezonun ortasindaki tabloyu cekip hangi takimlarin hangi ligde oldugunu
ogreniyoruz, sonra bu takimlarin gecmisini tek tek indiriyoruz.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from footintel.config import CURRENT_SEASON, RAW_DIR, SEASONS, season_start_year
from footintel.ingest.http import PoliteSession, SourceUnavailable

log = logging.getLogger(__name__)

BASE_URL = "http://api.clubelo.com"
RAW_SUBDIR = RAW_DIR / "clubelo"


def snapshot_date(season: str) -> date:
    """Sezon ortasi (15 Ocak); guncel sezonda bugun."""
    mid = date(season_start_year(season) + 1, 1, 15)
    return min(mid, date.today())


def _read_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), na_values=["None"])


def snapshot_path(season: str) -> Path:
    return RAW_SUBDIR / "snapshots" / f"{season}.csv"


def history_path(club: str) -> Path:
    return RAW_SUBDIR / "history" / f"{re.sub(r'[^a-zA-Z0-9-]', '', club)}.csv"


def download_snapshots(session: PoliteSession, *, force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for season in SEASONS:
        path = snapshot_path(season)
        if force or season == CURRENT_SEASON or not path.exists():
            response = session.get(f"{BASE_URL}/{snapshot_date(season).isoformat()}")
            if response.status_code >= 500:
                raise SourceUnavailable(f"ClubElo HTTP {response.status_code}: {response.url}")
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        out[season] = _read_csv(path.read_text(encoding="utf-8"))
    return out


def download_histories(session: PoliteSession, clubs: list[str]) -> dict[str, pd.DataFrame]:
    """Takim gecmisleri her calistirmada yeniden indirilir: gecmis her hafta uzar."""
    out = {}
    for club in sorted(set(clubs)):
        url = f"{BASE_URL}/{re.sub(r'[^a-zA-Z0-9-]', '', club)}"
        response = session.get(url)
        if response.status_code >= 500:
            raise SourceUnavailable(f"ClubElo HTTP {response.status_code}: {url}")
        response.raise_for_status()
        df = _read_csv(response.text)
        if df.empty:
            raise ValueError(f"ClubElo'da gecmis bulunamadi: {club!r} ({url})")
        path = history_path(club)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        out[club] = df
        log.info("ClubElo gecmisi: %s (%d satir)", club, len(df))
    return out
