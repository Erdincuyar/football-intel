"""football-data.co.uk'tan sezon CSV dosyalarini indirir.

Kural: ham veri oldugu gibi saklanir, hicbir degisiklik yapilmaz. Temizleme
ayri bir adimdir. Boylece temizleme kodunda hata yaparsak indirmeyi tekrarlamadan
duzeltebiliriz ve kaynagin bize tam olarak ne verdigini her zaman gorebiliriz.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from footintel.config import CURRENT_SEASON, LEAGUES, RAW_DIR, SEASONS

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_SUBDIR = RAW_DIR / "football_data"

# Siteye nazik davranalim: istekler arasinda bekleme ve acik bir User-Agent.
REQUEST_DELAY_SECONDS = 2.0
HEADERS = {"User-Agent": "footintel-portfolio-project/0.1 (personal research)"}


def raw_path(league: str, season: str) -> Path:
    return RAW_SUBDIR / f"{league}_{season}.csv"


def download_season(league: str, season: str, *, force: bool = False) -> Path | None:
    """Bir lig-sezon dosyasini indirir ve yolunu dondurur.

    Tamamlanmis sezonlar bir kez indirilir (dosya degismez). Guncel sezon her
    calistirmada yeniden indirilir, cunku site haftada iki kez yeni maclari ekler.
    """
    path = raw_path(league, season)
    is_current = season == CURRENT_SEASON
    if path.exists() and not (force or is_current):
        log.info("Atlandi (zaten var): %s", path.name)
        return path

    url = BASE_URL.format(season=season, league=league)
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code == 404:
        log.warning("Kaynakta yok: %s", url)
        return None
    response.raise_for_status()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)  # bytes olarak yaz: kodlamayi temizleme adimi cozer
    log.info("Indirildi: %s (%d bayt)", path.name, len(response.content))
    return path


def download_all(*, force: bool = False) -> list[Path]:
    paths: list[Path] = []
    for league in LEAGUES:
        for season in SEASONS:
            existed = raw_path(league, season).exists()
            path = download_season(league, season, force=force)
            if path is not None:
                paths.append(path)
            if not existed or season == CURRENT_SEASON or force:
                time.sleep(REQUEST_DELAY_SECONDS)
    return paths
