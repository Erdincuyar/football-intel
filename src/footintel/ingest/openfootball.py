"""openfootball/football.json: sezonun tum fiksturu (acik veri, ucretsiz, GitHub'da).

Neden? football-data.co.uk'in fikstur listesi yalnizca birkac gun sonrasini ve oranlarla
birlikte verir; milli mac arasinda listede ust lig maci olmaz. openfootball sezonun 38
haftasini bastan yayinlar. Biz onu yalnizca "hangi mac, hangi gun" icin kullaniyoruz;
skor ve istatistik kaynagimiz football-data.co.uk olarak kaliyor.

Super Lig openfootball'da yok: Super Lig icin on tahmin uretilemez (N/A), resmi tahmin
football-data fiksturu yayinlaninca yine gelir.
"""

from __future__ import annotations

import logging

from footintel.config import CURRENT_SEASON
from footintel.ingest.football_data import _get

log = logging.getLogger(__name__)
BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
CODES = {"E0": "en.1", "SP1": "es.1", "D1": "de.1", "I1": "it.1", "F1": "fr.1"}


def season_dir(season: str = CURRENT_SEASON) -> str:
    """'2627' -> '2026-27'"""
    return f"20{season[:2]}-{season[2:]}"


def download(league: str, season: str = CURRENT_SEASON) -> list[dict] | None:
    code = CODES.get(league)
    if code is None:
        return None
    url = BASE_URL.format(season=season_dir(season), code=code)
    try:
        r = _get(url)
    except Exception as exc:  # ag hatasi on tahmini durdurmasin
        log.warning("%s indirilemedi: %s", url, exc)
        return None
    if r.status_code != 200:
        log.warning("%s: HTTP %s", url, r.status_code)
        return None
    return r.json().get("matches", [])
