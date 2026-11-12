"""openfootball/football.json: sezonun tum fiksturu (acik veri, ucretsiz, GitHub'da).

Neden? football-data.co.uk'in fikstur listesi yalnizca birkac gun sonrasini ve oranlarla
birlikte verir; milli mac arasinda listede ust lig maci olmaz. openfootball sezonun 38
haftasini bastan yayinlar. Biz onu yalnizca "hangi mac, hangi gun" icin kullaniyoruz;
skor ve istatistik kaynagimiz football-data.co.uk olarak kaliyor.

Super Lig openfootball'da yok; onun icin fixturedownload.com'un ucretsiz JSON beslemesi
kullanilir (download_fixturedownload) ve ayni bicime cevrilir.
"""

from __future__ import annotations

import logging

import pandas as pd

from footintel.config import CURRENT_SEASON
from footintel.ingest.football_data import _get

log = logging.getLogger(__name__)
BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
CODES = {"E0": "en.1", "SP1": "es.1", "D1": "de.1", "I1": "it.1", "F1": "fr.1"}
FD_URL = "https://fixturedownload.com/feed/json/{slug}-{year}"
FD_SLUGS = {"T1": "super-lig"}


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


def from_fixturedownload(rows: list[dict]) -> list[dict]:
    """fixturedownload bicimini openfootball bicimine cevirir (round, date, time, team1/2)."""
    out = []
    for r in rows:
        ts = pd.Timestamp(r["DateUtc"].replace("Z", ""))
        m = {"round": f"Hafta {r.get('RoundNumber')}", "date": ts.date().isoformat(),
             "time": ts.strftime("%H:%M") + " UTC", "team1": r["HomeTeam"],
             "team2": r["AwayTeam"]}
        if r.get("HomeTeamScore") is not None and r.get("AwayTeamScore") is not None:
            m["score"] = {"ft": [r["HomeTeamScore"], r["AwayTeamScore"]]}
        out.append(m)
    return out


def download_any(league: str, season: str = CURRENT_SEASON) -> list[dict] | None:
    """Ligin kaynagina gore fikstur: openfootball, yoksa fixturedownload."""
    if league in CODES:
        return download(league, season)
    slug = FD_SLUGS.get(league)
    if slug is None:
        return None
    url = FD_URL.format(slug=slug, year=2000 + int(season[:2]))
    try:
        r = _get(url)
    except Exception as exc:
        log.warning("%s indirilemedi: %s", url, exc)
        return None
    if r.status_code != 200:
        log.warning("%s: HTTP %s", url, r.status_code)
        return None
    return from_fixturedownload(r.json())
