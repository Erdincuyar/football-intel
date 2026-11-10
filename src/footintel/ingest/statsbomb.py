"""StatsBomb acik verisi (github.com/statsbomb/open-data): olay (event) verisi.

Neden StatsBomb? Takim taktigini (pres yuksekligi, topa sahip olma, sahayi
egme) olcmek icin her pasin, presin, sutun SAHADAKI KONUMU gerekir. Bu "olay
verisi" profesyonel kulupler icin ucretli bir urundur; StatsBomb bazi sezonlari
arastirma ve egitim icin ucretsiz yayinlar. 2015/16 sezonu bizim 4 ligimiz icin
(PL, La Liga, Serie A, Ligue 1) TAM sezon olarak acik. Super Lig yok.

Lisans: StatsBomb acik verisi kullanilirken kaynak gosterilmelidir (README'de
logo + atif). Ticari kullanim icin degil; bu proje portfoy/egitim amacli.

Ham veri kurali icin bir istisna: mac basina ~3 MB JSON, 1520 mac ~4.6 GB
ediyor. Ham JSON'u saklamak yerine her maci indirir indirmez, tum olaylari
(yalnizca "Ball Receipt" haric) secili alanlarla tek bir Parquet'e yaziyoruz.
Kaynak sabit oldugu (2015/16 artik degismiyor) ve istenince yeniden
indirilebildigi icin bu bilgi kaybi kabul edilebilir.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import truststore

from footintel.config import RAW_DIR

truststore.inject_into_ssl()
log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_SUBDIR = RAW_DIR / "statsbomb"
SEASON_ID = 27            # StatsBomb'un "2015/2016" sezon kimligi
SEASON = "1516"           # projedeki sezon kodu bicimi
# Bizim lig kodu -> StatsBomb competition_id
COMPETITIONS: dict[str, int] = {"E0": 2, "SP1": 11, "I1": 12, "F1": 7}

# Taktik metrikler icin gereken olay turleri. "Ball Receipt*" (pasi alma) mac basina
# ~1000 satir ve bir pasin "recipient" alanindan zaten cikarilabiliyor; atliyoruz.
SKIP_TYPES = {"Ball Receipt*"}


def _get_json(url: str, session: requests.Session, attempts: int = 6):
    """GET + JSON. 429 (cok fazla istek) ve 5xx'te bekleyip yeniden dener.

    429, GitHub'in "yavasla" demesidir: Retry-After basligi varsa ona uyariz, yoksa
    5, 10, 20... sn bekleriz. Bu, kaynaga nazik davranmanin teknik karsiligi.
    """
    for i in range(attempts):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 5 * 2 ** i))
                log.warning("%s -> HTTP %d, %.0f sn bekleniyor", url, r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if i == attempts - 1:
                raise
            time.sleep(5 * 2 ** i)
    raise RuntimeError(f"{url}: {attempts} denemede alinamadi")


def _name(d: dict | None) -> str | None:
    return d.get("name") if isinstance(d, dict) else None


def slim_events(events: list[dict], match_id: int) -> pd.DataFrame:
    """Ic ice JSON olaylarini duz bir tabloya cevirir (bir satir = bir olay).

    Koordinatlar: saha 120 x 80 (yard). StatsBomb her olayi, olayi YAPAN takimin
    soldan saga hucum ettigi yone gore kaydeder. Yani x=120 her zaman olayi yapan
    takimin hucum ettigi kaledir. Savunma olaylari icin de ayni kural gecerli:
    savunan takimin x=100'deki bir mudahalesi, rakip yari sahada (onde) yapilmistir.
    """
    rows = []
    for e in events:
        t = e["type"]["name"]
        if t in SKIP_TYPES:
            continue
        loc = e.get("location") or [None, None]
        p, c, s, d = e.get("pass", {}), e.get("carry", {}), e.get("shot", {}), e.get("duel", {})
        end = p.get("end_location") or c.get("end_location") or s.get("end_location") or []
        rows.append({
            "match_id": match_id, "index": e["index"], "period": e["period"],
            "minute": e["minute"], "second": e["second"], "type": t,
            "team": _name(e.get("team")), "player": _name(e.get("player")),
            "position": _name(e.get("position")),
            "possession": e.get("possession"),
            "possession_team": _name(e.get("possession_team")),
            "play_pattern": _name(e.get("play_pattern")),
            "x": loc[0], "y": loc[1],
            "end_x": end[0] if len(end) > 0 else None,
            "end_y": end[1] if len(end) > 1 else None,
            "under_pressure": bool(e.get("under_pressure", False)),
            "counterpress": bool(e.get("counterpress", False)),
            "pass_length": p.get("length"), "pass_height": _name(p.get("height")),
            "pass_type": _name(p.get("type")), "pass_outcome": _name(p.get("outcome")),
            "pass_cross": bool(p.get("cross", False)),
            "pass_recipient": _name(p.get("recipient")),
            "shot_xg": s.get("statsbomb_xg"), "shot_outcome": _name(s.get("outcome")),
            "shot_type": _name(s.get("type")),
            "duel_type": _name(d.get("type")), "duel_outcome": _name(d.get("outcome")),
            "formation": (e.get("tactics") or {}).get("formation"),
        })
    return pd.DataFrame(rows)


def matches(league: str, session: requests.Session) -> pd.DataFrame:
    data = _get_json(f"{BASE_URL}/matches/{COMPETITIONS[league]}/{SEASON_ID}.json", session)
    return pd.DataFrame([{
        "match_id": m["match_id"], "league": league, "season": SEASON,
        "match_date": pd.Timestamp(m["match_date"]).date(), "match_week": m.get("match_week"),
        "home_team": m["home_team"]["home_team_name"],
        "away_team": m["away_team"]["away_team_name"],
        "home_score": m["home_score"], "away_score": m["away_score"],
    } for m in data]).sort_values(["match_date", "match_id"]).reset_index(drop=True)


def paths(league: str) -> tuple[Path, Path]:
    return (RAW_SUBDIR / f"matches_{league}_{SEASON}.parquet",
            RAW_SUBDIR / f"events_{league}_{SEASON}.parquet")


def download_league(league: str, *, workers: int = 4, force: bool = False) -> tuple[Path, Path]:
    """Bir ligin 2015/16 maclarini ve olaylarini indirir (sezon bitti: bir kez yeterli)."""
    mpath, epath = paths(league)
    if mpath.exists() and epath.exists() and not force:
        log.info("Atlandi (zaten var): %s", epath.name)
        return mpath, epath
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "footintel-portfolio-project/0.1 (personal research)"
    m = matches(league, session)

    def one(mid: int) -> pd.DataFrame:
        return slim_events(_get_json(f"{BASE_URL}/events/{mid}.json", session), mid)

    # Birkac paralel istek indirmeyi hizlandirir; daha fazlasi GitHub'dan 429
    # (Too Many Requests) getiriyor (16 is parcacigiyla denedik).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(one, m["match_id"]))
    ev = pd.concat(parts, ignore_index=True)
    RAW_SUBDIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(mpath, index=False)
    ev.to_parquet(epath, index=False)
    log.info("%s: %d mac, %d olay", league, len(m), len(ev))
    return mpath, epath


def load(league: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    leagues = [league] if league else list(COMPETITIONS)
    ms, es = [], []
    for lg in leagues:
        mpath, epath = paths(lg)
        ms.append(pd.read_parquet(mpath))
        es.append(pd.read_parquet(epath).assign(league=lg))
    return pd.concat(ms, ignore_index=True), pd.concat(es, ignore_index=True)
