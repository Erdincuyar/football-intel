"""FotMob: xG, ilk 11 ve piyasa degeri, teknik direktor, sakat/cezali oyuncular.

FotMob'un resmi bir API'si yok; sitenin kendi sayfalari veriyi `/api/data/...` adreslerinden
JSON olarak ceker, biz de ayni adresleri kullaniyoruz. Bu yuzden:
- format haber verilmeden degisebilir: her alan `_get` ile okunur, yoksa NaN yazilir,
  eksik alan yuzunden tum hat durmaz;
- istekler yavas atilir (PoliteSession) ve biten maclar bir kez indirilip saklanir,
  ayni mac ikinci kez istenmez.

Iki istek turu var:
- lig fiksturu: `leagues?id=<lig>&season=2025/2026` -> sezonun tum maclari (id, takimlar,
  tarih, skor). Tek istek.
- mac detayi: `matchDetails?matchId=<id>` -> bitmis macta xG ve diger takim istatistikleri,
  ilk 11 ve yedekler (piyasa degeriyle), teknik direktor, o mac icin eksik oyuncular.
  Oynanmamis macta yalnizca eksik oyuncu listesi (sakat / cezali, donus tahmini) gelir.

Sizinti notu: bitmis bir macin ilk 11'i ve xG'si mactan SONRA bilinir. Bunlar yalnizca
sonraki maclarin ozelliklerinde (gecmis ortalamalar) kullanilmalidir; ayni macin tahmininde
kullanilamaz. Bu kurali features/ katmani uygular.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from footintel.ingest.http import PoliteSession, SourceUnavailable

log = logging.getLogger(__name__)

BASE_URL = "https://www.fotmob.com/api/data"
# Bizim lig kodumuz -> FotMob lig id'si
LEAGUES = {"E0": 47, "SP1": 87, "D1": 54, "I1": 55, "F1": 53, "T1": 71}
# Mac istatistiklerinden saklananlar (FotMob anahtari -> bizim kolon adimiz)
STAT_KEYS = {
    "expected_goals": "xg", "expected_goals_non_penalty": "npxg",
    "expected_goals_open_play": "xg_open", "expected_goals_set_play": "xg_set",
    "BallPossesion": "possession", "total_shots": "shots", "ShotsOnTarget": "sot",
    "big_chance": "big_chances", "touches_opp_box": "box_touches",
    "own_half_passes": "own_half_passes", "opposition_half_passes": "opp_half_passes",
    "matchstats.headers.tackles": "tackles", "interceptions": "interceptions",
}
ROLES = {"starters": "starter", "subs": "sub", "unavailable": "unavailable"}
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 "
                                 "Safari/537.36", "Accept-Language": "en"}


def fm_season(season: str) -> str:
    """'2526' -> '2025/2026'."""
    return f"20{season[:2]}/20{season[2:]}"


def session(delay: float = 1.2) -> PoliteSession:
    return PoliteSession(delay=delay, headers=BROWSER_HEADERS)


def _json(s: PoliteSession, path: str) -> dict:
    r = s.get(f"{BASE_URL}/{path}")
    if not r.ok or not r.content:
        raise SourceUnavailable(f"FotMob {path}: HTTP {r.status_code}")
    return r.json()


def _get(o: Any, *keys: Any, default: Any = None) -> Any:
    """Ic ice sozluk/listeden guvenli okuma: herhangi bir adim yoksa default."""
    for k in keys:
        try:
            o = o[k]
        except (KeyError, IndexError, TypeError):
            return default
        if o is None:
            return default
    return o


def _num(v: Any) -> float:
    """'438 (87%)' -> 438.0, '3.55' -> 3.55, None -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, int | float):
        return float(v)
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


# --- Lig fiksturu ------------------------------------------------------------------------

def fetch_league(s: PoliteSession, league: str, season: str) -> dict:
    return _json(s, f"leagues?id={LEAGUES[league]}&season={fm_season(season)}")


def parse_fixtures(data: dict, league: str, season: str) -> pd.DataFrame:
    rows = []
    for m in _get(data, "fixtures", "allMatches", default=[]):
        st = m.get("status") or {}
        score = str(st.get("scoreStr") or "")
        hg, ag = (score.split(" - ") + [None, None])[:2] if " - " in score else (None, None)
        rows.append({
            "fm_id": int(m["id"]), "league": league, "season": season,
            "kickoff": pd.to_datetime(st.get("utcTime"), utc=True).tz_localize(None),
            "home_fm_id": int(_get(m, "home", "id")), "away_fm_id": int(_get(m, "away", "id")),
            "home_name": _get(m, "home", "name"), "away_name": _get(m, "away", "name"),
            "finished": bool(st.get("finished")), "cancelled": bool(st.get("cancelled")),
            "fm_home_goals": _num(hg), "fm_away_goals": _num(ag),
        })
    return pd.DataFrame(rows)


# --- Mac detayi --------------------------------------------------------------------------

def fetch_match(s: PoliteSession, fm_id: int) -> dict:
    return _json(s, f"matchDetails?matchId={fm_id}")


def _team_stats(data: dict) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for group in _get(data, "content", "stats", "Periods", "All", "stats", default=[]):
        for st in group.get("stats") or []:
            col = STAT_KEYS.get(st.get("key"))
            vals = st.get("stats")
            if col and isinstance(vals, list) and len(vals) == 2 and col not in out:
                out[col] = [_num(vals[0]), _num(vals[1])]
    return out


def parse_match(data: dict) -> tuple[dict, list[dict]]:
    """Mac satiri (takim istatistikleri, hocalar, ilk 11 degeri) ve oyuncu satirlari."""
    fm_id = int(_get(data, "general", "matchId"))
    lu = _get(data, "content", "lineup", default={})
    row: dict[str, Any] = {"fm_id": fm_id,
                           "finished": bool(_get(data, "general", "finished", default=False)),
                           "lineup_type": lu.get("lineupType")}
    stats = _team_stats(data)
    players: list[dict] = []
    for i, side in enumerate(("home", "away")):
        t = lu.get(f"{side}Team") or {}
        for col in STAT_KEYS.values():
            row[f"{side}_{col}"] = stats.get(col, [np.nan, np.nan])[i]
        row[f"{side}_coach_id"] = _get(t, "coach", "id")
        row[f"{side}_coach"] = _get(t, "coach", "name")
        row[f"{side}_formation"] = t.get("formation")
        row[f"{side}_xi_value"] = _num(t.get("totalStarterMarketValue"))
        row[f"{side}_xi_age"] = _num(t.get("averageStarterAge"))
        for role, label in ROLES.items():
            for p in t.get(role) or []:
                un = p.get("unavailability") or {}
                subs = _get(p, "performance", "substitutionEvents", default=[])
                players.append({
                    "fm_id": fm_id, "side": side, "team_fm_id": t.get("id"),
                    "player_id": p.get("id"), "name": p.get("name"), "role": label,
                    "position_id": p.get("usualPlayingPositionId"),
                    "market_value": _num(p.get("marketValue")), "age": _num(p.get("age")),
                    "sub_minute": _num(subs[0].get("time")) if subs else np.nan,
                    "unavailable_type": un.get("type"),
                    "expected_return": un.get("expectedReturn"),
                })
    return row, players


# --- Artimli toplama -----------------------------------------------------------------------

def _read(path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def update_season(s: PoliteSession, league: str, season: str, out_dir, budget: int,
                  upcoming_days: int = 10, now: pd.Timestamp | None = None
                  ) -> tuple[int, pd.DataFrame]:
    """Bir lig-sezonu gunceller; en fazla `budget` mac detayi indirir.

    Biten maclar bir kez indirilir ve `matches_/players_<lig>_<sezon>.parquet` dosyalarina
    eklenir (degismez kayit). Onumuzdeki `upcoming_days` gun icindeki maclarin eksik oyuncu
    listesi her calismada yeniden indirilir ve dondurulur (saklanmaz, cunku degisir).
    Dondurur: (kullanilan istek sayisi, yaklasan maclarin eksik oyunculari).
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = 1
    fx = parse_fixtures(fetch_league(s, league, season), league, season)
    fx.to_parquet(out_dir / f"fixtures_{league}_{season}.parquet", index=False)
    mp = out_dir / f"matches_{league}_{season}.parquet"
    pp = out_dir / f"players_{league}_{season}.parquet"
    old_m, old_p = _read(mp), _read(pp)
    have = set(old_m["fm_id"]) if not old_m.empty else set()
    todo = fx[fx["finished"] & ~fx["cancelled"] & ~fx["fm_id"].isin(have)]
    rows, players = [], []
    for fm_id in todo["fm_id"].head(max(budget - used, 0)):
        try:
            r, p = parse_match(fetch_match(s, fm_id))
        except (SourceUnavailable, ValueError) as exc:
            log.warning("FotMob mac %s alinamadi: %s", fm_id, exc)
            continue
        finally:
            used += 1
        rows.append(r)
        players += p
    if rows:
        pd.concat([old_m, pd.DataFrame(rows)], ignore_index=True).to_parquet(mp, index=False)
        pd.concat([old_p, pd.DataFrame(players)], ignore_index=True).to_parquet(pp, index=False)
    log.info("FotMob %s %s: %d yeni mac, %d mac kaldi", league, season, len(rows),
             len(todo) - len(rows))
    soon = fx[~fx["finished"] & ~fx["cancelled"] & (fx["kickoff"] >= now)
              & (fx["kickoff"] <= now + pd.Timedelta(days=upcoming_days))]
    up = []
    for fm_id in soon["fm_id"]:
        if used >= budget:
            break
        try:
            _, p = parse_match(fetch_match(s, fm_id))
        except (SourceUnavailable, ValueError) as exc:
            log.warning("FotMob yaklasan mac %s alinamadi: %s", fm_id, exc)
            continue
        finally:
            used += 1
        up += [x for x in p if x["role"] == "unavailable"]
    return used, pd.DataFrame(up)
