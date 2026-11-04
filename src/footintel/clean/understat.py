"""Understat JSON'unu mac bazinda bir tabloya cevirir ve football-data maclarina baglar.

Understat iki liste verir:
- `dates`: her mac (id, tarih, ev/deplasman takimi, gol, xG).
- `teams`: her takimin mac gecmisi (xG, npxG, PPDA, deep completion, xPts).
Ikisini (tarih, takim) anahtariyla birlestirip bir maca bir satir uretiyoruz.

Metrik sozlugu:
- xG: sutlarin gol olma olasiliklarinin toplami.
- npxG: penaltilar haric xG. Penaltilar takim kalitesinden cok sansla ilgili
  oldugu icin modellerde genelde npxG tercih edilir.
- PPDA: rakibin pas sayisi / bizim savunma aksiyonlarimiz (rakip sahanin
  %60'inda). Dusuk PPDA = yogun baski. Understat bunu kendi event verisinden
  hesapliyor; Super Lig icin bu metrik elimizde yok.
- deep: rakip ceza sahasina ~20 metre yakinlikta tamamlanan paslar (orta haric).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TEAM_FIELDS = {
    "npxG": "npxg",
    "deep": "deep",
    "xpts": "xpts",
}


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_league_season(path: Path, league: str, season: str) -> pd.DataFrame:
    """Ham JSON -> bir satiri bir mac olan tablo (yalnizca oynanmis maclar)."""
    data = json.loads(path.read_text(encoding="utf-8"))

    team_stats: dict[tuple[str, str], dict] = {}
    for team in data["teams"].values():
        for m in team["history"]:
            ppda = m.get("ppda") or {}
            att, deff = _num(ppda.get("att")), _num(ppda.get("def"))
            team_stats[(m["date"], str(team["id"]))] = {
                **{dst: _num(m.get(src)) for src, dst in TEAM_FIELDS.items()},
                "ppda": att / deff if att is not None and deff else None,
            }

    rows = []
    for m in data["dates"]:
        if not m.get("isResult"):
            continue  # henuz oynanmamis mac
        row = {
            "league": league,
            "season": season,
            "understat_match_id": int(m["id"]),
            "match_datetime": m["datetime"],
            "home_name": m["h"]["title"],
            "away_name": m["a"]["title"],
            "us_home_goals": int(m["goals"]["h"]),
            "us_away_goals": int(m["goals"]["a"]),
            "home_xg": _num(m["xG"]["h"]),
            "away_xg": _num(m["xG"]["a"]),
        }
        for side, key in (("home", "h"), ("away", "a")):
            stats = team_stats.get((m["datetime"], str(m[key]["id"])), {})
            for field in ("npxg", "ppda", "deep", "xpts"):
                row[f"{side}_{field}"] = stats.get(field)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_datetime"]).dt.date
    return df


def link_to_football_data(
    us: pd.DataFrame, fd: pd.DataFrame, name_to_id: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Understat maclarini football-data match_id'lerine baglar.

    Anahtar: (lig, ev takimi, deplasman takimi, tarih). Bir lig-sezonda iki takim
    ayni sahada yalnizca bir kez oynar; tarihi de ekleyince anahtar benzersizdir.
    Erteleme ya da saat dilimi farki nedeniyle tarih 1 gun kayabilir, bu yuzden
    once tam tarihle, eslesmeyenleri +-1 gun toleransla deneriz.

    Dogrulama: eslesen maclarda iki kaynagin skoru ayni olmali. Degilse yanlis
    esleme yapmisiz demektir.

    Donus: (eslesen satirlar, eslesmeyen satirlar)
    """
    us = us.copy()
    us["home_team_id"] = us["home_name"].map(name_to_id)
    us["away_team_id"] = us["away_name"].map(name_to_id)
    # pandas merge'de NaN anahtarlar birbirine eslesir; eslenmemis takimlari disarida birak.
    us = us.dropna(subset=["home_team_id", "away_team_id"])

    keys = ["league", "season", "home_team_id", "away_team_id"]
    fd_keys = fd[[*keys, "match_id", "match_date", "home_goals", "away_goals"]].copy()
    fd_keys["match_date"] = pd.to_datetime(fd_keys["match_date"])
    us["match_date"] = pd.to_datetime(us["match_date"])

    merged = us.merge(fd_keys, on=keys, how="left", suffixes=("", "_fd"))
    day_gap = (merged["match_date"] - merged["match_date_fd"]).abs().dt.days
    merged = merged[day_gap.le(1) | merged["match_id"].isna()]
    # Ayni Understat macina birden fazla aday kalirsa en yakin tarihi sec.
    merged["gap"] = (merged["match_date"] - merged["match_date_fd"]).abs()
    merged = merged.sort_values("gap").drop_duplicates("understat_match_id")

    linked = merged[merged["match_id"].notna()].copy()
    unlinked = us[~us["understat_match_id"].isin(linked["understat_match_id"])]

    score_mismatch = linked[
        (linked["us_home_goals"] != linked["home_goals"])
        | (linked["us_away_goals"] != linked["away_goals"])
    ]
    if len(score_mismatch):
        raise ValueError(
            f"{len(score_mismatch)} macta Understat ve football-data skoru farkli; "
            f"takim eslemesi yanlis olabilir: "
            f"{score_mismatch[['match_id', 'home_name', 'away_name']].head().to_dict('records')}"
        )
    cols = ["match_id", "understat_match_id",
            *[f"{s}_{f}" for s in ("home", "away") for f in ("xg", "npxg", "ppda", "deep", "xpts")]]
    return linked[cols].reset_index(drop=True), unlinked
