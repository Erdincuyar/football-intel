"""football-data.co.uk fikstur dosyasini (fixtures.csv) tahmin girdisine cevirir.

Sonuc dosyalariyla ayni kaynak oldugu icin takim yazimlari aynidir; ayni team_id
fonksiyonunu kullanarak ek bir eslestirme gerekmez.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from footintel.clean.football_data import ODDS_COLUMNS, read_raw_csv
from footintel.clean.teams import team_id
from footintel.config import CURRENT_SEASON, LEAGUES


def match_key(league: str, season: str, home_id: str, away_id: str) -> str:
    """Tarihten bagimsiz mac anahtari. Bir sezonda iki takim ayni sahada bir kez oynar;
    bu yuzden mac ertelense bile anahtar degismez ve tahmin sonucla eslesir."""
    return f"{league}-{season}-{home_id}-{away_id}"


def parse_fixtures(path: Path, season: str = CURRENT_SEASON) -> pd.DataFrame:
    raw = read_raw_csv(path)
    raw = raw[raw["Div"].isin(LEAGUES)]
    df = pd.DataFrame({
        "league": raw["Div"],
        "season": season,
        "match_date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed").dt.date,
        "kickoff_time": raw["Time"] if "Time" in raw else pd.NA,
        "home_team": raw["HomeTeam"].str.strip(),
        "away_team": raw["AwayTeam"].str.strip(),
    })
    df["home_team_id"] = df["home_team"].map(team_id)
    df["away_team_id"] = df["away_team"].map(team_id)
    df["match_key"] = [match_key(*r) for r in
                       df[["league", "season", "home_team_id", "away_team_id"]].itertuples(
                           index=False)]
    for src, dst in ODDS_COLUMNS.items():
        df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else float("nan")
    return df.drop_duplicates("match_key").reset_index(drop=True)
