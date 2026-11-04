"""DuckDB baglantisi ve yukleme fonksiyonlari.

Yukleme "idempotent": ayni veriyi iki kez yuklemek sonucu degistirmez. Bunu
lig-sezon bazinda "once sil, sonra ekle" ile sagliyoruz ve ikisini tek bir
transaction icinde yapiyoruz. Arada hata olursa hicbir sey degismez.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from footintel.config import DB_PATH, LEAGUES, SQL_DIR

ODDS_LAYOUT: dict[str, tuple[str, str, str]] = {
    "odds_home_avg": ("1x2", "home", "avg"),
    "odds_draw_avg": ("1x2", "draw", "avg"),
    "odds_away_avg": ("1x2", "away", "avg"),
    "odds_over25_avg": ("ou25", "over", "avg"),
    "odds_under25_avg": ("ou25", "under", "avg"),
    "odds_home_close": ("1x2", "home", "close"),
    "odds_draw_close": ("1x2", "draw", "close"),
    "odds_away_close": ("1x2", "away", "close"),
    "odds_over25_close": ("ou25", "over", "close"),
    "odds_under25_close": ("ou25", "under", "close"),
}

MATCH_COLUMNS = [
    "match_id", "league_id", "season", "season_label", "match_date", "kickoff_time",
    "home_team_id", "away_team_id", "result", "referee", "quality_flags",
    "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
]


def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute((SQL_DIR / "schema.sql").read_text())
    return con


def load_leagues(con: duckdb.DuckDBPyConnection) -> None:
    rows = [(lg.code, lg.name, lg.country) for lg in LEAGUES.values()]
    con.executemany("INSERT OR REPLACE INTO dim_league VALUES (?, ?, ?)", rows)


def odds_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Genis oran kolonlarini uzun formata cevirir (melt)."""
    cols = [c for c in ODDS_LAYOUT if c in matches]
    long = matches.melt(id_vars="match_id", value_vars=cols, var_name="col", value_name="odds")
    long = long.dropna(subset=["odds"])
    layout = pd.DataFrame.from_dict(
        ODDS_LAYOUT, orient="index", columns=["market", "selection", "snapshot"]
    )
    long = long.join(layout, on="col")
    return long[["match_id", "market", "selection", "snapshot", "odds"]]


def load_matches(con: duckdb.DuckDBPyConnection, matches: pd.DataFrame) -> int:
    """Temiz maclari yukler. Ayni lig-sezon yeniden yuklenirse once silinir."""
    teams = pd.concat([
        matches[["home_team_id", "home_team"]].set_axis(["team_id", "name"], axis=1),
        matches[["away_team_id", "away_team"]].set_axis(["team_id", "name"], axis=1),
    ]).drop_duplicates("team_id")
    fact = matches.rename(columns={"league": "league_id"})[MATCH_COLUMNS]
    odds = odds_long(matches)
    keys = matches[["league", "season"]].drop_duplicates()

    con.execute("BEGIN TRANSACTION")
    try:
        con.register("teams_df", teams)
        con.execute(
            "INSERT OR IGNORE INTO dim_team SELECT team_id, name FROM teams_df"
        )
        con.execute(
            "INSERT OR IGNORE INTO map_team_source "
            "SELECT 'football_data', name, team_id FROM teams_df"
        )
        for league, season in keys.itertuples(index=False):
            ids = "SELECT match_id FROM fact_match WHERE league_id = ? AND season = ?"
            con.execute(f"DELETE FROM fact_odds WHERE match_id IN ({ids})", [league, season])
            con.execute(
                "DELETE FROM fact_match WHERE league_id = ? AND season = ?", [league, season]
            )
        con.register("fact_df", fact)
        con.execute(f"INSERT INTO fact_match ({', '.join(MATCH_COLUMNS)}) SELECT * FROM fact_df")
        con.register("odds_df", odds)
        con.execute("INSERT INTO fact_odds SELECT * FROM odds_df")
        check_integrity(con)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(fact)


def check_integrity(con: duckdb.DuckDBPyConnection) -> None:
    """Semada FK olmayan iliskileri elle kontrol eder."""
    orphans = con.execute(
        "SELECT count(*) FROM fact_odds o "
        "LEFT JOIN fact_match m USING (match_id) WHERE m.match_id IS NULL"
    ).fetchone()[0]
    if orphans:
        raise ValueError(f"fact_odds: {orphans} satirin eslesen maci yok")
