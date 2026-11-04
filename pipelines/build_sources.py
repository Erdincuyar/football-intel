"""Understat ve ClubElo verisini indirir, takim isimlerini esler ve DuckDB'ye yukler.

On kosul: pipelines/build_football_data.py calismis olmali (takim kimlikleri ve
maclar oradan gelir).

Kullanim:
    uv run python pipelines/build_sources.py
    uv run python pipelines/build_sources.py --offline   # indirmeden, mevcut ham dosyalarla
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from footintel import db
from footintel.clean import understat as us_clean
from footintel.clean.team_matching import MatchResult, load_aliases, match_teams
from footintel.config import (
    CLEAN_DIR,
    CLUBELO_COUNTRIES,
    SEASONS,
    UNDERSTAT_LEAGUES,
)
from footintel.ingest import clubelo as elo_ingest
from footintel.ingest import understat as us_ingest
from footintel.ingest.http import PoliteSession

log = logging.getLogger("build_sources")
ELO_START = pd.Timestamp("2020-07-01")


def candidates_by_league_season(fd: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for (league, season), g in fd.groupby(["league", "season"]):
        teams = pd.concat([
            g[["home_team_id", "home_team"]].set_axis(["id", "name"], axis=1),
            g[["away_team_id", "away_team"]].set_axis(["id", "name"], axis=1),
        ]).drop_duplicates("id")
        out[(league, season)] = dict(zip(teams["id"], teams["name"], strict=True))
    return out


def resolve(
    source: str,
    names_by_ls: dict[tuple[str, str], list[str]],
    cands: dict[tuple[str, str], dict[str, str]],
    *,
    every_source_name: bool,
) -> tuple[dict[str, str], list[dict], list[str]]:
    """Tum lig-sezonlar icin isimleri esler. Donus: (isim->team_id, inceleme listesi, hatalar)."""
    aliases = load_aliases(source)
    mapping: dict[str, str] = {}
    review: list[dict] = []
    errors: list[str] = []
    for (league, season), names in sorted(names_by_ls.items()):
        cand = cands.get((league, season), {})
        results: list[MatchResult] = match_teams(sorted(set(names)), cand, aliases)
        used = {r.team_id for r in results if r.team_id}
        missing_fd = sorted(set(cand) - used)
        missing_src = [r.source_name for r in results if r.team_id is None]
        if missing_fd or (every_source_name and missing_src):
            errors.append(
                f"{source} {league} {season}: eslenemeyen kaynak isimleri {missing_src}; "
                f"karsiligi bulunamayan football-data takimlari "
                f"{[cand[t] + ' (' + t + ')' for t in missing_fd]}"
            )
        for r in results:
            if r.team_id is None:
                continue
            previous = mapping.get(r.source_name)
            if previous and previous != r.team_id:
                errors.append(
                    f"{source}: {r.source_name!r} hem {previous} hem {r.team_id} ile eslesti"
                )
            mapping[r.source_name] = r.team_id
            if r.method == "fuzzy" and r.score < 1.0:
                review.append({"source": source, "league": league, "season": season,
                               "source_name": r.source_name, "fd_name": cand[r.team_id],
                               "score": r.score})
    return mapping, review, errors


def build_understat(fd, cands, offline: bool):
    if not offline:
        us_ingest.download_all()
    frames = []
    for league in UNDERSTAT_LEAGUES:
        for season in SEASONS:
            path = us_ingest.raw_path(league, season)
            if path.exists():
                frames.append(us_clean.parse_league_season(path, league, season))
    us = pd.concat(frames, ignore_index=True)
    names = {}
    for (league, season), g in us.groupby(["league", "season"]):
        names[(league, season)] = list(set(g["home_name"]) | set(g["away_name"]))
    mapping, review, errors = resolve("understat", names, cands, every_source_name=True)
    return us, mapping, review, errors


def build_clubelo(cands, offline: bool):
    session = PoliteSession(delay=1.0)
    snapshots = (
        {s: elo_ingest._read_csv(elo_ingest.snapshot_path(s).read_text()) for s in SEASONS}
        if offline else elo_ingest.download_snapshots(session)
    )
    names = {}
    for (league, season) in cands:
        snap = snapshots[season]
        top = snap[(snap["Country"] == CLUBELO_COUNTRIES[league]) & (snap["Level"] == 1)]
        names[(league, season)] = top["Club"].tolist()
    mapping, review, errors = resolve("clubelo", names, cands, every_source_name=False)
    return session, mapping, review, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    cands = candidates_by_league_season(fd)

    us, us_map, us_review, us_err = build_understat(fd, cands, args.offline)
    session, elo_map, elo_review, elo_err = build_clubelo(cands, args.offline)

    review = pd.DataFrame(us_review + elo_review)
    if not review.empty:
        print("\nOtomatik eslemeler (skor < 1, kontrol et):")
        print(review.sort_values("score").to_string(index=False))

    errors = us_err + elo_err
    if errors:
        for e in errors:
            log.error(e)
        raise SystemExit(
            f"{len(errors)} takim eslestirme sorunu var. "
            "data/reference/team_aliases.csv dosyasina ekleme yapilmali."
        )

    # Understat -> football-data mac baglantisi
    linked, unlinked = us_clean.link_to_football_data(us, fd, us_map)
    link_rate = (
        us.assign(linked=~us["understat_match_id"].isin(unlinked["understat_match_id"]))
        .groupby(["league", "season"])["linked"].mean().mul(100).round(1)
    )
    print("\nUnderstat maclarinin football-data ile eslesme orani (%):")
    print(link_rate.to_string())
    if len(unlinked):
        print(unlinked[["league", "season", "match_date", "home_name", "away_name"]].head(20))
    if (link_rate < 98).any():
        raise SystemExit("Understat eslesme orani %98'in altinda; yukaridaki maclari incele.")

    # ClubElo gecmisleri
    elo_rows = []
    histories = (
        {c: pd.read_csv(elo_ingest.history_path(c), na_values=["None"]) for c in elo_map}
        if args.offline else elo_ingest.download_histories(session, list(elo_map))
    )
    for club, hist in histories.items():
        h = hist.assign(From=pd.to_datetime(hist["From"]), To=pd.to_datetime(hist["To"]))
        h = h[h["To"] >= ELO_START]
        elo_rows.append(pd.DataFrame({
            "team_id": elo_map[club], "valid_from": h["From"].dt.date,
            "valid_to": h["To"].dt.date, "elo": h["Elo"],
        }))
    elo = pd.concat(elo_rows, ignore_index=True).dropna(subset=["elo"])
    elo = elo.drop_duplicates(["team_id", "valid_from"])

    con = db.connect()
    n_us = db.replace_table(con, "fact_match_understat", linked)
    n_elo = db.replace_table(con, "fact_team_elo", elo)
    db.save_team_aliases(con, "understat", us_map)
    db.save_team_aliases(con, "clubelo", elo_map)
    linked.to_parquet(CLEAN_DIR / "matches_understat.parquet", index=False)
    elo.to_parquet(CLEAN_DIR / "team_elo.parquet", index=False)
    log.info("fact_match_understat: %d, fact_team_elo: %d satir", n_us, n_elo)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Understat ve ClubElo\n\n{n_us} Understat maci, {n_elo} Elo satiri.\n\n")
            f.write(f"Understat eslesme orani (%):\n\n```\n{link_rate.to_string()}\n```\n")
            if not review.empty:
                f.write("\nKontrol edilecek otomatik eslemeler:\n\n```\n"
                        f"{review.sort_values('score').to_string(index=False)}\n```\n")


if __name__ == "__main__":
    main()
