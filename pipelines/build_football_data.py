"""football-data.co.uk hattini bastan sona calistirir.

Kullanim:
    uv run python pipelines/build_football_data.py            # indir + temizle + yukle
    uv run python pipelines/build_football_data.py --offline  # indirmeden, mevcut ham dosyalarla
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from footintel import db
from footintel.clean.football_data import clean_matches, coverage_report, read_raw_csv
from footintel.config import CLEAN_DIR, LEAGUES, SEASONS
from footintel.ingest.football_data import download_all, raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="indirme adimini atla")
    parser.add_argument("--force", action="store_true", help="tum sezonlari yeniden indir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.offline:
        download_all(force=args.force)

    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            path = raw_path(league, season)
            if path.exists():
                frames.append(clean_matches(read_raw_csv(path), league, season))
    if not frames:
        raise SystemExit("Hic ham dosya yok. Once --offline olmadan calistir.")
    matches = pd.concat(frames, ignore_index=True)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(CLEAN_DIR / "matches_football_data.parquet", index=False)

    con = db.connect()
    db.load_leagues(con)
    n = db.load_matches(con, matches)
    logging.info("fact_match: %d mac yuklendi", n)

    report = coverage_report(matches)
    print("\nVeri kapsama raporu (% dolu):")
    print(report.to_string())


if __name__ == "__main__":
    main()
