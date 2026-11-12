"""Faz 4: onumuzdeki maclar icin tahmin uretir ve degistirilemez deftere yazar.

Akis:
1. football-data.co.uk fikstur listesini indir (--offline: en son indirilen dosyayi kullan).
2. Bugunden itibaren HORIZON_DAYS gun icindeki, henuz tahmin edilmemis maclari sec.
3. Her lig icin modeli bugune kadarki tum oynanmis maclarla fit et.
4. Tum pazarlarin olasiliklarini data/predictions/ altina yeni bir dosya olarak yaz.
CI bu dosyayi zamanlanmis calistirmalarda git'e commit'ler (bkz. .github/workflows/data.yml).

Kullanim:
    uv run python pipelines/predict_upcoming.py
    uv run python pipelines/predict_upcoming.py --dry-run   # deftere yazmadan goster
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from footintel import ledger
from footintel.analysis.eda import played
from footintel.clean.fixtures import kickoff_utc, parse_fixtures
from footintel.clean.football_data import read_raw_csv
from footintel.config import CLEAN_DIR, CURRENT_SEASON, LEAGUES, LEDGER_DIR
from footintel.ingest import football_data as fd_ingest
from footintel.models import backtest as B
from footintel.models import corners as C
from footintel.models import strength as S
from footintel.models.registry import (
    CORNER_CONFIG,
    CORNER_MODEL_FILE,
    GOAL_CONFIG,
    MODEL_VERSION,
)

log = logging.getLogger("predict_upcoming")
HORIZON_DAYS = 10


def predict_league(results: pd.DataFrame, fixtures: pd.DataFrame, league: str,
                   ref: pd.Timestamp, corner_reg: C.NBRegression) -> dict[str, tuple]:
    lg = results[results["league"] == league]
    teams = B.season_teams(lg)
    cur = teams.get((league, CURRENT_SEASON), set())
    fx = fixtures[fixtures["league"] == league]
    teams[(league, CURRENT_SEASON)] = cur | set(fx["home_team_id"]) | set(fx["away_team_id"])
    promoted, relegated = B.promoted_relegated(teams, league, CURRENT_SEASON)
    gm = S.fit(lg, ref, GOAL_CONFIG, promoted=promoted, relegated=relegated)
    cm = S.fit(lg, ref, CORNER_CONFIG, home_col="home_corners", away_col="away_corners",
               promoted=promoted, relegated=relegated)
    needs_shots = any(f in corner_reg.features for f in ("shot_rate", "opp_shot_rate"))
    sm = S.fit(lg, ref, C.SHOT_CONFIG, home_col="home_shots", away_col="away_shots",
               promoted=promoted, relegated=relegated) if needs_shots else None
    if not (gm.success and cm.success):
        log.warning("%s: optimizasyon tam yakinsamadi (gol=%s, korner=%s)",
                    league, gm.success, cm.success)
    out = {}
    for r in fx.itertuples(index=False):
        m = S.score_matrix(gm, r.home_team_id, r.away_team_id)
        out[r.match_key] = (S.goal_markets(m), m,
                            C.live_markets(corner_reg, gm, cm, r.home_team_id, r.away_team_id,
                                           shot_model=sm))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = now.normalize()
    run_date = today.date().isoformat()
    run_stamp = now.strftime("%Y-%m-%dT%H%M")   # gunde birden fazla calistirma olabilir
    if args.offline:
        path = max(fd_ingest.RAW_SUBDIR.glob("fixtures_*.csv"))
    else:
        path = fd_ingest.download_fixtures(run_date)
    raw = read_raw_csv(path)
    log.info("Fikstur dosyasi: %d satir, ligler: %s, tarihler: %s - %s", len(raw),
             raw["Div"].value_counts().to_dict() if "Div" in raw else "Div kolonu yok",
             raw["Date"].iloc[0] if len(raw) else "-", raw["Date"].iloc[-1] if len(raw) else "-")
    fixtures = parse_fixtures(path)
    dates = pd.to_datetime(fixtures["match_date"])
    fixtures = fixtures[(dates >= today) & (dates <= today + pd.Timedelta(days=HORIZON_DAYS))]
    # Bugun baslamis (ya da bitmis) maclar artik "mac oncesi" degildir.
    kick = pd.Series([kickoff_utc(d, t) for d, t in
                      fixtures[["match_date", "kickoff_time"]].itertuples(index=False)],
                     index=fixtures.index, dtype="datetime64[ns]")
    fixtures = fixtures[kick.isna() | (kick > now)]
    done = ledger.already_predicted(LEDGER_DIR, MODEL_VERSION)
    new = fixtures[~fixtures["match_key"].isin(done)]
    log.info("Fikstur: %d mac, %d tanesi zaten defterde, %d yeni",
             len(fixtures), len(fixtures) - len(new), len(new))

    results = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    corner_reg = C.NBRegression.from_dict(
        json.loads((CLEAN_DIR / CORNER_MODEL_FILE).read_text()))
    preds: dict[str, tuple] = {}
    for league in sorted(new["league"].unique()):
        preds.update(predict_league(results, new, league, today, corner_reg))
    rows = ledger.build(new, preds, MODEL_VERSION, now, ledger.snapshot_hash(results))

    view = rows[rows["market"].isin(["1x2", "ou25", "btts"])].pivot_table(
        index=["league", "match_date", "home_team", "away_team"],
        columns=["market", "selection"], values="prob").round(2)
    if not view.empty:
        view.columns = [f"{m}:{s}" for m, s in view.columns]
        view = view.rename(index=lambda x: LEAGUES[x].name if x in LEAGUES else x, level=0)
    text = view.to_string() if not view.empty else "Yeni mac yok."
    print(text)

    written = None
    if not args.dry_run:
        written = ledger.append(LEDGER_DIR, rows, run_stamp, MODEL_VERSION)
        log.info("Deftere yazildi: %s (%d satir)", written, len(rows))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Tahminler ({MODEL_VERSION}, {run_date})\n\n"
                    f"{len(new)} yeni mac. Defter dosyasi: `{written}`"
                    f"{' (dry-run, yazilmadi)' if args.dry_run else ''}\n\n"
                    f"```\n{text}\n```\n")


if __name__ == "__main__":
    main()
