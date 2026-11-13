"""On tahmin: her ligin bir sonraki haftasi icin, fikstur resmi listeye girmeden once.

Neden ayri? Resmi defter (predict_upcoming) football-data fiksturu ve oranlariyla, mactan
birkac gun once yazilir ve degistirilemez. Milli mac arasinda o liste bos kalir ama
kullanicilar bir sonraki haftaya simdiden bakmak ister. On tahmin ayni modelle uretilir,
her calistirmada yenilenir, deftere girmez ve basarisi olculmez.

Cikti: data/clean/preview_predictions.parquet (Faz 8 yayini Neon'a tasir).

Kullanim:
    uv run python pipelines/build_preview.py
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from footintel import ledger
from footintel.analysis import preview as P
from footintel.analysis.eda import played
from footintel.config import CLEAN_DIR, CURRENT_SEASON, LEAGUES, LEDGER_DIR
from footintel.ingest import openfootball as OF
from footintel.models import corners as C
from footintel.models import outcome as O
from footintel.models.live import predict_league
from footintel.models.registry import CORNER_MODEL_FILE, MODEL_VERSION, OUTCOME_MODEL_FILE

log = logging.getLogger("build_preview")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = now.normalize()
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    results = played(fd)
    cur = fd[fd["season"] == CURRENT_SEASON]
    corner_reg = C.NBRegression.from_dict(
        json.loads((CLEAN_DIR / CORNER_MODEL_FILE).read_text()))
    outcome_model = O.OutcomeModel.from_dict(
        json.loads((CLEAN_DIR / OUTCOME_MODEL_FILE).read_text()))
    official = ledger.already_predicted(LEDGER_DIR, MODEL_VERSION)

    parts = []
    for league in LEAGUES:
        matches = OF.download_any(league)
        if not matches:
            log.info("%s: acik fikstur bulunamadi, on tahmin N/A", LEAGUES[league].name)
            continue
        rnd = P.next_round(matches, today)
        c = cur[cur["league"] == league]
        cands = dict(zip(c["home_team_id"], c["home_team"], strict=True)) | dict(
            zip(c["away_team_id"], c["away_team"], strict=True))
        fx, unmatched = P.to_fixtures(rnd, league, CURRENT_SEASON, cands)
        if unmatched:
            log.warning("%s: eslesmeyen takim adlari (bu maclar atlandi): %s",
                        LEAGUES[league].name, unmatched)
        if fx.empty:
            continue
        preds = predict_league(results, fx, league, today, corner_reg, outcome_model)
        parts.append(P.prediction_rows(fx, preds, official, MODEL_VERSION, now))
        log.info("%s: %s, %d mac", LEAGUES[league].name, rnd[0].get("round"), len(fx))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_parquet(CLEAN_DIR / "preview_predictions.parquet", index=False)
    if not out.empty:
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(out[["league", "round", "match_date", "home_team", "away_team", "p_home",
                       "p_draw", "p_away", "pred_score", "p_over25"]].round(2)
                  .to_string(index=False))


if __name__ == "__main__":
    main()
