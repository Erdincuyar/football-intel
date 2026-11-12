"""Faz 9 asama 2: gecmis maclar icin surumlu, sonuclardan bagimsiz tahmin tablosu.

Sezonlar: 2025/26 ve 2026/27. Neden daha eskisi degil? Gol modelinin ayarlari 2024/25'te,
kart modelininki 2023/24'te secildi. Bu sezonlari "gecmis tahmin" diye gostermek, ayarlarin
o sezonu gormus olmasi yuzunden hafif bir sizinti olurdu. 2025/26'dan itibaren temiz.

Yazma: data/clean/historical_predictions.parquet (uygulama yedegi) ve, adres varsa, Neon'daki
app.historical_predictions (anahtar: match_id + model_version, yalnizca ekleme).

Asama 3: tahminler (Neon'daki degismez tablo, yoksa parquet) sonuclarla eslestirilir ve
degerlendirilir. Ciktilar turetilmistir, her calismada bastan yazilir:
match_results, prediction_evaluation (mac basina), evaluation_summary (isabet yuzdeleri),
evaluation_metrics (Brier, log loss, MAE). Neon'a Faz 8 yayin adimi tasir.

Kullanim:
    uv run python pipelines/build_history.py
"""

from __future__ import annotations

import logging

import pandas as pd

from footintel import store
from footintel.analysis import history_eval as E
from footintel.config import CLEAN_DIR
from footintel.models import history as H

log = logging.getLogger("build_history")
SEASONS = ["2526", "2627"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    bt = pd.read_parquet(CLEAN_DIR / "backtest_predictions.parquet")
    cards_path = CLEAN_DIR / "cards_backtest.parquet"
    cards = pd.read_parquet(cards_path) if cards_path.exists() else None
    fd = fd[fd["season"].isin(SEASONS)]
    hist = H.assemble(fd, bt, cards, pd.Timestamp.now(tz="UTC"))
    hist.to_parquet(CLEAN_DIR / "historical_predictions.parquet", index=False)
    log.info("Gecmis tahmin: %d mac (%s)", len(hist),
             hist.groupby(["league", "season"]).size().to_dict())
    url = store.database_url()
    if url:
        n = store.append_immutable(url, "historical_predictions", hist,
                                   ["match_id", "model_version"])
        log.info("Neon: %d yeni satir eklendi (var olanlar korundu)", n)
    evaluate(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"), url)


def evaluate(fd_all: pd.DataFrame, url: str | None) -> None:
    """Saklanan tahminleri (tum surumler) sonuclarla eslestirir; turetilmis tablolari yazar."""
    preds = store.read_table("historical_predictions", url=url or "")
    results = E.match_results(fd_all[fd_all["season"].isin(SEASONS)])
    ev = E.evaluate_matches(preds, results, E.league_priors(fd_all))
    keys = ["model_version", "league", "season"]
    summary = pd.concat([E.summarize(ev, keys).assign(scope="season"),
                         E.summarize(ev, [*keys, "matchweek"]).assign(scope="week")],
                        ignore_index=True)
    metrics = pd.concat([E.prob_metrics(ev, keys).assign(scope="season"),
                         E.prob_metrics(ev, [*keys, "matchweek"]).assign(scope="week")],
                        ignore_index=True)
    for name, df in (("match_results", results), ("prediction_evaluation", ev),
                     ("evaluation_summary", summary), ("evaluation_metrics", metrics)):
        df.to_parquet(CLEAN_DIR / f"{name}.parquet", index=False)
    log.info("Degerlendirilen mac: %d (sonucu olmayanlar haric)", len(ev))
    season = summary[summary["scope"] == "season"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("Sezon isabetleri:\n%s", season.round(3).to_string(index=False))
        log.info("Olasilik metrikleri:\n%s",
                 metrics[metrics["scope"] == "season"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
