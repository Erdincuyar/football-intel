"""Faz 9 asama 2: gecmis maclar icin surumlu, sonuclardan bagimsiz tahmin tablosu.

Sezonlar: 2025/26 ve 2026/27. Neden daha eskisi degil? Gol modelinin ayarlari 2024/25'te,
kart modelininki 2023/24'te secildi. Bu sezonlari "gecmis tahmin" diye gostermek, ayarlarin
o sezonu gormus olmasi yuzunden hafif bir sizinti olurdu. 2025/26'dan itibaren temiz.

Yazma: data/clean/historical_predictions.parquet (uygulama yedegi) ve, adres varsa, Neon'daki
app.historical_predictions (anahtar: match_id + model_version, yalnizca ekleme).

Kullanim:
    uv run python pipelines/build_history.py
"""

from __future__ import annotations

import logging

import pandas as pd

from footintel import store
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


if __name__ == "__main__":
    main()
