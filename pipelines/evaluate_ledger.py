"""Defterdeki tahminleri gerceklesen sonuclarla karsilastirir (prediction_eval).

Defterin kendisi degismez; degerlendirme her calistirmada defterden ve sonuclardan
yeniden hesaplanir ve DuckDB'ye yuklenir.

Kullanim:
    uv run python pipelines/evaluate_ledger.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from footintel import db, ledger
from footintel.config import CLEAN_DIR, LEDGER_DIR

log = logging.getLogger("evaluate_ledger")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    led = ledger.load(LEDGER_DIR)
    results = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    ev = ledger.evaluate(led, results)
    summary_df = ledger.market_summary(led, ev)

    n_matches = led["match_key"].nunique() if len(led) else 0
    n_done = led.loc[led["prediction_id"].isin(ev["prediction_id"]), "match_key"].nunique()
    text = (f"Defter: {n_matches} mac, {len(led)} tahmin satiri; sonuclanan mac: {n_done}\n\n"
            + (summary_df.round(4).to_string(index=False) if len(summary_df)
               else "Henuz sonuclanan tahmin yok."))
    print(text)

    con = db.connect()
    db.replace_table(con, "prediction_log", led)
    db.replace_table(con, "prediction_eval", ev)
    led_out = CLEAN_DIR / "prediction_eval.parquet"
    ev.to_parquet(led_out, index=False)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Tahmin defteri degerlendirmesi\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
