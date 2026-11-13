"""Faz 10: 1X2 sonuc modeli (dc-mr-v3): walk-forward test, kapi, canli model ve gecmis tahmin.

Adimlar:
1. Dixon-Coles'u 2022/23'ten bugune haftalik walk-forward calistir (her tahmin kendi
   tarihinde, yalnizca gecmisle).
2. Piyasa reytinglerini hesapla (yalnizca onceki maclarin oranlari).
3. Lojistik regresyonu her test ayinin basinda yeniden egit; test sezonlarinda
   DC ve piyasa ile ayni maclar uzerinde kiyasla (eslestirilmis bootstrap).
4. Kapi: yeni modelin DC'ye gore log loss farkinin %95 ust siniri 0'in altinda mi?
5. Tum oynanmis maclarla son modeli egit -> data/clean/outcome_model.json (canli tahmin).
6. Gecmis tahmin tablosu icin 2025/26-2026/27 walk-forward 1X2 -> outcome_backtest.parquet.

Model ve ayar secimi yalnizca 2023/24'te yapildi; test sezonlari secimde kullanilmadi.

Kullanim:
    uv run python pipelines/build_outcome.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.analysis.eda import implied_1x2, played
from footintel.config import CLEAN_DIR
from footintel.features.market_rating import match_ratings
from footintel.models import backtest as B
from footintel.models import outcome as O
from footintel.models.ml import paired_bootstrap
from footintel.models.registry import GOAL_CONFIG, OUTCOME_MODEL_FILE

log = logging.getLogger("build_outcome")
TRAIN_FROM = "2223"
TEST = ["2425", "2526", "2627"]
HISTORY = ["2526", "2627"]


def _losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def compare(data: pd.DataFrame, new: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Ayni maclarda DC, yeni model ve piyasa (kapanis) log loss'u."""
    cols = ["p_home", "p_draw", "p_away"]
    d = data.dropna(subset=["home_goals", "away_goals"]).merge(
        new.rename(columns={c: f"new_{c}" for c in cols}), on="match_id")
    mk = implied_1x2(d)[cols].to_numpy()
    ok = ~np.isnan(mk).any(axis=1)
    d, mk = d[ok].reset_index(drop=True), mk[ok]
    y = O.outcome(d)
    d["ll_dc"] = _losses(d[cols].to_numpy(), y)
    d["ll_new"] = _losses(d[[f"new_{c}" for c in cols]].to_numpy(), y)
    d["ll_market"] = _losses(mk, y)
    d["hit_dc"] = d[cols].to_numpy().argmax(1) == y
    d["hit_new"] = d[[f"new_{c}" for c in cols]].to_numpy().argmax(1) == y
    diff, lo, hi = paired_bootstrap(d["ll_new"].to_numpy(), d["ll_dc"].to_numpy())
    gap, glo, ghi = paired_bootstrap(d["ll_new"].to_numpy(), d["ll_market"].to_numpy())
    summary = {
        "n": int(len(d)), "test_seasons": TEST,
        "logloss_dc": float(d["ll_dc"].mean()), "logloss_new": float(d["ll_new"].mean()),
        "logloss_market": float(d["ll_market"].mean()),
        "acc_dc": float(d["hit_dc"].mean()), "acc_new": float(d["hit_new"].mean()),
        "diff_vs_dc": [diff, lo, hi], "diff_vs_market": [gap, glo, ghi],
        "gate_passed": bool(hi < 0),
    }
    by = pd.concat([d.groupby("season")[["ll_dc", "ll_new", "ll_market"]].mean(),
                    d.groupby("league")[["ll_dc", "ll_new", "ll_market"]].mean()])
    return summary, by


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    seasons = sorted(s for s in fd["season"].unique() if s >= TRAIN_FROM)
    dc = B.run_all(fd, seasons, GOAL_CONFIG)[["match_id", "p_home", "p_draw", "p_away"]]
    data = fd.merge(dc, on="match_id").merge(match_ratings(fd), on="match_id", how="left")
    log.info("DC walk-forward: %d mac, sezonlar %s", len(data), seasons)

    new = O.walk_forward(data, TEST, TRAIN_FROM)
    summary, by = compare(data, new)
    log.info("1X2 log loss (test %s, n=%d): DC %.4f, dc-mr-v3 %.4f, piyasa %.4f", TEST,
             summary["n"], summary["logloss_dc"], summary["logloss_new"],
             summary["logloss_market"])
    log.info("Fark (yeni - DC): %+.4f [%+.4f, %+.4f] -> kapi %s", *summary["diff_vs_dc"],
             "GECTI" if summary["gate_passed"] else "GECMEDI")
    log.info("Sezon / lig bazinda:\n%s", by.round(4).to_string())
    if not summary["gate_passed"]:
        log.warning("dc-mr-v3 bu veride DC'yi anlamli olarak gecmedi; sonucu inceleyin.")

    model = O.fit(data)
    (CLEAN_DIR / OUTCOME_MODEL_FILE).write_text(json.dumps(model.to_dict(), indent=1))
    (CLEAN_DIR / "outcome_eval.json").write_text(json.dumps(
        summary | {"by": by.round(5).reset_index(names="group").to_dict("records")}, indent=1))
    hist = new[new["match_id"].isin(fd.loc[fd["season"].isin(HISTORY), "match_id"])]
    hist.to_parquet(CLEAN_DIR / "outcome_backtest.parquet", index=False)
    log.info("Canli model %d macla egitildi; gecmis tablo icin %d mac", model.n_train, len(hist))

    step = os.environ.get("GITHUB_STEP_SUMMARY")
    if step:
        lo = summary["diff_vs_dc"]
        with Path(step).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 10: 1X2 modeli dc-mr-v3\n\nTest {', '.join(TEST)}, "
                    f"n = {summary['n']}. Log loss: DC {summary['logloss_dc']:.4f}, "
                    f"yeni {summary['logloss_new']:.4f}, piyasa "
                    f"{summary['logloss_market']:.4f}. Fark {lo[0]:+.4f} "
                    f"[{lo[1]:+.4f}, {lo[2]:+.4f}], kapi "
                    f"{'gecti' if summary['gate_passed'] else 'GECMEDI'}.\n\n"
                    f"```\n{by.round(4).to_string()}\n```\n")


if __name__ == "__main__":
    main()
