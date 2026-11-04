"""Kendi Elo puanlarimizi hesaplar: parametre secimi + zamansal dogrulama + yukleme.

Zamansal ayrim (data leakage'a karsi):
- 2021/22: isinma. Herkes 1500'den basladigi icin bu sezonun tahminleri olculmez.
- 2022/23 - 2024/25: parametre secimi (egitim).
- 2025/26: dokunulmamis test sezonu. Parametreler bu sezonu hic gormeden secilir.

Kullanim:
    uv run python pipelines/build_elo.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from footintel import db
from footintel.config import CLEAN_DIR
from footintel.features.elo import EloParams, evaluate, run_elo, tune

log = logging.getLogger("build_elo")

TRAIN = ["2223", "2324", "2425"]
TEST = ["2526"]
GRID_K = [10, 15, 20, 25, 30]
GRID_HFA = [30, 50, 70, 90]
GRID_REG = [0.0, 0.2, 0.4]


def baseline_mse(fd: pd.DataFrame, seasons: list[str]) -> float:
    """Takim gucunu hic bilmeyen model: egitim sezonlarindaki ortalama ev skoru."""
    def scores(df):
        gd = df["home_goals"] - df["away_goals"]
        return np.select([gd > 0, gd == 0], [1.0, 0.5], 0.0)
    train = fd[fd["season"].isin(TRAIN)].dropna(subset=["home_goals"])
    test = fd[fd["season"].isin(seasons)].dropna(subset=["home_goals"])
    return float(np.mean((scores(test) - scores(train).mean()) ** 2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    fd["home_goals"] = fd["home_goals"].astype("float")
    fd["away_goals"] = fd["away_goals"].astype("float")

    grid = tune(fd, TRAIN, GRID_K, GRID_HFA, GRID_REG)
    best = grid.iloc[0]
    params = EloParams(k=float(best["k"]), home_advantage=float(best["hfa"]),
                       season_regression=float(best["reg"]))
    elo = run_elo(fd, params)

    report = pd.DataFrame({
        "train_mse": [evaluate(fd, elo, TRAIN), baseline_mse(fd, TRAIN)],
        "test_mse": [evaluate(fd, elo, TEST), baseline_mse(fd, TEST)],
    }, index=["elo", "baseline (sabit ev avantaji)"]).round(4)
    per_league = pd.Series({
        lg: evaluate(fd[fd["league"] == lg], elo, TEST) for lg in sorted(fd["league"].unique())
    }, name="test_mse").round(4)

    # Yalnizca her ligin en son sezonundaki takimlar (kume dusenler listede kalmasin).
    latest = fd.groupby("league")["season"].transform("max")
    last = fd[fd["season"] == latest].merge(elo, on="match_id").sort_values("match_date")
    current = pd.concat([
        last[["league", "home_team", "home_elo_pre", "match_date"]].set_axis(
            ["league", "team", "elo", "date"], axis=1),
        last[["league", "away_team", "away_elo_pre", "match_date"]].set_axis(
            ["league", "team", "elo", "date"], axis=1),
    ]).sort_values("date").drop_duplicates("team", keep="last")
    top = (current.sort_values("elo", ascending=False).groupby("league").head(3)
           .sort_values(["league", "elo"], ascending=[True, False])[["league", "team", "elo"]]
           .round(0))

    print(f"\nEn iyi parametreler: {params}")
    print("\nIlk 5 kombinasyon (egitim MSE):")
    print(grid.head(5).to_string(index=False))
    print("\nModel vs baseline:")
    print(report.to_string())
    print("\nTest sezonu, lig bazinda:")
    print(per_league.to_string())
    print("\nGuncel en yuksek puanli takimlar (mac oncesi son puan):")
    print(top.to_string(index=False))

    con = db.connect()
    n = db.replace_table(con, "fact_match_elo", elo)
    elo.to_parquet(CLEAN_DIR / "match_elo.parquet", index=False)
    log.info("fact_match_elo: %d satir", n)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Kendi Elo modelimiz\n\nEn iyi parametreler: `{params}`\n\n")
            f.write(f"```\n{report.to_string()}\n\n{per_league.to_string()}\n```\n\n")
            f.write(f"Guncel ilk 3 (lig bazinda):\n\n```\n{top.to_string(index=False)}\n```\n")


if __name__ == "__main__":
    main()
