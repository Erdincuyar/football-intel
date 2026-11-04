"""Faz 3: gol ve korner modellerini ayarlar, walk-forward backtest yapar, piyasayla kiyaslar.

Zamansal ayrim:
- Ayar (tuning): 2024/25. Hiperparametreler (zaman azalmasi xi, ceza) burada secilir.
- Test: 2025/26, dokunulmamis. Ayrica 2026/27'nin oynanan kismi ikinci test olarak raporlanir.
Her iki donemde de model, her hafta yalnizca o haftadan onceki maclarla yeniden fit edilir.

Kullanim:
    uv run python pipelines/build_models.py
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from pathlib import Path

import pandas as pd

from footintel import db
from footintel.analysis.eda import calibration_table, played
from footintel.config import CLEAN_DIR, LEAGUES
from footintel.models import backtest as B
from footintel.models import strength as S

log = logging.getLogger("build_models")

TUNE = ["2425"]
TEST = ["2526"]
CURRENT = ["2627"]
GOAL_GRID = {"shot_weight": [0.0, 0.25, 0.5, 0.75], "xi": [0.001, 0.002, 0.004],
             "penalty": [4.0, 8.0, 16.0]}
CORNER_GRID = {"family": ["poisson", "negbin"], "xi": [0.002, 0.004, 0.008],
               "penalty": [32.0, 64.0, 128.0]}


def tune_goals(fd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for beta, xi, pen in itertools.product(*GOAL_GRID.values()):
        cfg = S.ModelConfig(xi=xi, penalty=pen, shot_weight=beta)
        t = time.time()
        preds = B.run_all(fd, TUNE, cfg)
        rows.append({"shot_weight": beta, "xi": xi, "penalty": pen,
                     "score_logloss": -preds["ll_score"].mean(),
                     "seconds": round(time.time() - t, 1)})
        log.info("gol ayari %s", rows[-1])
    return pd.DataFrame(rows).sort_values("score_logloss").reset_index(drop=True)


def tune_corners(fd: pd.DataFrame, goal_cfg: S.ModelConfig) -> pd.DataFrame:
    rows = []
    for fam, xi, pen in itertools.product(*CORNER_GRID.values()):
        cfg = S.ModelConfig(family=fam, dixon_coles=False, xi=xi, penalty=pen)
        preds = B.run_all(fd, TUNE, goal_cfg, cfg)
        rows.append({"family": fam, "xi": xi, "penalty": pen,
                     "corners_logloss": -preds["ll_corners"].mean()})
        log.info("korner ayari %s", rows[-1])
    return pd.DataFrame(rows).sort_values("corners_logloss").reset_index(drop=True)


def warn_if_edge(grid: pd.DataFrame, space: dict) -> str:
    """En iyi deger izgaranin kenarindaysa gercek optimum disarida olabilir."""
    best = grid.iloc[0]
    edges = [k for k, v in space.items()
             if isinstance(v[0], float) and best[k] in (min(v), max(v))]
    return f"UYARI: en iyi deger izgara kenarinda: {edges}" if edges else "izgara kenari: yok"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))

    goal_grid = tune_goals(fd)
    gbest = goal_grid.iloc[0]
    goal_cfg = S.ModelConfig(xi=float(gbest["xi"]), penalty=float(gbest["penalty"]),
                             shot_weight=float(gbest["shot_weight"]))
    corner_grid = tune_corners(fd, goal_cfg)
    cbest = corner_grid.iloc[0]
    corner_cfg = S.ModelConfig(family=str(cbest["family"]), dixon_coles=False,
                               xi=float(cbest["xi"]), penalty=float(cbest["penalty"]))

    sections: dict[str, pd.DataFrame | str] = {
        "Gol modeli ayari (2024/25, kesin skor log loss)": goal_grid,
        "Korner modeli ayari (2024/25, toplam korner log loss)": corner_grid,
        "Secilen ayarlar": (f"gol: {goal_cfg}\nkorner: {corner_cfg}\n"
                            f"gol {warn_if_edge(goal_grid, GOAL_GRID)}\n"
                            f"korner {warn_if_edge(corner_grid, CORNER_GRID)}"),
    }

    # Ablasyon (ayar sezonunda): Dixon-Coles ve sut verisi ayri ayri ne katiyor?
    variants = {
        "yalniz_gol_poisson": S.ModelConfig(dixon_coles=False, xi=goal_cfg.xi,
                                            penalty=goal_cfg.penalty),
        "yalniz_gol_dixon_coles": S.ModelConfig(xi=goal_cfg.xi, penalty=goal_cfg.penalty),
        "secilen_model": goal_cfg,
    }
    sections["Ablasyon (2024/25, kesin skor log loss)"] = pd.DataFrame({
        name: B.run_all(fd, TUNE, cfg).groupby("league")["ll_score"].mean().mul(-1)
        for name, cfg in variants.items()
    }).round(4)

    all_preds = []
    for label, seasons in (("2025/26 (test)", TEST), ("2026/27 (oynanan kisim)", CURRENT)):
        preds = B.run_all(fd, seasons, goal_cfg, corner_cfg)
        all_preds.append(preds)
        ev = B.evaluate(fd, preds, B.base_rates(fd, seasons[0]))
        ev.index = ev.index.map(lambda k: LEAGUES[k].name)
        sections[f"Model vs piyasa: {label}"] = ev.T.round(4)
    preds = pd.concat(all_preds, ignore_index=True)

    # Kalibrasyon (test sezonu): model ev galibiyeti olasiligi
    tp = preds[preds["season"].isin(TEST)].merge(fd[["match_id", "home_goals", "away_goals"]],
                                                 on="match_id")
    sections["Kalibrasyon (2025/26): modelin ev galibiyeti olasiligi"] = calibration_table(
        tp["p_home"], tp["home_goals"] > tp["away_goals"]).round(3)
    sections["Kalibrasyon (2025/26): modelin 2.5 ust olasiligi"] = calibration_table(
        tp["p_over25"], (tp["home_goals"] + tp["away_goals"]) > 2.5).round(3)

    # Guncel takim gucleri (bugune kadarki tum maclarla)
    today = pd.Timestamp(fd["match_date"].max()) + pd.Timedelta(days=1)
    teams = B.season_teams(fd)
    strength_rows = []
    for lg in sorted(fd["league"].unique()):
        season = max(s for (lgg, s) in teams if lgg == lg)
        prom, rel = B.promoted_relegated(teams, lg, season)
        m = S.fit(fd[fd["league"] == lg], today, goal_cfg, promoted=prom, relegated=rel)
        t = m.table()
        t = t[t["team"].isin(teams[(lg, season)])]
        t["league"] = LEAGUES[lg].name
        t["rho"], t["home_adv"] = m.rho, m.home
        strength_rows.append(t)
    strength = pd.concat(strength_rows)
    sections["Lig parametreleri (bugun): ev avantaji ve rho"] = (
        strength.groupby("league")[["home_adv", "rho"]].first().round(3))
    sections["En guclu hucumlar (bugun, lig basina 3)"] = (
        strength.sort_values("attack", ascending=False).groupby("league").head(3)
        [["league", "team", "attack", "defence"]].round(3))

    text = "\n\n".join(
        f"### {k}\n{v if isinstance(v, str) else v.to_string()}" for k, v in sections.items())
    print(text)

    preds.to_parquet(CLEAN_DIR / "backtest_predictions.parquet", index=False)
    con = db.connect()
    n = db.replace_table(con, "backtest_prediction", preds)
    log.info("backtest_prediction: %d satir", n)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 3: modeller\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
