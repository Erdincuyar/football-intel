"""Faz 9 asama 1: kart verisi kapsami ve kart modelinin walk-forward testi.

Zamansal ayrim:
- Ayar (ridge cezasi) secimi: 2023/24 (gecmis olarak 2021/22-2022/23).
- Test: 2024/25 ve 2025/26. Kiyas: onceki sezonlarin lig ortalamasi.
Kapi: sari kartta model, lig ortalamasindan dusuk MAE verir ve farkin %95 bootstrap
guven araligi 0'i icermez. Gecmezse kart tahmini arayuzde "dusuk guven" etiketiyle gosterilir.

Kullanim:
    uv run python pipelines/build_cards.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.config import CLEAN_DIR
from footintel.models import cards as K
from footintel.models.strength import ModelConfig

log = logging.getLogger("build_cards")
TUNE, TEST = ["2324"], ["2425", "2526"]
HISTORY = ["2526", "2627"]      # Mac Merkezi: ayar donemlerinden SONRAKI sezonlar
BASE_CFG = ModelConfig(family="negbin", dixon_coles=False, xi=0.004, penalty=16.0)
PENALTIES = [4.0, 16.0, 64.0]


def run(fd: pd.DataFrame, seasons: list[str], cfg: ModelConfig) -> pd.DataFrame:
    return pd.concat([K.walk_forward(fd, lg, seasons, cfg) for lg in sorted(fd["league"].unique())],
                     ignore_index=True)


def mae_diff_ci(fd: pd.DataFrame, preds: pd.DataFrame, base: pd.DataFrame,
                n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mac bazinda (model MAE - lig ortalamasi MAE); negatif = model daha iyi."""
    d = preds.merge(fd[["match_id", "home_yellow", "away_yellow"]], on="match_id").dropna(
        subset=["home_yellow", "away_yellow"])
    bh, ba = d["league"].map(base["home_yellow"]), d["league"].map(base["away_yellow"])
    em = (np.abs(d["home_yellow"] - d["exp_home_yellow"])
          + np.abs(d["away_yellow"] - d["exp_away_yellow"])) / 2
    eb = (np.abs(d["home_yellow"] - bh) + np.abs(d["away_yellow"] - ba)) / 2
    diff = (em - eb).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    cov = K.coverage(fd)

    tune = {}
    base_tune = K.league_baseline(fd, TUNE[0])
    for pen in PENALTIES:
        ev = K.evaluate(fd, run(fd, TUNE, replace(BASE_CFG, penalty=pen)), base_tune)
        tune[pen] = float((ev["yellow_mae_model"] * ev["matches"]).sum() / ev["matches"].sum())
        log.info("penalty=%s: sari kart MAE %.4f", pen, tune[pen])
    best = min(tune, key=tune.get)
    cfg = replace(BASE_CFG, penalty=best)

    parts, evs, cis = [], [], {}
    for season in TEST:
        base = K.league_baseline(fd, season)
        p = run(fd, [season], cfg)
        parts.append(p)
        evs.append(K.evaluate(fd, p, base).assign(season=season))
        cis[season] = mae_diff_ci(fd, p, base)
    ev = pd.concat(evs).reset_index()
    gate = all(hi < 0 for _, _, hi in cis.values())

    out = {"config": {"family": cfg.family, "xi": cfg.xi, "penalty": cfg.penalty},
           "tuning_mae": tune, "test": ev.round(4).to_dict("records"),
           "mae_diff_ci": {s: [round(v, 4) for v in c] for s, c in cis.items()},
           "gate_passed": gate, "coverage": cov.reset_index().to_dict("records")}
    (CLEAN_DIR / "cards_eval.json").write_text(json.dumps(out, indent=1, default=float))
    # Mac Merkezi icin 2025/26 (test) + 2026/27 (oynanan kisim) tahminleri
    hist = pd.concat([*(p for p in parts if p["season"].iloc[0] in HISTORY),
                      run(fd, [s for s in HISTORY if s not in TEST], cfg)], ignore_index=True)
    hist.to_parquet(CLEAN_DIR / "cards_backtest.parquet", index=False)

    pd.set_option("display.width", 200)
    text = "\n\n".join([
        "Kapsam (oynanmis maclarda dolu orani):\n" + cov.to_string(),
        "Ayar (2023/24, sari kart MAE): " + ", ".join(f"{k}: {v:.4f}" for k, v in tune.items())
        + f" -> penalty={best}",
        "Test (model vs lig ortalamasi):\n" + ev.round(3).to_string(index=False),
        "Sari kart MAE farki (model - lig ort., %95 GA): " + "; ".join(
            f"{s}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for s, (m, lo, hi) in cis.items()),
        f"Kapi: {'GECTI' if gate else 'GECMEDI'}",
    ])
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 9: kart modeli\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
