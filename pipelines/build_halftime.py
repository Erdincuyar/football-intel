"""Faz 9: ilk yari sonucu modelinin walk-forward testi (ilk golu atan takim yerine).

Zamansal ayrim kart modeliyle ayni:
- Ayar (ridge cezasi): 2023/24. Test: 2024/25 ve 2025/26.
- Kiyas: onceki sezonlarin lig ilk yari 1/X/2 frekansi. Olcu: Brier (kucuk = iyi).
Kapi: iki test sezonunda da Brier farkinin (model - kiyas) %95 GA ust siniri < 0.
Gecmezse arayuz ilk yari tahminini "dusuk guven" etiketiyle gosterir.

Kullanim:
    uv run python pipelines/build_halftime.py
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
from footintel.models import halftime as HT
from footintel.models.strength import ModelConfig

log = logging.getLogger("build_halftime")
TUNE, TEST = ["2324"], ["2425", "2526"]
HISTORY = ["2526", "2627"]
BASE_CFG = ModelConfig(family="poisson", dixon_coles=False, xi=0.004, penalty=16.0)
PENALTIES = [4.0, 16.0, 64.0]


def run(fd: pd.DataFrame, seasons: list[str], cfg: ModelConfig) -> pd.DataFrame:
    parts = [HT.walk_forward(fd, lg, seasons, cfg) for lg in sorted(fd["league"].unique())]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def diff_ci(rows: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    diff = (rows["brier_model"] - rows["brier_base"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    cov = (fd.dropna(subset=["home_goals"])
           .assign(ok=lambda d: d["home_goals_ht"].notna() & d["away_goals_ht"].notna())
           .groupby(["league", "season"])["ok"].mean().unstack().round(3))

    tune = {}
    freq_tune = HT.league_frequencies(fd, TUNE[0])
    for pen in PENALTIES:
        r = HT.brier_rows(fd, run(fd, TUNE, replace(BASE_CFG, penalty=pen)), freq_tune)
        tune[pen] = float(r["brier_model"].mean())
        log.info("penalty=%s: ilk yari Brier %.4f", pen, tune[pen])
    best = min(tune, key=tune.get)
    cfg = replace(BASE_CFG, penalty=best)

    parts, tables, cis = [], [], {}
    for season in TEST:
        p = run(fd, [season], cfg)
        parts.append(p)
        r = HT.brier_rows(fd, p, HT.league_frequencies(fd, season))
        tables.append(r.groupby("league")[["brier_model", "brier_base", "hit_model"]].mean()
                      .assign(n=r.groupby("league").size(), season=season))
        cis[season] = diff_ci(r)
    ev = pd.concat(tables).reset_index()
    gate = all(hi < 0 for _, _, hi in cis.values())

    out = {"config": {"family": cfg.family, "xi": cfg.xi, "penalty": cfg.penalty},
           "tuning_brier": tune, "test": ev.round(4).to_dict("records"),
           "brier_diff_ci": {s: [round(v, 4) for v in c] for s, c in cis.items()},
           "gate_passed": gate}
    (CLEAN_DIR / "halftime_eval.json").write_text(json.dumps(out, indent=1, default=float))
    hist = pd.concat([*(p for p in parts if p["season"].iloc[0] in HISTORY),
                      run(fd, [s for s in HISTORY if s not in TEST], cfg)], ignore_index=True)
    hist.to_parquet(CLEAN_DIR / "halftime_backtest.parquet", index=False)

    pd.set_option("display.width", 200)
    text = "\n\n".join([
        "Ilk yari skoru kapsami (oynanmis maclarda):\n" + cov.to_string(),
        "Ayar (2023/24, Brier): " + ", ".join(f"{k}: {v:.4f}" for k, v in tune.items())
        + f" -> penalty={best}",
        "Test (model vs lig frekansi):\n" + ev.round(4).to_string(index=False),
        "Brier farki (model - kiyas, %95 GA): " + "; ".join(
            f"{s}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for s, (m, lo, hi) in cis.items()),
        f"Kapi: {'GECTI' if gate else 'GECMEDI'}",
    ])
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 9: ilk yari modeli\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
