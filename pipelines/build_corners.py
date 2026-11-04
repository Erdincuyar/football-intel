"""Faz 5: Negatif Binom korner regresyonu. Ozellik secimi, test ve faz kapisi.

Zamansal ayrim:
- Ozellikler 2022/23'ten itibaren walk-forward uretilir (2021/22 yalnizca gecmis olarak).
- Ozellik seti secimi: 2022/23 + 2023/24 ile egit, 2024/25'te dogrula.
- Test: secilen seti 2022/23 - 2024/25 ile yeniden egit; 2025/26 (ve 2026/27'nin
  oynanan kismi) uzerinde lig ortalamasi ve Faz 3 modeliyle kiyasla.
Kapi: 2025/26'da toplam korner log loss'u lig ortalamasindan iyi ve farkin %95 bootstrap
guven araligi 0'i icermiyor.

Kullanim:
    uv run python pipelines/build_corners.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.analysis.eda import played
from footintel.config import CLEAN_DIR, LEAGUES
from footintel.models import corners as C
from footintel.models import strength as S
from footintel.models.backtest import binary_log_loss
from footintel.models.registry import CORNER_CONFIG, GOAL_CONFIG

log = logging.getLogger("build_corners")

FEATURE_SEASONS = ["2223", "2324", "2425", "2526", "2627"]
SELECT_TRAIN, SELECT_VAL = ["2223", "2324"], ["2425"]
FINAL_TRAIN, TEST, CURRENT = ["2223", "2324", "2425"], ["2526"], ["2627"]


def league_average_ll(fd: pd.DataFrame, before: str, eval_df: pd.DataFrame) -> pd.Series:
    """Baseline: her lig icin gecmis toplam kornerlere uydurulmus tek bir Negatif Binom.
    Takimlari hic bilmeyen tahminci (moment yontemi: r = ortalama^2 / (varyans - ortalama))."""
    hist = fd[fd["season"] < before].dropna(subset=["home_corners", "away_corners"])
    tot = hist["home_corners"] + hist["away_corners"]
    out = pd.Series(index=eval_df.index, dtype=float)
    for lg, g in eval_df.groupby("league"):
        t = tot[hist["league"] == lg]
        mu, var = t.mean(), t.var()
        log_r = np.log(mu**2 / (var - mu)) if var > mu else np.log(1e4)
        pmf = S.count_pmf(mu, 60, "negbin", log_r)
        k = g["total"].clip(upper=60).astype(int).to_numpy()
        out[g.index] = np.log(np.maximum(pmf[k], 1e-12))
    return out


def side_base_rate(fd: pd.DataFrame, before: str, side: str, line: float) -> pd.Series:
    hist = fd[fd["season"] < before].dropna(subset=[f"{side}_corners"])
    return (hist[f"{side}_corners"] > line).groupby(hist["league"]).mean()


def _bll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mac bazinda ikili log loss (bootstrap icin)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(fd, feats, sides, model: C.NBRegression, seasons: list[str]) -> dict:
    """Model vs lig ortalamasi vs Faz 3 modeli; lig bazinda ve toplam."""
    s = sides[sides["season"].isin(seasons)].dropna(subset=["corners", *model.features])
    ok = s.groupby("match_id")["side"].transform("count") == 2
    s = s[ok]
    pred = C.evaluate_sides(s, model.predict_mean(s), model.log_r)
    f3 = s.merge(feats[["match_id", "corner_log_r"]], on="match_id")
    p3 = C.evaluate_sides(s, np.exp(s["corner_rate"].to_numpy()),
                          float(f3["corner_log_r"].median()))
    res = fd[["match_id", "home_corners", "away_corners"]]
    pred = pred.merge(res, on="match_id")
    pred["total"] = pred["home_corners"] + pred["away_corners"]
    pred["ll_base"] = league_average_ll(fd, seasons[0], pred)
    pred["ll_phase3"] = pred["match_id"].map(p3.set_index("match_id")["ll_corners"])
    pred["p95_phase3"] = pred["match_id"].map(p3.set_index("match_id")["p_corners_over95"])

    over95 = (pred["total"] > 9.5).astype(float)
    base95 = (fd[fd["season"] < seasons[0]].dropna(subset=["home_corners", "away_corners"])
              .assign(o=lambda d: d["home_corners"] + d["away_corners"] > 9.5)
              .groupby("league")["o"].mean())
    rows, cis = [], []
    for lg, g in [*pred.groupby("league"), ("TOPLAM", pred)]:
        y = over95[g.index].to_numpy()
        b95 = g["league"].map(base95).to_numpy()
        row = {"lig": LEAGUES[lg].name if lg in LEAGUES else lg, "n": len(g),
               "toplam_ll_model": -g["ll_corners"].mean(),
               "toplam_ll_faz3": -g["ll_phase3"].mean(),
               "toplam_ll_lig_ort": -g["ll_base"].mean(),
               "u95_ll_model": binary_log_loss(g["p_corners_over95"].to_numpy(), y),
               "u95_ll_faz3": binary_log_loss(g["p95_phase3"].to_numpy(), y),
               "u95_ll_lig_ort": binary_log_loss(b95, y)}
        mean, lo, hi = C.bootstrap_diff(-g["ll_corners"].to_numpy(), -g["ll_base"].to_numpy())
        ci = {"lig": row["lig"], "toplam_fark": mean, "toplam_ci_alt": lo, "toplam_ci_ust": hi,
              "anlamli_iyi": hi < 0}
        for side, line in (("home", 4.5), ("away", 4.5)):
            ys = (g[f"{side}_corners"] > line).to_numpy(float)
            br = g["league"].map(side_base_rate(fd, seasons[0], side, line)).to_numpy()
            pm = g[f"p_{side}_corners_over{line:.1f}".replace(".", "")].to_numpy()
            row[f"{side}45_ll_model"] = binary_log_loss(pm, ys)
            row[f"{side}45_ll_lig_ort"] = binary_log_loss(br, ys)
            # Takim bazli pazar icin de mac bazinda fark ve guven araligi
            m_, lo_, hi_ = C.bootstrap_diff(_bll(pm, ys), _bll(br, ys))
            ci |= {f"{side}45_fark": m_, f"{side}45_ci_ust": hi_}
        rows.append(row)
        cis.append(ci)
    return {"table": pd.DataFrame(rows).set_index("lig"),
            "ci": pd.DataFrame(cis).set_index("lig"), "pred": pred}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    feats = pd.concat([C.match_features(fd, lg, FEATURE_SEASONS, GOAL_CONFIG, CORNER_CONFIG)
                       for lg in sorted(fd["league"].unique())], ignore_index=True)
    sides = C.side_table(feats, fd)
    log.info("Ozellikler: %d mac, %d taraf satiri", len(feats), len(sides))

    sel_rows = []
    for name, cols in C.FEATURE_SETS.items():
        m = C.NBRegression(cols).fit(sides[sides["season"].isin(SELECT_TRAIN)])
        val = sides[sides["season"].isin(SELECT_VAL)].dropna(subset=["corners", *cols])
        val = val[val.groupby("match_id")["side"].transform("count") == 2]
        ev = C.evaluate_sides(val, m.predict_mean(val), m.log_r)
        sel_rows.append({"ozellik_seti": name, "dogrulama_toplam_ll": -ev["ll_corners"].mean(),
                         "yakinsadi": m.success})
    selection = pd.DataFrame(sel_rows).sort_values("dogrulama_toplam_ll").reset_index(drop=True)
    best = selection.iloc[0]["ozellik_seti"]
    model = C.NBRegression(C.FEATURE_SETS[best]).fit(sides[sides["season"].isin(FINAL_TRAIN)])

    test = compare(fd, feats, sides, model, TEST)
    cur = compare(fd, feats, sides, model, CURRENT)
    passed = bool(test["ci"].loc["TOPLAM", "anlamli_iyi"])
    sections = {
        "Ozellik seti secimi (2024/25 dogrulama, toplam korner log loss)": selection.round(4),
        f"Secilen: {best}. Katsayilar (standartlastirilmis ozellikler)":
            model.coefficients().round(4).to_frame("katsayi"),
        "Dagilim parametresi r": f"{np.exp(model.log_r):.2f}",
        "Test 2025/26: log loss (kucuk = iyi)": test["table"].T.round(4),
        "Test 2025/26: model - lig ortalamasi, %95 bootstrap (negatif = model iyi)":
            test["ci"].round(4),
        "2026/27 (oynanan kisim): log loss": cur["table"].T.round(4),
        "2026/27: model - lig ortalamasi, %95 bootstrap": cur["ci"].round(4),
        "Faz 5 kapisi": "GECTI" if passed else "GECMEDI",
    }
    text = "\n\n".join(f"### {k}\n{v if isinstance(v, str) else v.to_string()}"
                       for k, v in sections.items())
    print(text)

    (CLEAN_DIR / "corner_model.json").write_text(json.dumps(
        {**model.to_dict(), "train_seasons": FINAL_TRAIN, "gate_passed": passed}, indent=2))
    pd.concat([test["pred"], cur["pred"]]).to_parquet(
        CLEAN_DIR / "corner_backtest.parquet", index=False)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 5: korner modeli\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
