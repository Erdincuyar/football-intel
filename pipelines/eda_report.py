"""EDA'nin sayisal ozetini yazdirir (notebook'taki tablolarin metin hali).

Notebook grafikleri gosterir; bu betik ayni fonksiyonlarla tablolari CI loguna ve
is ozetine yazar. Boylece notebook'u acmadan da bulgular okunabilir.

Kullanim:
    uv run python pipelines/eda_report.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from footintel.analysis import eda
from footintel.config import CLEAN_DIR

TEST = ["2526"]


def main() -> None:
    fd = eda.played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    fd["total_corners"] = fd["home_corners"] + fd["away_corners"]
    sections: dict[str, pd.DataFrame] = {}

    sections["Mac sayisi (lig x sezon)"] = fd.pivot_table(
        index="league", columns="season", values="match_id", aggfunc="count")
    sections["Dagilim: varyans/ortalama (1 ~ Poisson)"] = eda.dispersion(
        fd, ["home_goals", "away_goals", "total_goals", "total_corners"]
    ).pivot(index="league", columns="metric", values="var_mean_ratio").round(3)
    sections["Ortalamalar"] = eda.dispersion(
        fd, ["home_goals", "away_goals", "total_goals", "total_corners"]
    ).pivot(index="league", columns="metric", values="mean").round(2)
    sections["Toplam gol: gozlenen vs Poisson (tum ligler)"] = eda.poisson_fit(
        fd["total_goals"], max_k=7).set_index("goals").round(4)
    ha = eda.home_advantage(fd)
    sections["Ev sahibi galibiyet orani (lig x sezon)"] = ha.pivot(
        index="league", columns="season", values="home_win").round(3)
    sections["Dusuk skorlar: gozlenen / bagimsiz beklenen"] = eda.low_score_check(fd).pivot(
        index="league", columns="score", values="ratio"
    )[[f"{i}-{j}" for i, j in eda.SCORES]].round(3)
    sections["Ev-deplasman gol korelasyonu"] = (
        fd.groupby("league")[["home_goals", "away_goals"]].corr().unstack().iloc[:, 1]
        .rename("corr").to_frame().round(3))
    sections["Pazar gerceklesme oranlari"] = eda.market_rates(fd).round(3)
    sections["Piyasa citasi (kapanis oranlari, tum sezonlar)"] = eda.market_benchmark(
        fd).set_index("league").round(4)
    sections["Piyasa citasi (acilis/ortalama oranlar)"] = eda.market_benchmark(
        fd, snapshot="avg").set_index("league").round(4)
    p = eda.implied_1x2(fd)
    sections["Kalibrasyon: piyasanin ev galibiyeti olasiligi"] = eda.calibration_table(
        p["p_home"], fd["home_goals"] > fd["away_goals"]).round(3)

    elo_path = CLEAN_DIR / "match_elo.parquet"
    if elo_path.exists():
        sections["Elo vs piyasa (2025/26, beklenen ev skoru MSE)"] = eda.elo_vs_market(
            fd, pd.read_parquet(elo_path), TEST).round(4)

    text = "\n\n".join(f"### {k}\n{v.to_string()}" for k, v in sections.items())
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(f"## EDA ozeti\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
