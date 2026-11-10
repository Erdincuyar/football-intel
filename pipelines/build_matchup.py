"""Faz 7: stil gruplari (K-Means) ve taktiksel eslesme kurallarinin geriye donuk testi.

Adimlar:
1. Mac oncesi stil profili (her takimin o mactan onceki son 10 maci).
2. Beklentiler: gol (Dixon-Coles) ve korner (Faz 5 regresyonu), walk-forward.
3. Kurallar: kesif 2022/23-2023/24, dogrulama 2024/25-2025/26.
4. Stil gruplari: takim-sezon profilleri uzerinde K-Means; gruplarin StatsBomb 2015/16
   metrikleriyle (PPDA, pas payi, field tilt...) yorumlanmasi.
5. Guncel profiller (eslesme sayfasi icin).

Kullanim:
    uv run python pipelines/build_matchup.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.analysis import rules as R
from footintel.analysis import tactics as A
from footintel.analysis.eda import played
from footintel.config import CLEAN_DIR, CURRENT_SEASON
from footintel.features import style as ST
from footintel.models import corners as C
from footintel.models.registry import CORNER_CONFIG, GOAL_CONFIG

log = logging.getLogger("build_matchup")
MODEL_SEASONS = ["2223", "2324", "2425", "2526"]
DISCOVERY, CONFIRMATION = ["2223", "2324"], ["2425", "2526"]
CLUSTER_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
RICH = ["pass_share", "ppda", "def_height", "field_tilt", "long_share", "cross_share",
        "xg_share", "npxg_per_shot"]


def load_corner_model() -> C.NBRegression:
    """Faz 5'in kaydettigi korner regresyonu (data/clean/corner_model.json)."""
    d = json.loads((CLEAN_DIR / "corner_model.json").read_text())
    m = C.NBRegression(list(d["features"]))
    m.beta, m.mean, m.std = (np.asarray(d[k], dtype=float) for k in ("beta", "mean", "std"))
    m.log_r = float(d["log_r"])
    return m


def expectations(fd: pd.DataFrame) -> pd.DataFrame:
    """Taraf bazinda modelin mac oncesi beklentisi: gol, yenilen gol, korner."""
    feats = pd.concat([C.match_features(fd, lg, MODEL_SEASONS, GOAL_CONFIG, CORNER_CONFIG)
                       for lg in sorted(fd["league"].unique())], ignore_index=True)
    sides = C.side_table(feats)
    reg = load_corner_model()
    sides["exp_corners"] = reg.predict_mean(sides)
    g = feats.set_index("match_id")
    home = sides["is_home"] == 1
    sides["exp_goals"] = np.where(home, sides["match_id"].map(g["exp_goals_home"]),
                                  sides["match_id"].map(g["exp_goals_away"]))
    sides["exp_goals_against"] = np.where(home, sides["match_id"].map(g["exp_goals_away"]),
                                          sides["match_id"].map(g["exp_goals_home"]))
    return sides[["match_id", "is_home", "exp_goals", "exp_goals_against", "exp_corners"]]


def describe_cluster(center: np.ndarray, names: list[str]) -> str:
    """Merkezin en belirgin iki ozelligi: '+sut payi, -faul' gibi kisa bir etiket."""
    tr = {"shot_share": "sut payi", "sot_ratio": "isabet orani",
          "sot_ratio_against": "rakip isabet orani", "corner_share": "korner payi",
          "fouls": "faul", "fouls_drawn": "cekilen faul"}
    top = np.argsort(-np.abs(center))[:2]
    return ", ".join(f"{'+' if center[i] > 0 else '-'}{tr[names[i]]}" for i in top)


def clusters(sides: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """K-Means; k (3-6) en yuksek siluet skoruyla secilir."""
    ts = ST.team_season_style(sides[sides["season"].isin(CLUSTER_SEASONS)])
    ts = ts[ts["matches"] >= 10]
    zc = [f"z_{c}" for c in ST.CLUSTER_FEATURES]
    x = ts[zc].to_numpy(float)
    scores = {}
    for k in range(3, 7):
        lab, _, _ = ST.kmeans(x, k)
        scores[k] = ST.silhouette(x, lab)
    k = max(scores, key=scores.get)
    lab, centers, _ = ST.kmeans(x, k)
    ts["cluster"] = lab
    info = {"k": k, "silhouette": scores, "features": ST.CLUSTER_FEATURES,
            "centers": centers.round(4).tolist(),
            "labels": [describe_cluster(c, ST.CLUSTER_FEATURES) for c in centers]}
    # Gruplar puanla nasil iliskili? (her grubun ortalama puan/mac'i)
    ppg = sides.groupby(["league", "season", "team"])["points"].mean().rename("ppg")
    ts = ts.merge(ppg.reset_index(), on=["league", "season", "team"], how="left")
    info["ppg"] = ts.groupby("cluster")["ppg"].mean().round(3).to_dict()
    info["size"] = ts["cluster"].value_counts().sort_index().to_dict()
    return ts, info


def statsbomb_view(info: dict) -> pd.DataFrame | None:
    """Ayni stil gruplari 2015/16'da hangi zengin metriklere karsilik geliyor?"""
    path = CLEAN_DIR / "tactics_team_match_1516.parquet"
    if not path.exists():
        return None
    tm = pd.read_parquet(path)
    ts = ST.team_season_style(tm)
    centers = np.asarray(info["centers"])
    x = ts[[f"z_{c}" for c in ST.CLUSTER_FEATURES]].to_numpy(float)
    ts["cluster"] = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    rich = tm.groupby(["league", "team"])[RICH].mean().reset_index()
    out = ts.merge(rich, on=["league", "team"])
    return out.groupby("cluster")[RICH].mean().join(
        out.groupby("cluster").size().rename("n_teams"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    fd = fd.dropna(subset=["home_shots", "away_shots", "home_corners", "away_corners",
                           "home_fouls", "away_fouls"])
    sides = A.football_data_sides(fd)
    prof = ST.add_flags(ST.attach_opponent(ST.pre_match_profile(sides)))
    exp = expectations(fd)
    rows = R.residuals(prof.merge(exp, on=["match_id", "is_home"], how="inner"))
    log.info("Kural testi icin %d taraf satiri", len(rows))

    res = R.evaluate_rules(rows, DISCOVERY, CONFIRMATION)
    res.to_json(CLEAN_DIR / "matchup_rules.json", orient="records", indent=1)

    ts, info = clusters(sides)
    sb = statsbomb_view(info)
    if sb is not None:
        info["statsbomb_1516"] = sb.round(3).reset_index().to_dict("records")
    (CLEAN_DIR / "style_clusters.json").write_text(json.dumps(info, indent=1, default=str))
    ts.to_parquet(CLEAN_DIR / "style_team_season.parquet", index=False)

    cur = ST.current_profile(sides[sides["season"] == CURRENT_SEASON])
    cur.to_parquet(CLEAN_DIR / "style_now.parquet", index=False)
    th = {lg: ST.thresholds(prof, CURRENT_SEASON, lg).reset_index().to_dict("records")
          for lg in sorted(fd["league"].unique())}
    (CLEAN_DIR / "style_thresholds.json").write_text(json.dumps(th, default=float))

    pd.set_option("display.width", 200)
    cols = ["title", "target", "expected_sign", "n_fired_disc", "diff_disc", "q",
            "n_fired_conf", "diff_conf", "p_conf_dir", "kept", "reversed"]
    text = "\n\n".join([
        "Kurallar (takim kosulu tutan satirlarda: rakip kosulu var - yok, ortalama artik farki):\n"
        + res[cols].round(3).to_string(index=False),
        f"Stil gruplari: k={info['k']}, siluet="
        + ", ".join(f"k={k}: {v:.3f}" for k, v in info["silhouette"].items()),
        "\n".join(f"  grup {i}: {lab} | {info['size'].get(i)} takim-sezon | puan/mac "
                  f"{info['ppg'].get(i)}" for i, lab in enumerate(info["labels"])),
        "StatsBomb 2015/16 karsiligi:\n" + (sb.round(3).to_string() if sb is not None
                                           else "yok"),
    ])
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 7: taktiksel eslesme\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
