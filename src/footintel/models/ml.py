"""Faz 10: 1X2 icin makine ogrenmesi modelleri ve sizintisiz walk-forward degerlendirme.

Neden ayri bir walk-forward? Dixon-Coles her hafta takim guclerini yeniden fit eder.
Burada ise ozellikler (form, Elo, DC olasiliklari) zaten mac oncesi hesaplanmistir; model
yalnizca "bu ozelliklerden sonuca" eslemeyi ogrenir. Bu eslemeyi her test ayinin basinda,
O TARIHTEN ONCE OYNANMIS maclarla yeniden egitiyoruz. Boylece hicbir tahmin, kendisinden
sonraki bir macin sonucunu gormemis olur.

Modeller (hepsi ucretsiz, scikit-learn):
- "lr": cok sinifli lojistik regresyon. Dogrusal, yorumlanabilir, az veriyle kararli.
- "hgb": histogram tabanli gradient boosting (LightGBM'in scikit-learn esdegeri). Dogrusal
  olmayan iliskileri yakalar ama kucuk veride ezberlemeye (overfit) yatkindir.
- "stack": DC olasiliklarinin log-oranlari + form ozellikleri uzerinde lojistik regresyon.
  DC'nin takim gucu bilgisini korur, uzerine formun ekledigini ogrenir.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = np.array([0, 1, 2])          # ev, beraberlik, deplasman

# Ozellik gruplari: ablasyon (cikar-ve-olc) testinde grup grup eklenir.
GROUPS: dict[str, list[str]] = {
    "dc": ["dc_lo_home", "dc_lo_away"],
    "elo": ["d_elo"],
    "form": ["d_f_goals_for", "d_f_goals_against", "d_f_sot_for", "d_f_sot_against",
             "d_f_points"],
    "rakip": ["d_f_adj_sot_for", "d_f_adj_sot_against", "d_f_adj_goals_for"],
    "saha": ["h_v_points", "a_v_points", "h_v_sot_for", "a_v_sot_against",
             "h_v_goals_for", "a_v_goals_for"],
    "yorgunluk": ["h_rest_days", "a_rest_days", "h_games_21d", "a_games_21d"],
    "stil": ["h_f_shots_for", "a_f_shots_for", "h_f_corners_for", "a_f_corners_for",
             "h_f_fouls_for", "a_f_fouls_for", "h_f_yellow_for", "a_f_yellow_for",
             "h_f_sot_ratio", "a_f_sot_ratio"],
    "sezon": ["h_season_games", "a_season_games", "h_promoted", "a_promoted"],
}


def dc_logits(p: pd.DataFrame) -> pd.DataFrame:
    """DC olasiliklarini beraberlige gore log-oran ozelliklerine cevirir."""
    eps = 1e-6
    return pd.DataFrame({
        "dc_lo_home": np.log((p["p_home"] + eps) / (p["p_draw"] + eps)),
        "dc_lo_away": np.log((p["p_away"] + eps) / (p["p_draw"] + eps)),
    }, index=p.index)


def make_lr(c: float = 0.05) -> Callable[[], object]:
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=c, max_iter=2000))


def make_hgb(lr: float = 0.03, depth: int = 3, iters: int = 250,
             l2: float = 1.0) -> Callable[[], object]:
    return lambda: HistGradientBoostingClassifier(
        learning_rate=lr, max_depth=depth, max_iter=iters, l2_regularization=l2,
        min_samples_leaf=80, random_state=0)


def outcome(fd: pd.DataFrame) -> np.ndarray:
    gd = (fd["home_goals"] - fd["away_goals"]).to_numpy()
    return np.where(gd > 0, 0, np.where(gd == 0, 1, 2))


def walk_forward(
    data: pd.DataFrame,
    features: list[str],
    make_model: Callable[[], object],
    test_seasons: list[str],
    train_from: str,
) -> pd.DataFrame:
    """Her test ayi icin: tarih < ay basi olan oynanmis maclarla egit, o ayi tahmin et.

    data: match_id, league, season, match_date, y (0/1/2, oynanmamissa NaN) ve ozellikler.
    Ligler birlikte egitilir; lig farklari icin lig gostergeleri eklenir (one-hot).
    """
    d = data.copy()
    d["match_date"] = pd.to_datetime(d["match_date"])
    lg = pd.get_dummies(d["league"], prefix="lg", dtype=float)
    cols = [*features, *lg.columns]
    d = pd.concat([d, lg], axis=1)
    train_pool = d[(d["season"] >= train_from) & d["y"].notna()]
    out = []
    test = d[d["season"].isin(test_seasons)]
    for month, block in test.groupby(test["match_date"].dt.to_period("M")):
        start = month.start_time
        tr = train_pool[train_pool["match_date"] < start]
        model = make_model()
        model.fit(tr[cols].to_numpy(float), tr["y"].astype(int).to_numpy())
        p = model.predict_proba(block[cols].to_numpy(float))
        out.append(pd.DataFrame({"match_id": block["match_id"].to_numpy(),
                                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
                                 "train_n": len(tr)}))
    return pd.concat(out, ignore_index=True)


def scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Log loss, Brier ve isabet (en olasi sonuc). Kucuk = iyi (isabet haric)."""
    onehot = np.eye(3)[y]
    pc = np.clip((p * onehot).sum(1), 1e-12, 1)
    return {"logloss": float(-np.mean(np.log(pc))),
            "brier": float(np.mean(((p - onehot) ** 2).sum(1))),
            "acc": float(np.mean(p.argmax(1) == y)), "n": int(len(y))}


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mac bazinda kayip farki (a - b) icin ortalama ve %95 guven araligi.

    Ayni maclar uzerinde karsilastirdigimiz icin eslestirilmis (paired) bootstrap: aralik
    tamamen 0'in altindaysa a, b'den anlamli olarak iyidir.
    """
    diff = loss_a - loss_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
