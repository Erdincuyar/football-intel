"""Faz 10: 1X2 sonuc modeli (dc-mr-v3). Dixon-Coles + piyasa reytingi, lojistik regresyon.

Neden bu model? Faz 10 denetiminde (2024/25-2026/27, 4.751 mac, egitimde gorulmemis)
mevcut Dixon-Coles'a gore log loss'u 0,0116 dusurdu, %95 aralik [-0,016, -0,007].
Sut, korner, kart, yorgunluk gibi form ozellikleri ve gradient boosting anlamli katki
vermedi; bu yuzden model bilerek kucuk tutuldu (4 ozellik + lig gostergesi).

Girdiler (hepsi mac oncesi bilinir):
- dc_lo_home, dc_lo_away: DC olasiliklarinin beraberlige gore log-oranlari. Lojistik
  regresyon bunlari yeniden kalibre eder (DC beraberligi eksik tahmin ediyordu).
- d_mr3, d_mr6: piyasa reytingi farki (features/market_rating.py). Bu macin orani DEGIL.
- lig gostergeleri: ligler arasi beraberlik ve ev avantaji farki.

Model JSON olarak saklanir (pickle yerine): okunabilir, surum farki sorun cikarmaz ve
tahmin numpy ile yapilir. Egitim scikit-learn ile yapilir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from footintel.features.market_rating import FEATURES as MR_FEATURES

C_REG = 0.05            # L2 cezasi (kucuk = guclu ceza). 2023/24 dogrulamasinda secildi.
BASE_FEATURES = ["dc_lo_home", "dc_lo_away", *MR_FEATURES]
EPS = 1e-6


def design(df: pd.DataFrame) -> pd.DataFrame:
    """DC olasiliklari + reytinglerden ham ozellik tablosu (lig kolonu korunur)."""
    return pd.DataFrame({
        "league": df["league"].to_numpy(),
        "dc_lo_home": np.log((df["p_home"] + EPS) / (df["p_draw"] + EPS)).to_numpy(),
        "dc_lo_away": np.log((df["p_away"] + EPS) / (df["p_draw"] + EPS)).to_numpy(),
        **{c: df[c].to_numpy(dtype=float) for c in MR_FEATURES},
    }, index=df.index)


def outcome(df: pd.DataFrame) -> np.ndarray:
    gd = (df["home_goals"] - df["away_goals"]).to_numpy()
    return np.where(gd > 0, 0, np.where(gd == 0, 1, 2))


@dataclass
class OutcomeModel:
    leagues: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]
    coef: list[list[float]]
    intercept: list[float]
    n_train: int

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        base = x[BASE_FEATURES].to_numpy(dtype=float)
        base = np.where(np.isnan(base), np.array(self.medians), base)   # eksik -> medyan
        lg = np.column_stack([(x["league"] == g).to_numpy(float) for g in self.leagues])
        m = np.column_stack([base, lg])
        return (m - np.array(self.means)) / np.array(self.scales)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 3) olasilik: ev, beraberlik, deplasman."""
        z = self._matrix(design(df)) @ np.array(self.coef).T + np.array(self.intercept)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OutcomeModel:
        return cls(**d)


def fit(df: pd.DataFrame, leagues: list[str] | None = None) -> OutcomeModel:
    """df: oynanmis maclar; DC olasiliklari, reytingler, gol kolonlari ve league."""
    x = design(df)
    y = outcome(df)
    leagues = sorted(leagues or x["league"].unique())
    base = x[BASE_FEATURES].to_numpy(dtype=float)
    med = np.nanmedian(base, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    base = np.where(np.isnan(base), med, base)
    lg = np.column_stack([(x["league"] == g).to_numpy(float) for g in leagues])
    m = np.column_stack([base, lg])
    mean, scale = m.mean(axis=0), m.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    lr = LogisticRegression(C=C_REG, max_iter=2000).fit((m - mean) / scale, y)
    return OutcomeModel(leagues=leagues, medians=med.tolist(), means=mean.tolist(),
                        scales=scale.tolist(), coef=lr.coef_.tolist(),
                        intercept=lr.intercept_.tolist(), n_train=int(len(y)))


def walk_forward(data: pd.DataFrame, test_seasons: list[str], train_from: str,
                 min_train: int = 200) -> pd.DataFrame:
    """Her test ayinin basinda, o tarihten once oynanmis maclarla yeniden egit ve tahmin et.

    data: match_id, league, season, match_date, home_goals, away_goals, DC olasiliklari
    (walk-forward, yani her biri kendi tarihinde uretilmis) ve reytingler.
    """
    d = data.copy()
    d["match_date"] = pd.to_datetime(d["match_date"])
    pool = d[(d["season"] >= train_from) & d["home_goals"].notna() & d["away_goals"].notna()]
    leagues = sorted(d["league"].unique())
    test = d[d["season"].isin(test_seasons)]
    out = []
    for month, block in test.groupby(test["match_date"].dt.to_period("M")):
        tr = pool[pool["match_date"] < month.start_time]
        if len(tr) < min_train:          # cok az gecmis: tahmin yok (N/A), uydurma yok
            continue
        p = fit(tr, leagues).predict(block)
        out.append(pd.DataFrame({"match_id": block["match_id"].to_numpy(),
                                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}))
    cols = ["match_id", "p_home", "p_draw", "p_away"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
