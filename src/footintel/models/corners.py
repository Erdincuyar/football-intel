"""Faz 5: Negatif Binom regresyonu ile korner modeli.

Faz 3'teki korner modeli her takim icin yalnizca "korner hucumu/savunmasi" ogreniyordu ve
cogu ligde lig ortalamasini gecemedi. Burada kornerleri baska bilgilerle de acikliyoruz:

    log(m_taraf) = b0 + b1*ev + b2*korner_gucu + b3*sut_hacmi + b4*rakip_sut_hacmi
                   + b5*gol_ustunlugu

- korner_gucu: Faz 3 korner modelinin bu taraf icin log beklenen korner sayisi.
- sut_hacmi: Toplam sut sayisi uzerine kurulan ayni tip takim gucu modelinin log beklenen
  sutu. Korner cogunlukla engellenen sut ve ortalardan gelir; sut hacmi baski gucunun
  korner'den daha az gurultulu bir olcusudur.
- gol_ustunlugu: Gol modelinin beklenen gol farki (bu taraf - rakip). "Mac senaryosu":
  favori takim topa daha cok sahip olur, ama geride kalan takim da skor icin yuklenir.
  Isaretini ve buyuklugunu veri belirler.

Tum ozellikler walk-forward ile uretilir: her hafta, o haftadan ONCEKI maclarla fit
edilmis modellerden. Boylece regresyonun egitim verisi de canli tahmindeki bilgiyle
ayni kosullarda olusur.

Iki tarafin kornerleri bagimsiz kabul edilir; toplam dagilim konvolusyonla bulunur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from footintel.models import backtest as B
from footintel.models import strength as S

SHOT_CONFIG = S.ModelConfig(family="negbin", dixon_coles=False, xi=0.004, penalty=8.0)

FEATURE_SETS: dict[str, list[str]] = {
    "korner_gucu": ["is_home", "corner_rate"],
    "+sut": ["is_home", "corner_rate", "shot_rate", "opp_shot_rate"],
    "+senaryo": ["is_home", "corner_rate", "supremacy"],
    "tam": ["is_home", "corner_rate", "shot_rate", "opp_shot_rate", "supremacy"],
}


def match_features(fd: pd.DataFrame, league: str, seasons: list[str],
                   goal_cfg: S.ModelConfig, corner_cfg: S.ModelConfig,
                   shot_cfg: S.ModelConfig = SHOT_CONFIG, step_days: int = 7) -> pd.DataFrame:
    """Mac basina maç oncesi ozellikler (bir lig). Her 7 gunluk blokta modeller yeniden fit."""
    lg = fd[fd["league"] == league].copy()
    lg["match_date"] = pd.to_datetime(lg["match_date"])
    teams = B.season_teams(lg)
    rows = []
    for season in seasons:
        s = lg[lg["season"] == season]
        if s.empty:
            continue
        prom, rel = B.promoted_relegated(teams, league, season)
        gm = cm = sm = None
        for ref, block in s.groupby(B._blocks(s["match_date"], step_days)):
            kw = {"promoted": prom, "relegated": rel}
            gm = S.fit(lg, ref, goal_cfg, warm_start=gm, **kw)
            cm = S.fit(lg, ref, corner_cfg, home_col="home_corners", away_col="away_corners",
                       warm_start=cm, **kw)
            sm = S.fit(lg, ref, shot_cfg, home_col="home_shots", away_col="away_shots",
                       warm_start=sm, **kw)
            for r in block.itertuples(index=False):
                gh, ga = gm.rates(r.home_team_id, r.away_team_id)
                ch, ca = cm.rates(r.home_team_id, r.away_team_id)
                sh, sa = sm.rates(r.home_team_id, r.away_team_id)
                rows.append({
                    "match_id": r.match_id, "league": league, "season": season,
                    "exp_goals_home": gh, "exp_goals_away": ga,
                    "corner_rate_home": np.log(ch), "corner_rate_away": np.log(ca),
                    "shot_rate_home": np.log(sh), "shot_rate_away": np.log(sa),
                    "corner_log_r": cm.log_r,
                })
    return pd.DataFrame(rows)


def side_table(feat: pd.DataFrame, fd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mac bazli ozellikleri taraf bazli satirlara acar (ev ve deplasman ayri satir)."""
    out = []
    for side, opp, is_home in (("home", "away", 1.0), ("away", "home", 0.0)):
        t = pd.DataFrame({
            "match_id": feat["match_id"], "league": feat["league"], "season": feat["season"],
            "side": side, "is_home": is_home,
            "corner_rate": feat[f"corner_rate_{side}"],
            "shot_rate": feat[f"shot_rate_{side}"],
            "opp_shot_rate": feat[f"shot_rate_{opp}"],
            "supremacy": feat[f"exp_goals_{side}"] - feat[f"exp_goals_{opp}"],
        })
        if fd is not None:
            t = t.merge(fd[["match_id", f"{side}_corners"]].rename(
                columns={f"{side}_corners": "corners"}), on="match_id", how="left")
        out.append(t)
    return pd.concat(out, ignore_index=True)


@dataclass
class NBRegression:
    features: list[str]
    beta: np.ndarray | None = None
    log_r: float = np.log(10.0)
    mean: np.ndarray | None = None   # ozellik standardizasyonu (yorumlanabilirlik + hiz)
    std: np.ndarray | None = None
    success: bool = False

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=float)
        x = (x - self.mean) / self.std
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, df: pd.DataFrame, ridge: float = 1e-3) -> NBRegression:
        df = df.dropna(subset=[*self.features, "corners"])
        raw = df[self.features].to_numpy(dtype=float)
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0)
        self.std[self.std == 0] = 1.0
        x, y = self._x(df), df["corners"].to_numpy(dtype=float)
        k = x.shape[1]

        def obj(p):
            beta, log_r = p[:k], p[k]
            r = np.exp(log_r)
            m = np.exp(x @ beta)
            ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
            d_eta = r * (y - m) / (r + m)
            d_logr = r * (digamma(y + r) - digamma(r) + np.log(r / (r + m))
                          + 1 - (r + y) / (r + m))
            nll = -ll.sum() + ridge * np.sum(beta[1:] ** 2)
            g = np.empty(k + 1)
            g[:k] = -(x.T @ d_eta)
            g[1:k] += 2 * ridge * beta[1:]
            g[k] = -d_logr.sum()
            return nll, g

        p0 = np.zeros(k + 1)
        p0[0] = np.log(max(y.mean(), 1e-3))
        p0[k] = np.log(10.0)
        bounds = [(None, None)] * k + [(np.log(0.5), np.log(1e4))]
        res = minimize(obj, p0, jac=True, method="L-BFGS-B", bounds=bounds)
        self.beta, self.log_r, self.success = res.x[:k], float(res.x[k]), bool(res.success)
        self._obj = obj
        return self

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self._x(df) @ self.beta)

    def to_dict(self) -> dict:
        return {"features": self.features, "beta": self.beta.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(), "log_r": self.log_r}

    @classmethod
    def from_dict(cls, d: dict) -> NBRegression:
        m = cls(list(d["features"]))
        m.beta, m.mean, m.std = (np.asarray(d[k], dtype=float) for k in ("beta", "mean", "std"))
        m.log_r, m.success = float(d["log_r"]), True
        return m

    def coefficients(self) -> pd.Series:
        return pd.Series(self.beta, index=["sabit", *self.features])


def total_distribution(m_home: float, m_away: float, log_r: float,
                       max_k: int = 30) -> np.ndarray:
    ph = S.count_pmf(m_home, max_k, "negbin", log_r)
    pa = S.count_pmf(m_away, max_k, "negbin", log_r)
    return np.convolve(ph, pa)


def evaluate_sides(sides: pd.DataFrame, m: np.ndarray, log_r: float) -> pd.DataFrame:
    """Mac bazinda: toplam korner log olasiligi ve pazar olasiliklari (sides: ev+dep satirlari)."""
    d = sides.assign(m=m)
    h = d[d["side"] == "home"].set_index("match_id")
    a = d[d["side"] == "away"].set_index("match_id")
    a = a.loc[h.index]
    rows = []
    for mid in h.index:
        mh, ma = h.at[mid, "m"], a.at[mid, "m"]
        dist = total_distribution(mh, ma, log_r)
        k = np.arange(len(dist))
        row = {"match_id": mid, "league": h.at[mid, "league"], "season": h.at[mid, "season"],
               "exp_home_corners": mh, "exp_away_corners": ma}
        for line in (8.5, 9.5, 10.5, 11.5):
            row[f"p_corners_over{line:.1f}".replace(".", "")] = float(dist[k > line].sum())
        for side, mean in (("home", mh), ("away", ma)):
            pmf = S.count_pmf(mean, 30, "negbin", log_r)
            kk = np.arange(len(pmf))
            for line in (3.5, 4.5, 5.5):
                row[f"p_{side}_corners_over{line:.1f}".replace(".", "")] = float(
                    pmf[kk > line].sum())
        hc, ac = h.at[mid, "corners"], a.at[mid, "corners"]
        if not (pd.isna(hc) or pd.isna(ac)):
            row["ll_corners"] = float(np.log(max(dist[min(int(hc + ac), len(dist) - 1)],
                                                 1e-12)))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """Ortalama(a - b) ve %95 guven araligi (mac bazinda yeniden ornekleme).

    Neden? Iki modelin log loss farki kucuk olabilir; bu fark sans eseri mi, yoksa
    gercek mi? Maclari tekrar tekrar rastgele secip farki yeniden hesaplariz. Aralik
    0'i icermiyorsa fark istatistiksel olarak anlamlidir.
    """
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def live_markets(model: NBRegression, goal_model: S.FittedModel, corner_model: S.FittedModel,
                 home: str, away: str, shot_model: S.FittedModel | None = None
                 ) -> dict[str, float]:
    """Canli tahmin: tek mac icin toplam ve takim bazli korner pazarlari."""
    gh, ga = goal_model.rates(home, away)
    ch, ca = corner_model.rates(home, away)
    sh, sa = shot_model.rates(home, away) if shot_model is not None else (np.nan, np.nan)
    if any(f in model.features for f in ("shot_rate", "opp_shot_rate")) and shot_model is None:
        raise ValueError("Secilen korner modeli sut ozelligi istiyor; shot_model verilmeli")
    rows = pd.DataFrame({
        "is_home": [1.0, 0.0], "corner_rate": [np.log(ch), np.log(ca)],
        "supremacy": [gh - ga, ga - gh],
        "shot_rate": [np.log(sh), np.log(sa)], "opp_shot_rate": [np.log(sa), np.log(sh)],
    })
    mh, ma = model.predict_mean(rows)
    sides = pd.DataFrame({"match_id": ["x", "x"], "league": "", "season": "",
                          "side": ["home", "away"], "corners": [np.nan, np.nan]})
    out = evaluate_sides(sides, np.array([mh, ma]), model.log_r).iloc[0].to_dict()
    return {k: float(v) for k, v in out.items() if k.startswith(("p_", "exp_"))}
