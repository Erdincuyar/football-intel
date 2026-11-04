"""Kesifsel veri analizi (EDA) fonksiyonlari.

Amac: Faz 3'teki modellerin varsayimlarini gercek veriyle sinamak.
- Goller Poisson dagilimina uyuyor mu? (Poisson'da varyans = ortalama)
- Ev ve deplasman golleri bagimsiz mi? Dusuk skorlarda (0-0, 1-1) sapma var mi?
  Varsa Dixon-Coles duzeltmesi gerekir.
- Kornerler asiri dagilimli mi (varyans > ortalama)? Oyleyse Poisson yerine
  Negatif Binom kullanmaliyiz.
- Bahis piyasasi ne kadar iyi tahmin ediyor? Modelimizin gecmesi gereken cita bu.

Fonksiyonlar yalnizca tablo dondurur; cizimler notebook'ta yapilir. Boylece
hesaplar test edilebilir ve ayni sayilar hem notebook'ta hem CI raporunda kullanilir.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def played(fd: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca skoru bilinen maclar; gol kolonlari float (NA -> NaN) olur."""
    df = fd.dropna(subset=["home_goals", "away_goals"]).copy()
    for c in df.columns:
        if str(df[c].dtype) in ("Int64", "Float64"):
            df[c] = df[c].astype("float64")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    return df


def dispersion(df: pd.DataFrame, cols: list[str], by: str = "league") -> pd.DataFrame:
    """Her kolon icin ortalama, varyans ve varyans/ortalama orani.

    Oran ~1 ise Poisson uygun; belirgin sekilde > 1 ise asiri dagilim var.
    """
    rows = []
    for key, g in df.groupby(by):
        for c in cols:
            x = g[c].dropna()
            mean, var = x.mean(), x.var(ddof=1)
            rows.append({by: key, "metric": c, "n": len(x), "mean": mean, "var": var,
                         "var_mean_ratio": var / mean if mean else np.nan})
    return pd.DataFrame(rows)


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_fit(x: pd.Series, max_k: int = 6) -> pd.DataFrame:
    """Gozlenen frekans vs ayni ortalamali Poisson'un bekledigi frekans.

    Son satir "max_k ve ustu"dur, boylece olasiliklar 1'e tamamlanir.
    """
    x = x.dropna().astype(int)
    lam = x.mean()
    rows = []
    for k in range(max_k + 1):
        if k < max_k:
            obs, exp = (x == k).mean(), poisson_pmf(k, lam)
        else:
            obs = (x >= k).mean()
            exp = 1 - sum(poisson_pmf(j, lam) for j in range(max_k))
        rows.append({"goals": f"{k}+" if k == max_k else str(k), "observed": obs,
                     "poisson": exp})
    return pd.DataFrame(rows)


def home_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """Lig-sezon bazinda ev/beraberlik/deplasman yuzdesi ve ortalama gol farki."""
    gd = df["home_goals"] - df["away_goals"]
    out = df.assign(home_win=gd > 0, draw=gd == 0, away_win=gd < 0, goal_diff=gd)
    return (out.groupby(["league", "season"])[["home_win", "draw", "away_win", "goal_diff"]]
            .mean().reset_index())


SCORES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]


def low_score_check(df: pd.DataFrame, by: str = "league") -> pd.DataFrame:
    """Belirli skorlarin gozlenen orani / bagimsizlik varsayiminda beklenen orani.

    Beklenen: P(ev = i) * P(dep = j), iki marjinal dagilim veriden alinir. Oran 1'den
    buyukse o skor bagimsizligin soylediginden sik gorulur.
    Dikkat: lig genelinde takim gucu farklari da bagimliliga yol acar; kesin
    sonuc Faz 3'te takim gucu kontrol edilerek (Dixon-Coles rho) alinacak.
    """
    rows = []
    for key, g in df.groupby(by):
        h, a = g["home_goals"].astype(int), g["away_goals"].astype(int)
        for i, j in SCORES:
            obs = ((h == i) & (a == j)).mean()
            exp = (h == i).mean() * (a == j).mean()
            rows.append({by: key, "score": f"{i}-{j}", "observed": obs, "expected_indep": exp,
                         "ratio": obs / exp if exp else np.nan})
    return pd.DataFrame(rows)


def market_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Bahis pazarlarinin gerceklesme oranlari (lig bazinda)."""
    out = df.assign(
        over15=df["total_goals"] > 1.5,
        over25=df["total_goals"] > 2.5,
        over35=df["total_goals"] > 3.5,
        btts=(df["home_goals"] > 0) & (df["away_goals"] > 0),
    )
    return out.groupby("league")[["total_goals", "over15", "over25", "over35", "btts"]].mean()


def implied_1x2(df: pd.DataFrame, snapshot: str = "close") -> pd.DataFrame:
    """Oranlardan olasilik: 1/oran, sonra toplam 1 olacak sekilde normalize.

    Normalizasyondan onceki toplam - 1 = bahis sirketinin marji (overround).
    Basit (orantili) normalizasyon kullaniyoruz; favori-surpriz yanliligini
    duzelten yontemler (Shin) Faz 5'te.
    """
    cols = [f"odds_{s}_{snapshot}" for s in ("home", "draw", "away")]
    inv = 1 / df[cols].astype("float64")
    total = inv.sum(axis=1, skipna=False)
    probs = inv.div(total, axis=0)
    probs.columns = ["p_home", "p_draw", "p_away"]
    return probs.assign(overround=total - 1)


def outcome_matrix(df: pd.DataFrame) -> np.ndarray:
    gd = (df["home_goals"] - df["away_goals"]).to_numpy()
    return np.column_stack([gd > 0, gd == 0, gd < 0]).astype(float)


def brier_1x2(p: np.ndarray, y: np.ndarray) -> float:
    """Cok sinifli Brier skoru: sum_k (p_k - y_k)^2, maclar uzerinden ortalama. 0-2 arasi."""
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def log_loss_1x2(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Gerceklesen sonuca verilen olasiligin -log'u. Kendinden emin yanlis tahmini cok cezalar."""
    return float(-np.mean(np.log(np.clip(np.sum(p * y, axis=1), eps, 1))))


def market_benchmark(df: pd.DataFrame, snapshot: str = "close") -> pd.DataFrame:
    """Lig bazinda piyasanin Brier, log loss ve marji. Faz 3 modelinin citasi."""
    p = implied_1x2(df, snapshot)
    ok = p[["p_home", "p_draw", "p_away"]].notna().all(axis=1)
    d, p = df[ok], p[ok]
    rows = []
    for league, idx in d.groupby("league").groups.items():
        pp = p.loc[idx, ["p_home", "p_draw", "p_away"]].to_numpy()
        y = outcome_matrix(d.loc[idx])
        uniform = np.full_like(pp, 1 / 3)
        rows.append({"league": league, "n": len(idx), "brier": brier_1x2(pp, y),
                     "log_loss": log_loss_1x2(pp, y), "brier_uniform": brier_1x2(uniform, y),
                     "overround": p.loc[idx, "overround"].mean()})
    return pd.DataFrame(rows)


def calibration_table(prob: pd.Series, outcome: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Tahmin edilen olasilik dilimlerine gore gerceklesme orani.

    Iyi kalibre bir modelde "%40 dedigimiz maclarin ~%40'i" gerceklesir.
    """
    d = pd.DataFrame({"p": prob, "y": outcome.astype(float)}).dropna()
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (d.groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
            .reset_index())


def elo_vs_market(df: pd.DataFrame, elo: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Beklenen ev skoru (G=1, B=0.5, M=0) uzerinden Elo ile piyasayi ayni olcekte karsilastirir."""
    d = df[df["season"].isin(seasons)].merge(elo, on="match_id")
    p = implied_1x2(d)
    d = d.assign(market_expected=p["p_home"] + 0.5 * p["p_draw"]).dropna(
        subset=["market_expected"])
    y = outcome_matrix(d) @ np.array([1.0, 0.5, 0.0])
    return pd.DataFrame({
        "n": [len(d)] * 2,
        "mse": [np.mean((d["home_expected"] - y) ** 2), np.mean((d["market_expected"] - y) ** 2)],
    }, index=["elo", "piyasa (kapanis)"])
