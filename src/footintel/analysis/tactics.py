"""Faz 6 analizleri: bir taktik metrik ne kadar guvenilir, neyi anlatir, neyle yaklasilir?

Uc soru soruyoruz:

1. Guvenilirlik (reliability): Metrik takimin kalici bir ozelligini mi olcuyor, yoksa
   macdan maca gurultu mu? Her takim-sezonun maclarini tek/cift diye ikiye boleriz;
   iki yarinin takim ortalamalari arasindaki korelasyon yuksekse metrik "stil"dir.
   Yari ornekle olcuyoruz, tam sezona Spearman-Brown duzeltmesiyle cikariyoruz:
   r_tam = 2r / (1 + r).

2. Ongoru gucu: Sezonun ILK yarisindaki metrik, IKINCI yarisindaki puan ortalamasini ne
   kadar iyi tahmin ediyor? Gol farki mi, sut payi mi, xG payi mi daha iyi? Bu, Faz 3'te
   gol modeline neden isabetli sut (SoT) ekledigimizin de bagimsiz bir kontroludur.

3. Yaklasiklik (proxy): Zengin metrikleri (field tilt, xG payi, pas payi) football-data'nin
   basit sayilarindan (sut, isabetli sut, korner, faul) ne kadar geri uretebiliriz?
   Super Lig ve guncel sezonlar icin olay verisi yok; elimizde yalnizca bu sayilar var.
   Dogrulama "bir ligi disarida birak" (leave-one-league-out) ile yapilir: model 3 ligde
   egitilir, 4. ligde test edilir. Boylece baska bir lige (or. Super Lig) tasindiginda
   ne kadar isleyecegine dair durust bir tahmin elde ederiz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _halves(tm: pd.DataFrame, how: str) -> pd.Series:
    order = tm.sort_values(["match_date", "match_id"]).groupby(
        ["league", "season", "team"]).cumcount().reindex(tm.index)
    if how == "oddeven":
        return (order % 2).rename("half")
    n = tm.groupby(["league", "season", "team"])["match_id"].transform("size")
    return (order >= n / 2).astype(int).rename("half")


def reliability(tm: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Tek/cift mac ayrimiyla split-half guvenilirlik (takim-sezon bazinda)."""
    h = _halves(tm, "oddeven")
    means = tm.assign(half=h).groupby(["league", "season", "team", "half"])[metrics].mean()
    a, b = means.xs(0, level="half"), means.xs(1, level="half")
    rows = []
    for m in metrics:
        r = a[m].corr(b[m])
        rows.append({"metric": m, "split_half_r": r, "season_reliability": 2 * r / (1 + r)})
    return pd.DataFrame(rows).set_index("metric").sort_values("season_reliability",
                                                              ascending=False)


def predictive_power(tm: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Ilk yari metrik ortalamasi -> ikinci yari puan ortalamasi (Pearson r, R^2)."""
    t = tm.assign(goal_share=(tm["goals"] / (tm["goals"] + tm["goals_against"])).where(
        tm["goals"] + tm["goals_against"] > 0), goal_diff=tm["goals"] - tm["goals_against"])
    h = _halves(t, "firstsecond")
    keys = ["league", "season", "team"]
    first = t[h == 0].groupby(keys)[[*metrics, "goal_diff", "points"]].mean()
    second = t[h == 1].groupby(keys)["points"].mean().rename("points_2nd")
    d = first.join(second)
    rows = []
    for m in [*metrics, "goal_diff", "points"]:
        r = d[m].corr(d["points_2nd"])
        rows.append({"metric": m, "r": r, "r2": r * r})
    return pd.DataFrame(rows).set_index("metric").sort_values("r2", ascending=False)


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


def proxy_fit(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Hedef metrigi basit sayilardan dogrusal regresyonla yaklasik hesaplar.

    df: takim-sezon ortalamalari (her satir bir takim-sezon, 'league' kolonu ile).
    Donus: tum veriyle fit edilmis katsayilar + leave-one-league-out R^2 (dis lig testi).
    """
    d = df.dropna(subset=[target, *features])
    preds = pd.Series(np.nan, index=d.index)
    for lg in d["league"].unique():
        tr, te = d[d["league"] != lg], d[d["league"] == lg]
        beta = _ols(tr[features].to_numpy(float), tr[target].to_numpy(float))
        preds.loc[te.index] = _predict(beta, te[features].to_numpy(float))
    y = d[target]
    r2_out = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    beta = _ols(d[features].to_numpy(float), y.to_numpy(float))
    return {"target": target, "features": features, "beta": beta.tolist(),
            "r2_leave_league_out": float(r2_out),
            "rmse_leave_league_out": float(np.sqrt(((y - preds) ** 2).mean())),
            "target_sd": float(y.std())}


def apply_proxy(fit: dict, df: pd.DataFrame) -> pd.Series:
    return pd.Series(_predict(np.asarray(fit["beta"]), df[fit["features"]].to_numpy(float)),
                     index=df.index)


def team_season(tm: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["league", "season", "team"]
    out = tm.groupby(keys)[[*metrics, "points"]].mean()
    out["matches"] = tm.groupby(keys).size()
    return out.reset_index()


# --- Dis dogrulama: PPDA'miz Understat'inkiyle ayni siralamayi veriyor mu? -------------

def understat_team_ppda(data: dict) -> pd.DataFrame:
    """Understat lig-sezon JSON'undan takim basina ortalama PPDA (mac bazinda att/def).

    Understat'in tanimi bizimkine yakin ama ayni degil (alan sinirlari ve sayilan savunma
    aksiyonlari farkli olabilir). Bu yuzden seviyeyi degil, takimlarin SIRALAMASINI
    karsilastiriyoruz: iki olcum ayni seyi olcuyorsa siralama tutarli olmali.
    """
    rows = []
    for team in data["teams"].values():
        vals = []
        for m in team["history"]:
            p = m.get("ppda") or {}
            try:
                att, deff = float(p["att"]), float(p["def"])
            except (KeyError, TypeError, ValueError):
                continue
            if deff > 0:
                vals.append(att / deff)
        if vals:
            rows.append({"team_us": team["title"], "ppda_us": float(np.mean(vals)),
                         "matches_us": len(vals)})
    return pd.DataFrame(rows)


def ppda_external_check(tm: pd.DataFrame, understat: dict[str, pd.DataFrame]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """Lig bazinda Spearman siralama korelasyonu: bizim PPDA vs Understat PPDA.

    Doner: (lig ozeti, takim tablosu). Isimler team_matching ile eslenir; eslesmeyen takim
    sayisi ozet tabloda gorunur (sessizce dusurulmez).
    """
    from footintel.clean.team_matching import match_teams

    ours = tm.groupby(["league", "team"])["ppda"].mean().rename("ppda_ours").reset_index()
    teams, summary = [], []
    for lg, us in understat.items():
        o = ours[ours["league"] == lg]
        res = match_teams(us["team_us"].tolist(), {t: t for t in o["team"]})
        us = us.assign(team=[r.team_id for r in res])
        j = o.merge(us.dropna(subset=["team"]), on="team")
        teams.append(j)
        rho = j["ppda_ours"].rank().corr(j["ppda_us"].rank()) if len(j) >= 5 else np.nan
        summary.append({"league": lg, "teams_matched": len(j),
                        "teams_unmatched": int(us["team"].isna().sum()),
                        "spearman": rho, "mean_ours": j["ppda_ours"].mean(),
                        "mean_understat": j["ppda_us"].mean()})
    return pd.DataFrame(summary), (pd.concat(teams, ignore_index=True) if teams
                                   else pd.DataFrame())


# --- football-data.co.uk'tan guncel stil profili --------------------------------------

# Faul sayilari bilerek disarida: 2015/16 tanim kontrolunde StatsBomb, football-data'dan
# %8-17 fazla faul sayiyor (sut, isabetli sut ve kornerde fark %1-3). Faul ile egitilen bir
# model football-data'ya tasininca sistematik olarak kayardi. Paylar ise iki kaynakta ayni.
PROXY_FEATURES = ["shot_share", "sot_share", "corner_share"]
PROXY_TARGETS = ["xg_share", "xt_share", "field_tilt", "pass_share"]


def football_data_sides(fd: pd.DataFrame) -> pd.DataFrame:
    """football-data mac tablosunu takim-mac satirlarina acar (ayni isimlerle)."""
    cols = {"shots": "shots", "shots_on_target": "sot", "corners": "corners",
            "fouls": "fouls", "goals": "goals"}
    out = []
    for side, opp, home in (("home", "away", 1), ("away", "home", 0)):
        t = pd.DataFrame({"match_id": fd["match_id"], "league": fd["league"],
                          "season": fd["season"], "match_date": fd["match_date"],
                          "team": fd[f"{side}_team"], "is_home": home})
        if f"{side}_team_id" in fd:
            t["team_id"] = fd[f"{side}_team_id"].to_numpy()
            t["opp_id"] = fd[f"{opp}_team_id"].to_numpy()
        for src, dst in {**cols, "yellow": "yellows"}.items():
            if f"{side}_{src}" not in fd:
                continue
            t[dst] = pd.to_numeric(fd[f"{side}_{src}"], errors="coerce").to_numpy(float)
            t[f"{dst}_against"] = pd.to_numeric(fd[f"{opp}_{src}"],
                                                errors="coerce").to_numpy(float)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["points"] = np.select([t["goals"] > t["goals_against"],
                             t["goals"] == t["goals_against"]], [3, 1], 0)
    return t


def style_profile(fd: pd.DataFrame, proxies: list[dict], min_matches: int = 5) -> pd.DataFrame:
    """Takim-sezon basit sayilari + bunlardan yaklasik zengin metrikler.

    Paylar once mac bazinda hesaplanip sezon icin ortalanir; proxy modelini StatsBomb
    tarafinda (team_season) tam olarak bu tanimla egittik. Tanim ayni olmazsa model
    baska bir seyi tahmin eder.
    """
    s = football_data_sides(fd).dropna(subset=["shots", "sot", "corners", "fouls"])
    for c, a in (("shot", "shots"), ("sot", "sot"), ("corner", "corners"), ("goal", "goals")):
        tot = s[a] + s[f"{a}_against"]
        s[f"{c}_share"] = (s[a] / tot).where(tot > 0)
    keys = ["league", "season", "team"]
    g = s.groupby(keys)
    out = g[["shot_share", "sot_share", "corner_share", "goal_share", "fouls",
             "fouls_against"]].mean()
    out["matches"] = g.size()
    out["ppg"] = g["points"].mean()
    out = out[out["matches"] >= min_matches].reset_index()
    for p in proxies:
        out[f"{p['target']}_est"] = apply_proxy(p, out)
    return out
