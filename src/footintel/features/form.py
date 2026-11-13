"""Faz 10: performans tabanli form ozellikleri (sizintisiz).

Neden? Galibiyet/maglubiyet formu gurultuludur: 1-0 kazanilan ama 2 isabetli suta karsi
9 isabetli sut yenen mac "iyi form" degildir. Bu modul her takim icin, YALNIZCA mactan
onceki maclardan hesaplanan, ustel agirlikli hareketli ortalamalar uretir:

- uretim: gol, sut, isabetli sut, korner (attigi ve yedigi)
- rakip kalitesine gore duzeltme: "rakibin normalde yedigi isabetli suttan ne kadar fazla
  urettin?" Zayif rakiplere karsi sisirilmis istatistikleri dengeler.
- ic saha / deplasman ayrimi: ev sahibinin yalnizca ic saha maclari, deplasmanin
  yalnizca deplasman maclari
- disiplin ve oyun tarzi: faul, sari kart, sut basina isabet
- yorgunluk: son macindan bu yana gun ve son 21 gundeki mac sayisi (yalnizca lig maclari;
  kupa/Avrupa maclari veri kaynagimizda yok, bu bir sinirlamadir)

Sizinti kurali: her ozellik `shift(1)` ile bir mac geri kaydirilir, yani mac satirinda
o macin kendi istatistigi asla yer almaz. Testte bu, gelecekteki maclari degistirip
gecmis ozelliklerin degismedigini kontrol ederek dogrulanir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (ozellik, ev sutunu, deplasman sutunu): takim perspektifinde "for" ve "against" olur
STATS = [
    ("goals", "home_goals", "away_goals"),
    ("shots", "home_shots", "away_shots"),
    ("sot", "home_shots_on_target", "away_shots_on_target"),
    ("corners", "home_corners", "away_corners"),
    ("fouls", "home_fouls", "away_fouls"),
    ("yellow", "home_yellow", "away_yellow"),
]
HALFLIFE = 6          # mac; ~son 6 mac agirligin yarisini tasir
VENUE_HALFLIFE = 4    # ic saha / deplasman ozel serileri daha kisa


def team_long(fd: pd.DataFrame) -> pd.DataFrame:
    """Mac basina iki satir: takim perspektifinden attigi (_for) ve yedigi (_against)."""
    base = ["match_id", "league", "season", "match_date"]
    parts = []
    for side, opp, is_home in (("home", "away", 1), ("away", "home", 0)):
        d = fd[base].copy()
        d["team"], d["opp"], d["is_home"] = fd[f"{side}_team_id"], fd[f"{opp}_team_id"], is_home
        for name, h, a in STATS:
            own, oth = (h, a) if is_home else (a, h)
            d[f"{name}_for"] = fd[own] if own in fd else np.nan
            d[f"{name}_against"] = fd[oth] if oth in fd else np.nan
        gf, ga = d["goals_for"], d["goals_against"]
        d["points"] = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        d.loc[gf.isna() | ga.isna(), "points"] = np.nan
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["match_date"] = pd.to_datetime(out["match_date"])
    return out.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)


def _ewm_prev(g: pd.Series, halflife: float) -> pd.Series:
    """Onceki maclarin ustel agirlikli ortalamasi (bu mac haric). Eksik deger atlanir."""
    return g.shift(1).ewm(halflife=halflife, ignore_na=True, min_periods=1).mean()


def _promoted(t: pd.DataFrame) -> pd.Series:
    """Takim bir onceki sezon ayni ligde yoksa 1 (yukselen). Ilk sezon icin NaN.

    Sezonun takim listesi sezon basinda bilinir, bu yuzden sizinti degildir.
    """
    teams = t.groupby(["league", "season"])["team"].agg(set).to_dict()
    seasons = {lg: sorted(s for (l2, s) in teams if l2 == lg) for lg, _ in teams}
    out = []
    for lg, se, tm in zip(t["league"], t["season"], t["team"], strict=True):
        i = seasons[lg].index(se)
        out.append(np.nan if i == 0 else float(tm not in teams[(lg, seasons[lg][i - 1])]))
    return pd.Series(out, index=t.index)


def team_form(fd: pd.DataFrame) -> pd.DataFrame:
    """Her (match_id, team) icin mac oncesi form ozellikleri."""
    t = team_long(fd)
    cols = [f"{n}_{s}" for n, _, _ in STATS for s in ("for", "against")] + ["points"]
    g = t.groupby("team", sort=False)
    out = t[["match_id", "team", "opp", "is_home", "match_date", "league", "season"]].copy()
    for c in cols:
        out[f"f_{c}"] = g[c].transform(lambda s: _ewm_prev(s, HALFLIFE))
    out["season_games"] = t.groupby(["team", "season"], sort=False).cumcount()
    out["promoted"] = _promoted(t)
    out["f_sot_ratio"] = out["f_goals_for"] / out["f_sot_for"].replace(0, np.nan)

    # rakip duzeltmesi: bu macta uretilen - rakibin mac oncesi "yedigi" ortalamasi
    opp_def = out[["match_id", "team", "f_sot_against", "f_goals_against", "f_sot_for",
                   "f_goals_for"]].rename(columns={"team": "opp"})
    m = t[["match_id", "team", "opp", "sot_for", "sot_against", "goals_for",
           "goals_against"]].merge(opp_def, on=["match_id", "opp"], how="left")
    t = t.merge(m.assign(adj_sot_for=m["sot_for"] - m["f_sot_against"],
                         adj_sot_against=m["sot_against"] - m["f_sot_for"],
                         adj_goals_for=m["goals_for"] - m["f_goals_against"],
                         adj_goals_against=m["goals_against"] - m["f_goals_for"])
                [["match_id", "team", "adj_sot_for", "adj_sot_against", "adj_goals_for",
                  "adj_goals_against"]], on=["match_id", "team"], how="left")
    g = t.groupby("team", sort=False)
    for c in ("adj_sot_for", "adj_sot_against", "adj_goals_for", "adj_goals_against"):
        out[f"f_{c}"] = g[c].transform(lambda s: _ewm_prev(s, HALFLIFE))

    # ic saha / deplasman ozel form
    gv = t.groupby(["team", "is_home"], sort=False)
    for c in ("goals_for", "goals_against", "sot_for", "sot_against", "points"):
        out[f"v_{c}"] = gv[c].transform(lambda s: _ewm_prev(s, VENUE_HALFLIFE))

    # yorgunluk / fikstur yogunlugu (yalnizca lig maclari)
    prev = g["match_date"].shift(1)
    out["rest_days"] = (t["match_date"] - prev).dt.days.clip(upper=21)
    out["games_21d"] = g["match_date"].transform(
        lambda s: pd.Series([(s.iloc[:i] > d - pd.Timedelta(days=21)).sum()
                             for i, d in enumerate(s)], index=s.index))
    return out


def match_features(fd: pd.DataFrame, elo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mac satiri basina ev (h_), deplasman (a_) ve fark (d_) ozellikleri."""
    tf = team_form(fd)
    feat = [c for c in tf.columns if c.startswith(("f_", "v_")) or c in (
        "rest_days", "games_21d", "season_games", "promoted")]
    h = tf[tf["is_home"] == 1][["match_id", *feat]].add_prefix("h_").rename(
        columns={"h_match_id": "match_id"})
    a = tf[tf["is_home"] == 0][["match_id", *feat]].add_prefix("a_").rename(
        columns={"a_match_id": "match_id"})
    x = fd[["match_id"]].merge(h, on="match_id", how="left").merge(a, on="match_id", how="left")
    for c in ("f_goals_for", "f_goals_against", "f_sot_for", "f_sot_against", "f_points",
              "f_adj_sot_for", "f_adj_sot_against", "f_adj_goals_for", "f_corners_for"):
        x[f"d_{c}"] = x[f"h_{c}"] - x[f"a_{c}"]
    x["d_venue_points"] = x["h_v_points"] - x["a_v_points"]
    x["d_rest"] = x["h_rest_days"] - x["a_rest_days"]
    if elo is not None:
        e = elo[["match_id", "home_elo_pre", "away_elo_pre"]]
        x = x.merge(e, on="match_id", how="left")
        x["d_elo"] = x["home_elo_pre"] - x["away_elo_pre"]
        x = x.drop(columns=["home_elo_pre", "away_elo_pre"])
    return x
