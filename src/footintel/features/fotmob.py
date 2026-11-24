"""Faz 11: FotMob verisinden mac oncesi ozellikler (xG formu, baski, eksik oyuncu, hoca).

Her ozellik yalnizca ONCEKI maclardan hesaplanir (shift(1) + ussel agirlikli ortalama),
boylece bir macin xG'si veya ilk 11'i kendi tahminine sizmaz.

Ozellikler (ev - deplasman farki olarak, "d_" onekiyle):
- xg_for / xg_against: son maclarda uretilen ve yenilen beklenen gol (npxG, penalti haric;
  penaltiler sansa daha bagli).
- tilt: rakip yari sahadaki paslarin iki takimin toplamindaki payi. Hangi takimin oyunu
  rakip sahaya yikip baski kurdugunu gosterir ("field tilt" yaklasimi; FotMob PPDA vermiyor).
- missing: takimin duzenli oyuncularindan (son 6 macin en az 3'unde ilk 11) mac kadrosunda
  hic olmayanlarin piyasa degeri / takimin tipik ilk 11 degeri. Gecmis maclarda "kadroda
  yok" (ne ilk 11 ne yedek), yaklasan maclarda FotMob'un sakat/cezali listesi kullanilir:
  ikisi de "o mac oynayamayacak duzenli oyuncu" demektir.
- new_coach: takimin mevcut hocasiyla oynadigi mac sayisi 5'ten azsa 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 6
REGULAR_WINDOW, REGULAR_MIN = 6, 3
NEW_COACH_GAMES = 5
FEATURES = ["d_xg_for", "d_xg_against", "d_tilt", "d_missing", "h_new_coach", "a_new_coach"]


# --- football-data <-> FotMob eslemesi -----------------------------------------------------

def team_map(fd: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Takim kimligi eslemesi (lig basina): ayni gun, ayni skorla oynanan maclar.

    Isim benzerligi yerine sonuc eslemesi kullanilir: 'Man United' ile 'Manchester United'
    gibi yazim farklari sorun olmaz. Her FotMob takimi, en cok eslestigi takima baglanir.
    """
    a = fd[["league", "match_date", "home_team_id", "away_team_id", "home_goals",
            "away_goals"]].assign(day=lambda x: pd.to_datetime(x["match_date"]).dt.normalize())
    b = fx[fx["finished"]].assign(day=lambda x: x["kickoff"].dt.normalize())
    m = a.merge(b, left_on=["league", "day", "home_goals", "away_goals"],
                right_on=["league", "day", "fm_home_goals", "fm_away_goals"])
    cols = ["league", "team", "fm_team"]
    pairs = pd.concat([m[["league", "home_team_id", "home_fm_id"]].set_axis(cols, axis=1),
                       m[["league", "away_team_id", "away_fm_id"]].set_axis(cols, axis=1)])
    cnt = pairs.value_counts().rename("n").reset_index()
    cnt = cnt.sort_values("n", ascending=False).drop_duplicates(["league", "fm_team"])
    return cnt.drop_duplicates(["league", "team"])[["league", "team", "fm_team", "n"]]


def link(fd: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """football-data match_id -> FotMob fm_id (lig, takimlar ve +-1 gun)."""
    tm = team_map(fd, fx)
    f = fd[["match_id", "league", "match_date", "home_team_id", "away_team_id"]].copy()
    f["day"] = pd.to_datetime(f["match_date"]).dt.normalize()
    for side in ("home", "away"):
        f = f.merge(tm.rename(columns={"team": f"{side}_team_id", "fm_team": f"{side}_fm_id"})
                    .drop(columns="n"), on=["league", f"{side}_team_id"], how="left")
    g = fx.assign(fday=fx["kickoff"].dt.normalize())[
        ["fm_id", "league", "home_fm_id", "away_fm_id", "fday"]]
    m = f.merge(g, on=["league", "home_fm_id", "away_fm_id"])
    m = m[(m["fday"] - m["day"]).abs() <= pd.Timedelta(days=1)]
    return m.drop_duplicates("match_id")[["match_id", "fm_id"]]


# --- Takim-mac tablosu --------------------------------------------------------------------

def team_long(matches: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Her mac iki satir (ev, deplasman): takimin o macta urettigi ve yedigi."""
    m = matches.merge(fx[["fm_id", "kickoff", "home_fm_id", "away_fm_id"]], on="fm_id")
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "fm_id": m["fm_id"], "kickoff": m["kickoff"], "team": m[f"{side}_fm_id"],
            "is_home": int(side == "home"),
            "npxg_for": m[f"{side}_npxg"], "npxg_against": m[f"{opp}_npxg"],
            "tilt": m[f"{side}_opp_half_passes"]
            / (m[f"{side}_opp_half_passes"] + m[f"{opp}_opp_half_passes"]),
            "coach": m[f"{side}_coach_id"], "xi_value": m[f"{side}_xi_value"],
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["kickoff", "fm_id"],
                                                          ignore_index=True)


def _ewm_prev(s: pd.Series) -> pd.Series:
    return s.shift(1).ewm(halflife=HALF_LIFE, ignore_na=True).mean()


def team_form(long: pd.DataFrame) -> pd.DataFrame:
    """Mac oncesi xG ve baski formu (yalnizca onceki maclar)."""
    t = long.copy()
    g = t.groupby("team", sort=False)
    for c in ("npxg_for", "npxg_against", "tilt", "xi_value"):
        t[f"f_{c}"] = g[c].transform(_ewm_prev)
    # Hoca: bu hocanin bu takimla daha once kac maca ciktigi (toplam, ust uste degil).
    # Toplam sayilir, cunku ceza alan hocanin yerine yardimcisi 1-2 mac cikinca asil hoca
    # geri geldiginde "yeni hoca" sayilmamali. Yaklasan macta hoca bilinmez (NaN): son
    # macin hocasi surer kabul edilir.
    coach = g["coach"].ffill()
    t["coach_games"] = t.groupby([t["team"], coach]).cumcount()
    row = g.cumcount()
    start = row.groupby([t["team"], coach]).transform("min")   # hocanin ilk gorundugu sira
    known = start >= NEW_COACH_GAMES          # verinin basindaki hoca: ne zaman geldigi bilinmez
    t["new_coach"] = np.where(~known & (t["coach_games"] < NEW_COACH_GAMES), np.nan,
                              (t["coach_games"] < NEW_COACH_GAMES).astype(float))
    return t


# --- Eksik duzenli oyuncular -------------------------------------------------------------

def regulars(players: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Her (fm_id, takim) icin ONCEKI son 6 macin en az 3'unde ilk 11'de olan oyuncular."""
    st = players[players["role"] == "starter"][["fm_id", "team_fm_id", "player_id",
                                                "market_value"]]
    order = long[["fm_id", "team", "kickoff"]].rename(columns={"team": "team_fm_id"})
    out = []
    for team, g in order.sort_values("kickoff").groupby("team_fm_id"):
        ids = g["fm_id"].tolist()
        s = st[st["team_fm_id"] == team]
        by = {k: v for k, v in s.groupby("fm_id")}
        for i, fm_id in enumerate(ids):
            prev = [by[j] for j in ids[max(0, i - REGULAR_WINDOW):i] if j in by]
            if len(prev) < REGULAR_MIN:
                continue
            p = pd.concat(prev)
            c = p.groupby("player_id").agg(n=("fm_id", "size"), value=("market_value", "last"))
            c = c[c["n"] >= REGULAR_MIN]
            out.append(pd.DataFrame({"fm_id": fm_id, "team_fm_id": team,
                                     "player_id": c.index, "value": c["value"].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["fm_id", "team_fm_id", "player_id", "value"])


def missing_share(reg: pd.DataFrame, squad: pd.DataFrame, typical: pd.DataFrame,
                  absent: bool) -> pd.DataFrame:
    """Duzenli oyunculardan eksik olanlarin degeri / takimin tipik ilk 11 degeri.

    absent=True: gecmis mac, `squad` o macin kadrosu (ilk 11 + yedek); kadroda olmayan eksik.
    absent=False: yaklasan mac, `squad` sakat/cezali listesi; listede olan eksik.
    """
    key = ["fm_id", "team_fm_id", "player_id"]
    s = squad[key].drop_duplicates().assign(_in=1)
    r = reg.merge(s, on=key, how="left")
    miss = r["_in"].isna() if absent else r["_in"].notna()
    lost = r[miss].groupby(["fm_id", "team_fm_id"])["value"].sum()
    base = reg[["fm_id", "team_fm_id"]].drop_duplicates().set_index(["fm_id", "team_fm_id"])
    base["lost"] = lost
    base = base.fillna({"lost": 0.0}).join(typical.set_index(["fm_id", "team_fm_id"]))
    base["missing"] = (base["lost"] / base["typical"]).clip(0, 1.5)
    return base.reset_index()[["fm_id", "team_fm_id", "missing"]]


def match_features(fx: pd.DataFrame, matches: pd.DataFrame, players: pd.DataFrame,
                   upcoming: pd.DataFrame | None = None, missing_from: str = "absent"
                   ) -> pd.DataFrame:
    """fm_id basina mac oncesi FotMob ozellikleri (bitmis + yaklasan maclar).

    fx: tum fikstur (bitmis ve yaklasan); matches/players: bitmis mac detaylari;
    upcoming: yaklasan maclarin sakat/cezali listesi (players ile ayni kolonlar).
    missing_from: gecmis maclarda eksik oyuncu tanimi. "absent": kadroda (ilk 11 + yedek)
    olmayan duzenli oyuncu; "listed": FotMob'un o mac icin sakat/cezali listesindeki.
    """
    future = fx[~fx["fm_id"].isin(matches["fm_id"]) & ~fx["cancelled"]][["fm_id"]]
    long = team_long(pd.concat([matches, future], ignore_index=True), fx)
    tf = team_form(long)
    reg = regulars(players, long)
    typical = tf[["fm_id", "team", "f_xi_value"]].rename(
        columns={"team": "team_fm_id", "f_xi_value": "typical"})
    past = reg[reg["fm_id"].isin(matches["fm_id"])]
    if missing_from == "absent":
        done = players[players["role"].isin(["starter", "sub"])]
        miss = [missing_share(past, done, typical, True)]
    else:
        miss = [missing_share(past, players[players["role"] == "unavailable"], typical, False)]
    if upcoming is not None and not upcoming.empty:
        miss.append(missing_share(reg[reg["fm_id"].isin(upcoming["fm_id"])], upcoming,
                                  typical, False))
    miss = pd.concat(miss, ignore_index=True).rename(columns={"team_fm_id": "team"})
    tf = tf.merge(miss, on=["fm_id", "team"], how="left")
    keep = ["f_npxg_for", "f_npxg_against", "f_tilt", "missing", "new_coach"]
    h = tf[tf["is_home"] == 1].set_index("fm_id")[keep].add_prefix("h_")
    a = tf[tf["is_home"] == 0].set_index("fm_id")[keep].add_prefix("a_")
    x = h.join(a, how="inner")
    out = pd.DataFrame({
        "d_xg_for": x["h_f_npxg_for"] - x["a_f_npxg_for"],
        "d_xg_against": x["h_f_npxg_against"] - x["a_f_npxg_against"],
        "d_tilt": x["h_f_tilt"] - x["a_f_tilt"],
        "d_missing": x["h_missing"] - x["a_missing"],
        "h_new_coach": x["h_new_coach"], "a_new_coach": x["a_new_coach"],
        "h_missing": x["h_missing"], "a_missing": x["a_missing"],
        "h_xg_for": x["h_f_npxg_for"], "a_xg_for": x["a_f_npxg_for"],
    }, index=x.index)
    return out.reset_index()


def absentees(fx: pd.DataFrame, matches: pd.DataFrame, players: pd.DataFrame,
              upcoming: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Yaklasan maclarda sakat/cezali listesindeki DUZENLI oyuncular (en degerli `top` isim).

    Dondurur: fm_id, team_fm_id, names ("A, B"), n (eksik duzenli oyuncu sayisi).
    """
    cols = ["fm_id", "team_fm_id", "names", "n"]
    if upcoming is None or upcoming.empty:
        return pd.DataFrame(columns=cols)
    future = fx[~fx["fm_id"].isin(matches["fm_id"]) & ~fx["cancelled"]][["fm_id"]]
    long = team_long(pd.concat([matches, future], ignore_index=True), fx)
    reg = regulars(players, long)
    key = ["fm_id", "team_fm_id", "player_id"]
    m = reg.merge(upcoming[[*key, "name"]].drop_duplicates(key), on=key)
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.sort_values("value", ascending=False)
    return (m.groupby(["fm_id", "team_fm_id"])
            .agg(names=("name", lambda s: ", ".join(s.head(top))), n=("name", "size"))
            .reset_index())


def for_fixtures(fd: pd.DataFrame, fixtures: pd.DataFrame, fx: pd.DataFrame,
                 matches: pd.DataFrame, players: pd.DataFrame,
                 upcoming: pd.DataFrame | None) -> pd.DataFrame:
    """Yaklasan maclar (match_key) icin xG formu ve eksik duzenli oyuncular.

    fd: oynanmis maclar (takim eslemesi icin); fixtures: match_key, league, match_date,
    home_team_id, away_team_id. Bu ozellikler modele girmez, yalnizca yazilarda kullanilir.
    """
    rows = fixtures.rename(columns={"match_key": "match_id"})[
        ["match_id", "league", "match_date", "home_team_id", "away_team_id"]]
    ln = link(pd.concat([fd, rows.assign(home_goals=np.nan, away_goals=np.nan)],
                        ignore_index=True), fx)
    ln = ln[ln["match_id"].isin(rows["match_id"])]
    x = match_features(fx, matches, players, upcoming)
    x = ln.merge(x, on="fm_id", how="left")
    ab = absentees(fx, matches, players, upcoming)
    f = fx.set_index("fm_id")
    ab = ab.astype({"fm_id": "int64", "team_fm_id": "int64"})
    for side in ("home", "away"):
        x["team_fm_id"] = x["fm_id"].map(f[f"{side}_fm_id"])
        x = x.merge(ab.rename(columns={"names": f"{side[0]}_absent", "n": f"{side[0]}_n_absent"}),
                    on=["fm_id", "team_fm_id"], how="left").drop(columns="team_fm_id")
    keep = ["h_xg_for", "a_xg_for", "h_missing", "a_missing", "h_absent", "a_absent",
            "h_n_absent", "a_n_absent"]
    return x.rename(columns={"match_id": "match_key"})[["match_key", *keep]]
