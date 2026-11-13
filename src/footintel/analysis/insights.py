"""Faz 11: tahmin yazilari (3 satir, nedenleriyle) ve haftanin kuponu.

Yazilar neden kural tabanli? Ucretli bir dil modeli kullanmadan (0 TL kurali) ve uydurma
cumle uretme riski olmadan, her cumle veride olan bir sayiya dayanir. Bir neden icin veri
yoksa o cumle yazilmaz.

Neden adaylari (mac oncesi, sizintisiz; features/form.py ve features/market_rating.py):
- piyasa reytingi: takimin onceki maclarinda piyasanin ona bicitigi beklenen puan
- son 6 mac puani, isabetli sut uretimi ve yenilen isabetli sut
- ic saha / deplasman formu, dinlenme gunu farki
- FotMob varsa: penaltisiz beklenen gol (npxG) formu
Ucuncu satir: FotMob'un sakat/cezali listesinde takimin duzenli oyunculari varsa once o
yazilir. Bu bilgi modelin icinde degildir (Faz 11 testinde canli kullanilabilir bicimi
tahmini iyilestirmedi); okuyucuya uyari olarak verilir.
Her aday, ayni ligdeki tum maclarin dagilimina gore standartlastirilir (z-skoru); tahmini
destekleyen en guclu iki aday yazilir. Ucuncu satir her zaman bir uyaridir.

Kupon: en olasi secimler, yalnizca kalibrasyonu test edilmis pazarlardan (1X2, 2.5, KG),
mac basina en fazla bir, lig basina en fazla iki secim. Birlesik olasilik carpimla verilir
(secimler bagimsiz varsayilir).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.features.form import team_form, team_long
from footintel.features.market_rating import match_ratings

COUPON_SIZE = 3
MAX_PER_LEAGUE = 2
MIN_PROB = 0.60          # bunun altindaki secim kupona girmez; az aday varsa kupon kisa kalir
# Bir nedenin yaziya girmesi icin en kucuk fark (kucuk farklar "neden" sayilmaz):
# piyasa reytingi (beklenen puan), son 6 mac puani, mac basina isabetli sut, npxG.
MIN_GAP = {"mr": 0.3, "pts": 3, "sot": 0.8, "xg": 0.4}
# Eksik duzenli oyuncularin degeri tipik ilk 11 degerinin bu payini gecerse uyari yazilir.
MISSING_WARN = 0.12


# --- Mac oncesi baglam -------------------------------------------------------------------

def _last_points(results: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Her (match_id, takim) icin ONCEKI n macin toplam puani ve mac sayisi."""
    t = team_long(results)
    g = t.groupby("team", sort=False)["points"]
    t["last_pts"] = g.transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    t["last_n"] = g.transform(lambda s: s.shift(1).notna().rolling(n, min_periods=1).sum())
    return t[["match_id", "team", "last_pts", "last_n"]]


def context(results: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Yaklasan maclar icin mac oncesi ozellikler (match_key basina tek satir).

    fixtures: match_key, league, season, match_date, home_team_id, away_team_id.
    Fikstur satirlari oynanmis maclarin sonuna eklenir; ozellikler kaydirilmis (shift) seriler
    oldugu icin fikstur satiri yalnizca kendinden onceki maclari gorur.
    """
    cols = ["league", "season", "match_date", "home_team_id", "away_team_id"]
    fx = fixtures[["match_key", *cols]].rename(columns={"match_key": "match_id"})
    both = pd.concat([results, fx], ignore_index=True)
    both["match_date"] = pd.to_datetime(both["match_date"])
    keys = set(fx["match_id"])
    tf = team_form(both)
    tf = tf[tf["match_id"].isin(keys)]
    lp = _last_points(both)
    tf = tf.merge(lp, on=["match_id", "team"], how="left")
    keep = ["f_sot_for", "f_sot_against", "f_goals_for", "f_goals_against", "v_points",
            "rest_days", "season_games", "promoted", "last_pts", "last_n"]
    h = tf[tf["is_home"] == 1].set_index("match_id")[keep].add_prefix("h_")
    a = tf[tf["is_home"] == 0].set_index("match_id")[keep].add_prefix("a_")
    mr = match_ratings(both).set_index("match_id")
    out = h.join(a, how="outer").join(mr, how="left")
    return out.rename_axis("match_key").reset_index()


# --- Yazilar ------------------------------------------------------------------------------

def _pct(p: float) -> str:
    return f"%{100 * p:.0f}"


def _reasons(r: pd.Series, side: int) -> list[tuple[float, str]]:
    """(guc, cumle) adaylari. side: +1 ev sahibi lehine, -1 deplasman lehine, 0 dengeli."""
    home, away = r["home_team"], r["away_team"]
    fav, dog = (home, away) if side >= 0 else (away, home)
    out = []
    mr = r.get("d_mr6")
    if pd.notna(mr) and side != 0 and np.sign(mr) == side and abs(mr) >= MIN_GAP["mr"]:
        out.append((abs(mr) / 0.5, f"Piyasa son maclarda {fav} takimini {dog} takimindan "
                                   "belirgin bicimde daha guclu fiyatladi."))
    hp, ap, hn, an = (r.get(k) for k in ("h_last_pts", "a_last_pts", "h_last_n", "a_last_n"))
    if pd.notna(hp) and pd.notna(ap) and hn >= 3 and an >= 3:
        d = (hp - ap) * (1 if side >= 0 else -1)
        if (d > 0 or side == 0) and abs(hp - ap) >= MIN_GAP["pts"]:
            fp, dp = (hp, ap) if side >= 0 else (ap, hp)
            fn, dn = (hn, an) if side >= 0 else (an, hn)
            out.append((abs(hp - ap) / 4, f"Son {int(fn)} macta {fav} {int(fp)} puan topladi, "
                                          f"{dog} son {int(dn)} macta {int(dp)} puan."))
    hs, as_ = r.get("h_f_sot_for"), r.get("a_f_sot_for")
    if pd.notna(hs) and pd.notna(as_):
        d = (hs - as_) * (1 if side >= 0 else -1)
        if d >= MIN_GAP["sot"]:
            fs, ds = (hs, as_) if side >= 0 else (as_, hs)
            out.append((abs(hs - as_) / 1.2, f"{fav} son maclarda mac basina {fs:.1f} isabetli "
                                              f"sut uretiyor, {dog} {ds:.1f}."))
    hx, ax = r.get("h_xg_for"), r.get("a_xg_for")
    if pd.notna(hx) and pd.notna(ax) and side != 0 and (hx - ax) * side >= MIN_GAP["xg"]:
        fx_, dx = (hx, ax) if side > 0 else (ax, hx)
        out.append((abs(hx - ax) / 0.6, f"{fav} son maclarda mac basina {fx_:.1f} beklenen gol "
                                        f"(xG, penaltisiz) uretiyor, {dog} {dx:.1f}."))
    hv, av = r.get("h_v_points"), r.get("a_v_points")
    if pd.notna(hv) and pd.notna(av) and side > 0 and hv - av > 0.5:
        out.append(((hv - av) / 0.8, f"{home} ic sahada mac basina {hv:.1f} puan aliyor, "
                                     f"{away} deplasmanda {av:.1f}."))
    hr, ar = r.get("h_rest_days"), r.get("a_rest_days")
    if pd.notna(hr) and pd.notna(ar) and abs(hr - ar) >= 3 and side != 0:
        rested = home if hr > ar else away
        if rested == fav:
            out.append((abs(hr - ar) / 4, f"{rested} rakibinden {int(abs(hr - ar))} gun daha "
                                          "fazla dinlendi."))
    return sorted(out, key=lambda x: -x[0])


def _caution(r: pd.Series, best: str, p_best: float) -> str:
    miss = {k: r.get(f"{k}_missing") for k in ("h", "a")}
    miss = {k: v for k, v in miss.items() if pd.notna(v) and v >= MISSING_WARN
            and isinstance(r.get(f"{k}_absent"), str)}
    if miss:
        k = max(miss, key=miss.get)
        team = r["home_team"] if k == "h" else r["away_team"]
        n = int(r.get(f"{k}_n_absent") or 0)
        more = f" ve {n - 3} oyuncu daha" if n > 3 else ""
        return (f"Dikkat: {team} takiminda duzenli oyunculardan {r[f'{k}_absent']}{more} "
                f"sakat veya cezali (tipik ilk 11 degerinde %{100 * miss[k]:.0f} kayip); bu bilgi "
                "modelin tahmininde yok.")
    mk = r.get("market_p_" + {"H": "home", "D": "draw", "A": "away"}[best])
    if pd.notna(mk) and abs(p_best - mk) >= 0.08:
        side = "iyimser" if p_best > mk else "kotumser"
        return (f"Dikkat: model bu sonucta piyasadan {100 * abs(p_best - mk):.0f} puan daha "
                f"{side}; boyle maclarda model gecmiste daha sik yanildi.")
    if r.get("h_promoted") == 1 or r.get("a_promoted") == 1:
        t = r["home_team"] if r.get("h_promoted") == 1 else r["away_team"]
        return f"Dikkat: {t} lige yeni cikti; bu takim icin gecmis veri az."
    games = min(r.get("h_season_games", 99), r.get("a_season_games", 99))
    if pd.notna(games) and games < 4:
        return "Dikkat: sezonun ilk haftalari; takim gucleri henuz oturmadi."
    if p_best < 0.45:
        return (f"Dikkat: dengeli mac; en olasi sonucun olasiligi yalnizca {_pct(p_best)}, "
                f"beraberlik {_pct(r['p_draw'])}.")
    return f"Dikkat: bu sonucun gerceklesmeme olasiligi {_pct(1 - p_best)}."


def note(r: pd.Series) -> list[str]:
    """Bir mac icin 3 satir: tahmin, nedenler, uyari."""
    probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
    best = max(probs, key=probs.get)
    p_best = probs[best]
    ou = f"2.5 {'Ust' if r['p_over25'] >= 0.5 else 'Alt'} " \
         f"{_pct(max(r['p_over25'], 1 - r['p_over25']))}"
    kg = f"KG {'Var' if r['p_btts'] >= 0.5 else 'Yok'} {_pct(max(r['p_btts'], 1 - r['p_btts']))}"
    who = {"H": f"{r['home_team']} kazanir", "D": "beraberlik",
           "A": f"{r['away_team']} kazanir"}[best]
    line1 = (f"Tahmin: {who} ({_pct(p_best)}); {ou}; {kg}; en olasi skor "
             f"{r.get('pred_score', '?')}.")
    side = {"H": 1, "A": -1, "D": 0}[best]
    rs = _reasons(r, side)
    if rs:
        line2 = " ".join(t for _, t in rs[:2])
    else:
        line2 = ("Iki takimin son maclardaki gostergeleri birbirine yakin; belirgin bir "
                 "ustunluk yok.")
    return [line1, line2, _caution(r, best, p_best)]


def add_market(df: pd.DataFrame, led: pd.DataFrame | None) -> pd.DataFrame:
    """Resmi defterdeki maclar icin tahmin anindaki piyasa olasiliklari (yalnizca uyari icin).

    Defter, 1X2 icin football-data ortalama oranlarindan marji temizlenmis olasiligi
    (market_prob) saklar. Defterde olmayan macta piyasa bilgisi yok, uyari onu kullanmaz.
    """
    out = df.copy()
    for k in ("home", "draw", "away"):
        out[f"market_p_{k}"] = np.nan
    if led is None or led.empty:
        return out
    m = led[led["market"] == "1x2"].pivot_table(index="match_key", columns="selection",
                                                 values="market_prob")
    for k in ("home", "draw", "away"):
        if k in m:
            out[f"market_p_{k}"] = out["match_key"].map(m[k])
    return out


# --- Kupon --------------------------------------------------------------------------------

def candidates(preds: pd.DataFrame) -> pd.DataFrame:
    """Mac basina pazar basina en olasi secim (1X2, 2.5, KG)."""
    rows = []
    for r in preds.itertuples(index=False):
        probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}
        b = max(probs, key=probs.get)
        label = {"H": "1", "D": "X", "A": "2"}[b]
        rows.append((r.match_key, "1x2", label, probs[b]))
        rows.append((r.match_key, "ou25", "2.5 Ust" if r.p_over25 >= 0.5 else "2.5 Alt",
                     max(r.p_over25, 1 - r.p_over25)))
        rows.append((r.match_key, "btts", "KG Var" if r.p_btts >= 0.5 else "KG Yok",
                     max(r.p_btts, 1 - r.p_btts)))
    c = pd.DataFrame(rows, columns=["match_key", "market", "selection", "prob"])
    return c.merge(preds.drop(columns=[x for x in ("market", "selection", "prob")
                                       if x in preds]), on="match_key")


def pick_coupon(preds: pd.DataFrame, size: int = COUPON_SIZE,
                max_per_league: int = MAX_PER_LEAGUE, min_prob: float = MIN_PROB
                ) -> pd.DataFrame:
    """En olasi secimler; mac basina 1, lig basina en fazla max_per_league."""
    c = candidates(preds).sort_values(["prob", "match_key"], ascending=[False, True])
    picked, used, per_league = [], set(), {}
    for r in c.itertuples(index=False):
        if r.prob < min_prob or len(picked) >= size:
            break
        if r.match_key in used or per_league.get(r.league, 0) >= max_per_league:
            continue
        picked.append(r._asdict())
        used.add(r.match_key)
        per_league[r.league] = per_league.get(r.league, 0) + 1
    return pd.DataFrame(picked)


def selection_won(market: str, selection: str, hg: float, ag: float) -> float:
    if pd.isna(hg) or pd.isna(ag):
        return np.nan
    if market == "1x2":
        res = "1" if hg > ag else ("2" if hg < ag else "X")
        return float(res == selection)
    if market == "ou25":
        return float((hg + ag > 2.5) == (selection == "2.5 Ust"))
    return float(((hg > 0) and (ag > 0)) == (selection == "KG Var"))


def coupon_backtest(hist: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Ayni kurali gecmis haftalara uygular. Hafta = maclarin ISO haftasi (tum ligler).

    hist: historical_predictions (tek surum); results: match_results.
    """
    h = hist.copy()
    h["match_key"] = h["match_id"].astype(str)
    h["week"] = pd.to_datetime(h["match_date"]).dt.strftime("%G-W%V")
    res = results.assign(match_key=results["match_id"].astype(str))[
        ["match_key", "home_goals", "away_goals"]]
    rows = []
    for week, g in h.groupby("week"):
        cp = pick_coupon(g)
        if cp.empty:
            continue
        cp = cp.merge(res, on="match_key", how="left")
        won = [selection_won(m, s, a, b) for m, s, a, b in
               cp[["market", "selection", "home_goals", "away_goals"]].itertuples(index=False)]
        if any(pd.isna(w) for w in won):
            continue                      # oynanmamis mac varsa hafta degerlendirilmez
        rows.append({"week": week, "n_picks": len(cp), "picks_won": int(sum(won)),
                     "coupon_won": bool(all(won)), "combined_prob": float(cp["prob"].prod()),
                     "mean_prob": float(cp["prob"].mean())})
    return pd.DataFrame(rows)
