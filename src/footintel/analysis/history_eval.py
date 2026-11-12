"""Faz 9 asama 3: gecmis tahminleri gercek sonuclarla eslestirme ve degerlendirme.

Akis (uc ayri tablo, hicbiri digerini degistirmez):
    historical_predictions (degismez)  +  match_results (sonuc hatti)
        -> prediction_evaluation (turetilmis; her calismada bastan uretilir)

Dogru/yanlis kurallari (her biri bir "pazar"):
- result: en yuksek olasilikli sonuc (1/X/2) gerceklesen mi?
- score: en olasi skor birebir tuttu mu?
- goals_total, corners_*, yellow_*: beklenen deger yuvarlanir (0.5 yukari). Iki isaret:
  tam isabet (ok) ve +-1 tolerans (ok_pm1). Ikisi ayri gosterilir; toleransli oran tek basina
  basari diye sunulmaz.
- ou25, btts, red: olasilik > 0.5 ise "evet" tahmini.

Kiyas cizgileri (baseline) olmadan bir yuzde anlamsizdir; "%52 dogru" iyi mi kotu mu?
- result ve ou25: kapanis oranlarindan piyasa tahmini (en guclu rakip).
- btts, red ve sayilar: ligin BIR ONCEKI sezon ortalamasi ("hicbir sey bilmeyen" tahminci).
  Onceki sezonu olmayan ligde baseline N/A olur (uydurulmaz).

Eksik veri: tahmin veya sonuc yoksa isaret NA olur (pandas boolean). Ozetlerdeki n yalnizca
degerlendirilebilen maclari sayar; yuzde hep n ile birlikte verilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.analysis.eda import brier_1x2, implied_1x2, log_loss_1x2

RESULT_COLS = ["home_goals", "away_goals", "home_goals_ht", "away_goals_ht", "home_corners",
               "away_corners", "home_yellow", "away_yellow", "home_red", "away_red"]
ODDS_COLS = ["odds_home_close", "odds_draw_close", "odds_away_close", "odds_over25_close",
             "odds_under25_close"]
# Sayi pazarlari: ad -> (tahmin kolonu, gercek kolon(lar)i)
COUNT_MARKETS = {
    "goals_total": ("exp_total_goals", ("home_goals", "away_goals")),
    "corners_home": ("exp_home_corners", ("home_corners",)),
    "corners_away": ("exp_away_corners", ("away_corners",)),
    "corners_total": ("exp_corners", ("home_corners", "away_corners")),
    "yellow_home": ("exp_home_yellow", ("home_yellow",)),
    "yellow_away": ("exp_away_yellow", ("away_yellow",)),
    "yellow_total": ("exp_total_yellow", ("home_yellow", "away_yellow")),
}
BINARY_MARKETS = {"ou25": "p_over25", "btts": "p_btts", "red": "p_red_any"}
MARKETS = ["result", "score", *BINARY_MARKETS, *COUNT_MARKETS]
EVAL_VERSION = "eval-v1"


def match_results(fd: pd.DataFrame) -> pd.DataFrame:
    """Sonuc hatti: yalnizca oynanmis maclar; kimlik + gercek istatistikler + kapanis oranlari."""
    cols = ["match_id", "league", "season", "match_date", "home_team", "away_team",
            *[c for c in RESULT_COLS + ODDS_COLS if c in fd]]
    out = fd.loc[fd["home_goals"].notna() & fd["away_goals"].notna(), cols].copy()
    gd = out["home_goals"] - out["away_goals"]
    out["result"] = np.select([gd > 0, gd < 0], ["H", "A"], "D")
    return out.reset_index(drop=True)


def league_priors(fd: pd.DataFrame) -> pd.DataFrame:
    """Her (lig, sezon) icin BIR ONCEKI sezonun ortalamalari. Sizintisiz baseline."""
    d = match_results(fd)
    d["total_goals"] = d["home_goals"] + d["away_goals"]
    d["btts"] = ((d["home_goals"] > 0) & (d["away_goals"] > 0)).astype(float)
    for a in ("red", "corners", "yellow"):
        if f"home_{a}" not in d:
            d[f"home_{a}"] = d[f"away_{a}"] = np.nan
    reds = d["home_red"] + d["away_red"]
    d["red_any"] = (reds > 0).astype(float).where(reds.notna())
    d["corners_total"] = d["home_corners"] + d["away_corners"]
    d["yellow_total"] = d["home_yellow"] + d["away_yellow"]
    stats = {"total_goals": "base_goals_total", "btts": "base_p_btts",
             "red_any": "base_p_red", "home_corners": "base_corners_home",
             "away_corners": "base_corners_away", "corners_total": "base_corners_total",
             "home_yellow": "base_yellow_home", "away_yellow": "base_yellow_away",
             "yellow_total": "base_yellow_total"}
    per = d.groupby(["league", "season"])[list(stats)].mean().rename(columns=stats)
    per = per.reset_index().sort_values(["league", "season"]).reset_index(drop=True)
    # Sezon kodlari ("2425") sirali; her satira ayni ligin bir onceki sezonunu koy.
    # Arada sezon eksikse (veri yok) "onceki" yanlis olur; bu yuzden ardisiklik kontrolu.
    prev = per.groupby("league")[list(stats.values())].shift(1)
    prev_season = per.groupby("league")["season"].shift(1)
    expected = per["season"].map(lambda s: f"{int(s[:2]) - 1:02d}{int(s[2:]) - 1:02d}")
    prev = prev.where(prev_season == expected)
    return pd.concat([per[["league", "season"]], prev], axis=1)


def _round_half_up(x: pd.Series) -> pd.Series:
    return np.floor(x.astype("float64") + 0.5)


def _flag(pred: pd.Series, actual: pd.Series, ok: pd.Series) -> pd.Series:
    """Tahmin veya sonuc eksikse NA; aksi halde dogru/yanlis."""
    return ok.astype("boolean").mask(pred.isna() | actual.isna())


def evaluate_matches(preds: pd.DataFrame, results: pd.DataFrame,
                     priors: pd.DataFrame) -> pd.DataFrame:
    """Mac basina tahmin/gercek/isaret tablosu. Sonucu henuz olmayan mac DAHIL EDILMEZ."""
    res = results.drop(columns=["league", "season", "match_date", "home_team", "away_team"])
    d = preds.merge(res.rename(columns={"result": "actual_result"}), on="match_id",
                    how="inner")
    d = d.merge(priors, on=["league", "season"], how="left")
    for c in ("exp_home_yellow", "exp_away_yellow"):
        if c not in d:
            d[c] = np.nan
    d["exp_total_yellow"] = d["exp_home_yellow"] + d["exp_away_yellow"]

    # 1X2: model vs piyasa favorisi
    mk = implied_1x2(d) if set(ODDS_COLS[:3]) <= set(d) else pd.DataFrame(
        np.nan, index=d.index, columns=["p_home", "p_draw", "p_away"])
    d[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]] = mk[["p_home", "p_draw", "p_away"]].values
    mk_ok = d[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].notna().all(axis=1)
    fav = np.array(["H", "D", "A"])[np.nan_to_num(
        d[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(), nan=-1).argmax(axis=1)]
    d["result_pred"] = d["pred_result"]
    d["result_actual"] = d["actual_result"]
    d["result_ok"] = _flag(d["result_pred"], d["result_actual"],
                           d["result_pred"] == d["result_actual"])
    d["result_base_pred"] = pd.Series(fav, index=d.index).where(mk_ok)
    d["result_base_ok"] = _flag(d["result_base_pred"], d["result_actual"],
                                d["result_base_pred"] == d["result_actual"])

    # Skor: en olasi skor (baseline yok)
    actual_score = (d["home_goals"].astype("Int64").astype("string") + "-"
                    + d["away_goals"].astype("Int64").astype("string"))
    d["score_pred"], d["score_actual"] = d["pred_score"], actual_score
    d["score_ok"] = _flag(d["score_pred"], d["score_actual"],
                          d["score_pred"] == d["score_actual"])

    # Evet/hayir pazarlari
    total = d["home_goals"] + d["away_goals"]
    if {"odds_over25_close", "odds_under25_close"} <= set(d):
        io, iu = 1 / d["odds_over25_close"], 1 / d["odds_under25_close"]
        d["mkt_p_over25"] = io / (io + iu)
    else:
        d["mkt_p_over25"] = np.nan
    reds = d.get("home_red", pd.Series(np.nan, index=d.index)) + d.get(
        "away_red", pd.Series(np.nan, index=d.index))
    actual_bin = {
        "ou25": (total > 2.5).astype(float),
        "btts": ((d["home_goals"] > 0) & (d["away_goals"] > 0)).astype(float),
        "red": (reds > 0).astype(float).where(reds.notna()),
    }
    base_p = {"ou25": d["mkt_p_over25"], "btts": d.get("base_p_btts"),
              "red": d.get("base_p_red")}
    for m, pcol in BINARY_MARKETS.items():
        p = d[pcol] if pcol in d else pd.Series(np.nan, index=d.index)
        d[f"{m}_pred"] = (p > 0.5).astype(float).where(p.notna())
        d[f"{m}_actual"] = actual_bin[m]
        d[f"{m}_ok"] = _flag(d[f"{m}_pred"], d[f"{m}_actual"],
                             d[f"{m}_pred"] == d[f"{m}_actual"])
        bp = base_p[m] if base_p[m] is not None else pd.Series(np.nan, index=d.index)
        d[f"{m}_base_p"] = bp
        d[f"{m}_base_pred"] = (bp > 0.5).astype(float).where(bp.notna())
        d[f"{m}_base_ok"] = _flag(d[f"{m}_base_pred"], d[f"{m}_actual"],
                                  d[f"{m}_base_pred"] == d[f"{m}_actual"])

    # Sayi pazarlari: tam isabet ve +-1
    for m, (pcol, acols) in COUNT_MARKETS.items():
        exp = d[pcol] if pcol in d else pd.Series(np.nan, index=d.index)
        actual = sum((d[c] if c in d else pd.Series(np.nan, index=d.index)) for c in acols)
        d[f"{m}_exp"] = exp
        d[f"{m}_pred"] = _round_half_up(exp)
        d[f"{m}_actual"] = actual
        diff = (d[f"{m}_pred"] - actual).abs()
        d[f"{m}_ok"] = _flag(d[f"{m}_pred"], actual, diff == 0)
        d[f"{m}_ok_pm1"] = _flag(d[f"{m}_pred"], actual, diff <= 1)
        base = d.get(f"base_{m}", pd.Series(np.nan, index=d.index))
        d[f"{m}_base_exp"] = base
        bpred = _round_half_up(base)
        bdiff = (bpred - actual).abs()
        d[f"{m}_base_ok"] = _flag(bpred, actual, bdiff == 0)
        d[f"{m}_base_ok_pm1"] = _flag(bpred, actual, bdiff <= 1)

    keep = ["match_id", "model_version", "created_at", "league", "season", "matchweek",
            "match_date", "home_team", "away_team", "home_goals", "away_goals",
            "p_home", "p_draw", "p_away", "mkt_p_home", "mkt_p_draw", "mkt_p_away",
            "p_over25", "mkt_p_over25", "p_btts", "p_red_any"]
    keep = [c for c in keep if c in d]
    market_cols = [c for c in d.columns
                   if any(c.startswith(f"{m}_") for m in MARKETS) and c not in keep]
    out = d[keep + market_cols].copy()
    out["eval_version"] = EVAL_VERSION
    return out.sort_values(["league", "season", "matchweek", "match_date",
                            "match_id"]).reset_index(drop=True)


def summarize(ev: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Uzun ozet: grup x pazar -> n, isabet, yuzde (model ve baseline yan yana).

    Yuzde, n == 0 ise NaN'dir (uygulama bunu "N/A" gosterir); asla 0 diye doldurulmaz.
    """
    rows = []
    for key, g in ev.groupby(by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in MARKETS:
            ok = g.get(f"{m}_ok")
            if ok is None:
                continue
            row = dict(zip(by, key, strict=True))
            row.update(market=m, n=int(ok.notna().sum()), hits=int(ok.fillna(False).sum()))
            row["pct"] = row["hits"] / row["n"] if row["n"] else np.nan
            if f"{m}_ok_pm1" in g:
                pm1 = g[f"{m}_ok_pm1"]
                row["hits_pm1"] = int(pm1.fillna(False).sum())
                row["pct_pm1"] = row["hits_pm1"] / row["n"] if row["n"] else np.nan
            base = g.get(f"{m}_base_ok")
            if base is not None:
                row["base_n"] = int(base.notna().sum())
                row["base_pct"] = (base.fillna(False).sum() / row["base_n"]
                                   if row["base_n"] else np.nan)
            if f"{m}_base_ok_pm1" in g:
                b1 = g[f"{m}_base_ok_pm1"]
                row["base_pct_pm1"] = (b1.fillna(False).sum() / row["base_n"]
                                       if row["base_n"] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _brier_bin(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss_bin(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prob_metrics(ev: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Olasilik kalitesi (Brier, log loss) ve sayi hatasi (MAE); model ve kiyas yan yana.

    Karsilastirma adil olsun diye model ve kiyas AYNI maclar uzerinde hesaplanir (ikisinin de
    degeri olan maclar). Kucuk = iyi.
    """
    rows = []
    for key, g in ev.groupby(by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(by, key, strict=True))

        def add(metric, n, model, ref, ref_name, base=base):
            rows.append({**base, "metric": metric, "n": int(n), "model": model,
                         "baseline": ref, "baseline_name": ref_name})

        y = np.column_stack([g["result_actual"] == r for r in "HDA"]).astype(float)
        pm = g[["p_home", "p_draw", "p_away"]].to_numpy(float)
        pk = g[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float)
        ok = ~np.isnan(pm).any(axis=1) & ~np.isnan(pk).any(axis=1)
        if ok.any():
            add("brier_1x2", ok.sum(), brier_1x2(pm[ok], y[ok]),
                brier_1x2(pk[ok], y[ok]), "piyasa")
            add("logloss_1x2", ok.sum(), log_loss_1x2(pm[ok], y[ok]),
                log_loss_1x2(pk[ok], y[ok]), "piyasa")
        for m, pcol, ref_name in (("ou25", "p_over25", "piyasa"),
                                  ("btts", "p_btts", "onceki sezon"),
                                  ("red", "p_red_any", "onceki sezon")):
            if pcol not in g:
                continue
            p, b, yy = (g[pcol].to_numpy(float), g[f"{m}_base_p"].to_numpy(float),
                        g[f"{m}_actual"].to_numpy(float))
            ok = ~np.isnan(p) & ~np.isnan(yy) & ~np.isnan(b)
            if ok.any():
                add(f"brier_{m}", ok.sum(), _brier_bin(p[ok], yy[ok]),
                    _brier_bin(b[ok], yy[ok]), ref_name)
                add(f"logloss_{m}", ok.sum(), _logloss_bin(p[ok], yy[ok]),
                    _logloss_bin(b[ok], yy[ok]), ref_name)
        for m in COUNT_MARKETS:
            e, b, a = (g[f"{m}_exp"].to_numpy(float), g[f"{m}_base_exp"].to_numpy(float),
                       g[f"{m}_actual"].to_numpy(float))
            ok = ~np.isnan(e) & ~np.isnan(a) & ~np.isnan(b)
            if ok.any():
                add(f"mae_{m}", ok.sum(), float(np.mean(np.abs(e[ok] - a[ok]))),
                    float(np.mean(np.abs(b[ok] - a[ok]))), "onceki sezon")
    return pd.DataFrame(rows)
