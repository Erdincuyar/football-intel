"""Faz 9: kart tahminleri (walk-forward).

Sari kart: korner modeliyle ayni yapi. Her takimin "kart gorme" egilimi (hucum parametresi)
ve rakibe kart gordurme egilimi (savunma parametresi), ev sahibi etkisi ve zaman agirlikli
Negatif Binom. Neden ayni yapi? Kart sayisi da takim-mac basina bir sayim; bir takimin
kart egilimi (oyun stili, sertlik) ve rakibinin topu tutup faul cekmesi ikisi de etkiler.

Kirmizi kart: cok seyrek (mac basina ~0.1-0.2). Takim bazli tahmin gurultulu olur. Bu yuzden
takimin gecmis kirmizi kart oranini lig oranina dogru buzuyoruz (shrinkage):
    oran = (takim kirmizilari + K * lig orani) / (takim maclari + K)
ve "macta en az bir kirmizi" olasiligi = 1 - exp(-(ev orani + dep orani)) (Poisson).

Sizinti yok: her hafta (7 gunluk blok) icin model YALNIZCA o bloktan onceki maclarla
fit edilir; backtest.walk_forward ile ayni kural.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.models import strength as S
from footintel.models.backtest import _blocks, promoted_relegated, season_teams

RED_SHRINK_K = 30.0          # takimin kac maci lig oranina esit agirlik tasir
RED_WINDOW_DAYS = 730


def coverage(fd: pd.DataFrame) -> pd.DataFrame:
    """Oynanmis maclarda kart ve korner kolonlarinin dolu olma orani (lig x sezon)."""
    cols = ["home_yellow", "away_yellow", "home_red", "away_red", "home_corners",
            "away_corners"]
    d = fd.dropna(subset=["home_goals", "away_goals"])
    out = d.groupby(["league", "season"])[cols].apply(lambda g: g.notna().mean())
    out["matches"] = d.groupby(["league", "season"]).size()
    return out.round(3)


def red_rates(hist: pd.DataFrame, k: float = RED_SHRINK_K) -> tuple[dict, float]:
    """Gecmis maclardan takim basina buzulmus kirmizi kart orani (mac basina)."""
    h = hist.dropna(subset=["home_red", "away_red"])
    sides = pd.concat([
        pd.DataFrame({"team": h["home_team_id"], "red": h["home_red"]}),
        pd.DataFrame({"team": h["away_team_id"], "red": h["away_red"]})])
    if sides.empty:
        return {}, np.nan
    league_rate = float(sides["red"].mean())
    g = sides.groupby("team")["red"].agg(["sum", "count"])
    rate = (g["sum"] + k * league_rate) / (g["count"] + k)
    return rate.to_dict(), league_rate


def walk_forward(fd: pd.DataFrame, league: str, seasons: list[str], cfg: S.ModelConfig,
                 step_days: int = 7) -> pd.DataFrame:
    """Tek lig: mac basina beklenen sari kart (ev, dep) ve kirmizi kart olasiligi."""
    lg = fd[fd["league"] == league].copy()
    lg["match_date"] = pd.to_datetime(lg["match_date"])
    teams = season_teams(lg)
    rows = []
    for season in seasons:
        s = lg[lg["season"] == season]
        if s.empty:
            continue
        promoted, relegated = promoted_relegated(teams, league, season)
        ym = None
        for ref, block in s.groupby(_blocks(s["match_date"], step_days)):
            ym = S.fit(lg, ref, cfg, home_col="home_yellow", away_col="away_yellow",
                       promoted=promoted, relegated=relegated, warm_start=ym)
            past = lg[(lg["match_date"] < ref)
                      & ((ref - lg["match_date"]).dt.days <= RED_WINDOW_DAYS)]
            rr, lrate = red_rates(past)
            for r in block.itertuples(index=False):
                lam, mu = ym.rates(r.home_team_id, r.away_team_id)
                rh = rr.get(r.home_team_id, lrate)
                ra = rr.get(r.away_team_id, lrate)
                rows.append({"match_id": r.match_id, "league": league, "season": season,
                             "fit_date": ref.date(), "exp_home_yellow": lam,
                             "exp_away_yellow": mu,
                             "p_red_any": float(1 - np.exp(-(rh + ra))),
                             "log_r_yellow": ym.log_r})
    return pd.DataFrame(rows)


def league_baseline(fd: pd.DataFrame, before: str) -> pd.DataFrame:
    """Takimlari bilmeyen tahminci: onceki sezonlarin lig ortalamasi (ev/dep ayri)."""
    h = fd[fd["season"] < before].dropna(subset=["home_yellow", "away_yellow"])
    out = h.groupby("league")[["home_yellow", "away_yellow"]].mean()
    r = fd[fd["season"] < before].dropna(subset=["home_red", "away_red"])
    out["p_red_any"] = ((r["home_red"] + r["away_red"]) > 0).groupby(r["league"]).mean()
    return out


def evaluate(fd: pd.DataFrame, preds: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Lig bazinda MAE (sari kart, takim basina), +-1 isabet, kirmizi kart Brier; model vs
    lig ortalamasi. Kart verisi olmayan maclar disarida (uydurma yok)."""
    d = preds.merge(fd[["match_id", "home_yellow", "away_yellow", "home_red", "away_red"]],
                    on="match_id").dropna(subset=["home_yellow", "away_yellow"])
    rows = []
    for lg, g in d.groupby("league"):
        b = base.loc[lg]
        y = np.r_[g["home_yellow"], g["away_yellow"]]
        m = np.r_[g["exp_home_yellow"], g["exp_away_yellow"]]
        bl = np.r_[np.full(len(g), b["home_yellow"]), np.full(len(g), b["away_yellow"])]
        red = g.dropna(subset=["home_red", "away_red"])
        any_red = ((red["home_red"] + red["away_red"]) > 0).to_numpy(float)
        rows.append({
            "league": lg, "matches": len(g),
            "yellow_mae_model": np.abs(y - m).mean(), "yellow_mae_base": np.abs(y - bl).mean(),
            "yellow_hit1_model": (np.abs(y - np.round(m)) <= 1).mean(),
            "yellow_hit1_base": (np.abs(y - np.round(bl)) <= 1).mean(),
            "red_rate": any_red.mean(),
            "red_brier_model": ((red["p_red_any"].to_numpy() - any_red) ** 2).mean(),
            "red_brier_base": ((b["p_red_any"] - any_red) ** 2).mean(),
        })
    return pd.DataFrame(rows).set_index("league")
