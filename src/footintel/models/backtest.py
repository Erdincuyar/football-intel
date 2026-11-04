"""Walk-forward backtest: gercek hayattaki gibi, her hafta yalnizca gecmisi bilerek tahmin.

Neden walk-forward? Klasik "rastgele %80 egitim / %20 test" ayrimi futbolda gelecegi
sizdirir: Mart'taki bir maci tahmin ederken Nisan'daki maclarla egitilmis olursunuz.
Burada her 7 gunluk blok icin modeli o bloktan ONCEKI maclarla yeniden fit ediyor,
sonra bloktaki maclari tahmin ediyoruz. Boylece her tahmin, o gun gercekten
yapilabilecek tahminin aynisidir.

Bilinen (sizinti sayilmayan) bilgi: sezonun takim listesi. Sezon basinda hangi
takimlarin yukseldigi ve kume dustugu bellidir.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footintel.analysis.eda import brier_1x2, implied_1x2, log_loss_1x2, outcome_matrix
from footintel.models import strength as S

log = logging.getLogger(__name__)


def season_teams(fd: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for (lg, season), g in fd.groupby(["league", "season"]):
        out[(lg, season)] = set(g["home_team_id"]) | set(g["away_team_id"])
    return out


def promoted_relegated(teams: dict, league: str, season: str) -> tuple[set[str], set[str]]:
    seasons = sorted(s for (lg, s) in teams if lg == league)
    i = seasons.index(season)
    if i == 0:
        return set(), set()
    cur, prev = teams[(league, season)], teams[(league, seasons[i - 1])]
    return cur - prev, prev - cur


def _blocks(dates: pd.Series, step_days: int) -> pd.Series:
    d = pd.to_datetime(dates)
    start = d.min().normalize()
    return start + pd.to_timedelta(((d - start).dt.days // step_days) * step_days, unit="D")


def walk_forward(
    fd: pd.DataFrame,
    league: str,
    seasons: list[str],
    goal_cfg: S.ModelConfig,
    corner_cfg: S.ModelConfig | None = None,
    step_days: int = 7,
) -> pd.DataFrame:
    """Tek lig, verilen sezonlar icin mac bazinda tahminler."""
    lg = fd[fd["league"] == league].copy()
    lg["match_date"] = pd.to_datetime(lg["match_date"])
    teams = season_teams(lg)
    rows = []
    for season in seasons:
        s = lg[lg["season"] == season]
        if s.empty:
            continue
        promoted, relegated = promoted_relegated(teams, league, season)
        gm = cm = None
        for ref, block in s.groupby(_blocks(s["match_date"], step_days)):
            gm = S.fit(lg, ref, goal_cfg, promoted=promoted, relegated=relegated,
                       warm_start=gm)
            if corner_cfg is not None:
                cm = S.fit(lg, ref, corner_cfg, home_col="home_corners",
                           away_col="away_corners", promoted=promoted, relegated=relegated,
                           warm_start=cm)
            for r in block.itertuples(index=False):
                m = S.score_matrix(gm, r.home_team_id, r.away_team_id)
                row = {"match_id": r.match_id, "league": league, "season": season,
                       "fit_date": ref.date(), **S.goal_markets(m)}
                if not (pd.isna(r.home_goals) or pd.isna(r.away_goals)):
                    hg, ag = min(int(r.home_goals), 10), min(int(r.away_goals), 10)
                    row["ll_score"] = float(np.log(max(m[hg, ag], 1e-12)))
                if cm is not None:
                    row.update(S.corner_markets(cm, r.home_team_id, r.away_team_id))
                    if not (pd.isna(r.home_corners) or pd.isna(r.away_corners)):
                        row["ll_corners"] = _corner_total_ll(
                            cm, r.home_team_id, r.away_team_id,
                            int(r.home_corners) + int(r.away_corners))
                rows.append(row)
    return pd.DataFrame(rows)


def _corner_total_ll(model: S.FittedModel, home: str, away: str, total: int,
                     max_k: int = 30) -> float:
    lam, mu = model.rates(home, away)
    fam, lr = model.config.family, model.log_r
    dist = np.convolve(S.count_pmf(lam, max_k, fam, lr), S.count_pmf(mu, max_k, fam, lr))
    return float(np.log(max(dist[min(total, len(dist) - 1)], 1e-12)))


def run_all(fd: pd.DataFrame, seasons: list[str], goal_cfg: S.ModelConfig,
            corner_cfg: S.ModelConfig | None = None) -> pd.DataFrame:
    parts = [walk_forward(fd, lg, seasons, goal_cfg, corner_cfg)
             for lg in sorted(fd["league"].unique())]
    return pd.concat(parts, ignore_index=True)


# --- Degerlendirme ----------------------------------------------------------------------

def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _market_ou25(df: pd.DataFrame) -> pd.Series:
    inv_o = 1 / df["odds_over25_close"].astype("float64")
    inv_u = 1 / df["odds_under25_close"].astype("float64")
    return inv_o / (inv_o + inv_u)


def evaluate(fd: pd.DataFrame, preds: pd.DataFrame, base_rates: pd.DataFrame) -> pd.DataFrame:
    """Lig bazinda model vs piyasa vs baseline. Kucuk = iyi (hepsi kayip fonksiyonu).

    base_rates: lig bazinda egitim donemindeki BTTS ve korner 9.5 ustu oranlari
    (index = league; kolonlar btts, corners_over95). "Hicbir sey bilmeyen" tahminci.
    """
    d = preds.merge(fd.drop(columns=["league", "season"]), on="match_id")
    d = d.dropna(subset=["home_goals", "away_goals"])
    total = d["home_goals"] + d["away_goals"]
    rows = []
    for league, g in d.groupby("league"):
        idx = g.index
        y = outcome_matrix(g)
        p_model = g[["p_home", "p_draw", "p_away"]].to_numpy()
        mk = implied_1x2(g)
        ok = mk[["p_home", "p_draw", "p_away"]].notna().all(axis=1).to_numpy()
        p_mk = mk[["p_home", "p_draw", "p_away"]].to_numpy()
        ou_y = (total[idx] > 2.5).to_numpy(float)
        ou_mk = _market_ou25(g).to_numpy()
        ok_ou = ~np.isnan(ou_mk)
        btts_y = ((g["home_goals"] > 0) & (g["away_goals"] > 0)).to_numpy(float)
        row = {
            "league": league, "n": len(g),
            "1x2_logloss_model": log_loss_1x2(p_model[ok], y[ok]),
            "1x2_logloss_market": log_loss_1x2(p_mk[ok], y[ok]),
            "1x2_brier_model": brier_1x2(p_model[ok], y[ok]),
            "1x2_brier_market": brier_1x2(p_mk[ok], y[ok]),
            "ou25_logloss_model": binary_log_loss(g["p_over25"].to_numpy()[ok_ou], ou_y[ok_ou]),
            "ou25_logloss_market": binary_log_loss(ou_mk[ok_ou], ou_y[ok_ou]),
            "btts_logloss_model": binary_log_loss(g["p_btts"].to_numpy(), btts_y),
            "btts_logloss_base": binary_log_loss(
                np.full(len(g), base_rates.loc[league, "btts"]), btts_y),
            "score_logloss_model": float(-g["ll_score"].mean()),
        }
        if "p_corners_over95" in g:
            c = g.dropna(subset=["home_corners", "away_corners", "p_corners_over95"])
            cy = ((c["home_corners"] + c["away_corners"]) > 9.5).to_numpy(float)
            row["corners95_logloss_model"] = binary_log_loss(c["p_corners_over95"].to_numpy(), cy)
            row["corners95_logloss_base"] = binary_log_loss(
                np.full(len(c), base_rates.loc[league, "corners_over95"]), cy)
            row["corners_total_logloss_model"] = float(-c["ll_corners"].mean())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("league")
    return out


def base_rates(fd: pd.DataFrame, before_season: str) -> pd.DataFrame:
    d = fd[fd["season"] < before_season].dropna(subset=["home_goals", "away_goals"])
    corners = d["home_corners"] + d["away_corners"]
    return pd.DataFrame({
        "btts": ((d["home_goals"] > 0) & (d["away_goals"] > 0)).groupby(d["league"]).mean(),
        "corners_over95": (corners > 9.5).where(corners.notna()).groupby(d["league"]).mean(),
    })
