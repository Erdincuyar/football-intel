"""Canli tahmin: bir ligin modelini bugune kadarki maclarla fit edip verilen fiksturu tahmin eder.

Hem resmi tahmin defteri (predict_upcoming) hem on tahmin (build_preview) ayni fonksiyonu
kullanir; boylece iki yol arasinda model farki olamaz.
"""

from __future__ import annotations

import logging

import pandas as pd

from footintel.config import CURRENT_SEASON
from footintel.models import backtest as B
from footintel.models import corners as C
from footintel.models import strength as S
from footintel.models.registry import CORNER_CONFIG, GOAL_CONFIG

log = logging.getLogger(__name__)


def predict_league(results: pd.DataFrame, fixtures: pd.DataFrame, league: str,
                   ref: pd.Timestamp, corner_reg: C.NBRegression) -> dict[str, tuple]:
    lg = results[results["league"] == league]
    teams = B.season_teams(lg)
    cur = teams.get((league, CURRENT_SEASON), set())
    fx = fixtures[fixtures["league"] == league]
    teams[(league, CURRENT_SEASON)] = cur | set(fx["home_team_id"]) | set(fx["away_team_id"])
    promoted, relegated = B.promoted_relegated(teams, league, CURRENT_SEASON)
    gm = S.fit(lg, ref, GOAL_CONFIG, promoted=promoted, relegated=relegated)
    cm = S.fit(lg, ref, CORNER_CONFIG, home_col="home_corners", away_col="away_corners",
               promoted=promoted, relegated=relegated)
    needs_shots = any(f in corner_reg.features for f in ("shot_rate", "opp_shot_rate"))
    sm = S.fit(lg, ref, C.SHOT_CONFIG, home_col="home_shots", away_col="away_shots",
               promoted=promoted, relegated=relegated) if needs_shots else None
    if not (gm.success and cm.success):
        log.warning("%s: optimizasyon tam yakinsamadi (gol=%s, korner=%s)",
                    league, gm.success, cm.success)
    out = {}
    for r in fx.itertuples(index=False):
        m = S.score_matrix(gm, r.home_team_id, r.away_team_id)
        out[r.match_key] = (S.goal_markets(m), m,
                            C.live_markets(corner_reg, gm, cm, r.home_team_id, r.away_team_id,
                                           shot_model=sm))
    return out
