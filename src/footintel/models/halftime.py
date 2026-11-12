"""Faz 9: ilk yari sonucu (1/X/2) tahmini. "Ilk golu atan takim"in yerine.

Neden ilk yari? football-data.co.uk gol dakikasi vermez; "ilk golu kim atti" hesaplanamaz.
Ilk yari skoru (HTHG/HTAG) ise her macta var ve gercek veriyle olculebilir.

Model: tam mac gol modeliyle ayni yapi (takim hucum/savunma + ev avantaji), ama ilk yari
golleri uzerinde fit edilir. Ilk yari golleri azdir (mac basina ~1.2); beraberlik cok
sik (~%40) oldugu icin en olasi sonuc cogu macta X cikar. Bu bir hata degil, verinin
kendisi; basariyi oran degil Brier skoruyla, lig frekansina karsi olcuyoruz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.models import strength as S
from footintel.models.backtest import _blocks, promoted_relegated, season_teams

HT_COLS = ["p_ht_home", "p_ht_draw", "p_ht_away", "pred_ht_result"]


def walk_forward(fd: pd.DataFrame, league: str, seasons: list[str], cfg: S.ModelConfig,
                 step_days: int = 7) -> pd.DataFrame:
    lg = fd[fd["league"] == league].copy()
    lg["match_date"] = pd.to_datetime(lg["match_date"])
    # Ilk yari skoru olmayan maclar fit'e girmez (uydurma yok)
    fit_data = lg.dropna(subset=["home_goals_ht", "away_goals_ht"])
    teams = season_teams(lg)
    rows = []
    for season in seasons:
        s = lg[lg["season"] == season]
        if s.empty:
            continue
        promoted, relegated = promoted_relegated(teams, league, season)
        model = None
        for ref, block in s.groupby(_blocks(s["match_date"], step_days)):
            model = S.fit(fit_data, ref, cfg, home_col="home_goals_ht",
                          away_col="away_goals_ht", promoted=promoted, relegated=relegated,
                          warm_start=model)
            for r in block.itertuples(index=False):
                g = S.goal_markets(S.score_matrix(model, r.home_team_id, r.away_team_id,
                                                  max_goals=8))
                rows.append({"match_id": r.match_id, "league": league, "season": season,
                             "p_ht_home": g["p_home"], "p_ht_draw": g["p_draw"],
                             "p_ht_away": g["p_away"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        probs = out[["p_ht_home", "p_ht_draw", "p_ht_away"]].to_numpy()
        out["pred_ht_result"] = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
    return out


def ht_outcome(fd: pd.DataFrame) -> pd.Series:
    d = fd["home_goals_ht"] - fd["away_goals_ht"]
    return pd.Series(np.select([d > 0, d < 0], ["H", "A"], "D"), index=fd.index).where(
        d.notna())


def league_frequencies(fd: pd.DataFrame, before: str) -> pd.DataFrame:
    """Kiyas: onceki sezonlarda ligin ilk yari 1/X/2 frekanslari."""
    h = fd[fd["season"] < before].dropna(subset=["home_goals_ht", "away_goals_ht"])
    y = ht_outcome(h)
    return pd.get_dummies(y).groupby(h["league"]).mean().reindex(
        columns=["H", "D", "A"], fill_value=0.0)


def brier_rows(fd: pd.DataFrame, preds: pd.DataFrame, freq: pd.DataFrame) -> pd.DataFrame:
    """Mac basina Brier: model ve lig frekansi (ayni maclar)."""
    d = preds.merge(fd[["match_id", "home_goals_ht", "away_goals_ht"]], on="match_id")
    d = d.dropna(subset=["home_goals_ht", "away_goals_ht"])
    d = d[d["league"].isin(freq.index)]
    y = pd.get_dummies(ht_outcome(d)).reindex(columns=["H", "D", "A"], fill_value=0).to_numpy(
        float)
    pm = d[["p_ht_home", "p_ht_draw", "p_ht_away"]].to_numpy()
    pb = freq.loc[d["league"]].to_numpy(float)
    return pd.DataFrame({"league": d["league"].to_numpy(), "season": d["season"].to_numpy(),
                         "brier_model": ((pm - y) ** 2).sum(axis=1),
                         "brier_base": ((pb - y) ** 2).sum(axis=1),
                         "hit_model": (d["pred_ht_result"].to_numpy()
                                       == ht_outcome(d).to_numpy())})
