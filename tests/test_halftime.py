"""Faz 9: ilk yari sonucu modeli (sentetik veri)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.analysis import history_eval as E
from footintel.analysis import history_view as V
from footintel.models import halftime as HT
from footintel.models.strength import ModelConfig
from tests.test_history import league

CFG = ModelConfig(family="poisson", dixon_coles=False, xi=0.0, penalty=16.0)


def with_ht(seed: int = 0) -> pd.DataFrame:
    fd = league(seed)
    rng = np.random.default_rng(seed + 1)
    fd["home_goals_ht"] = np.minimum(fd["home_goals"], rng.poisson(0.6, len(fd)))
    fd["away_goals_ht"] = np.minimum(fd["away_goals"], rng.poisson(0.5, len(fd)))
    return fd


def test_probabilities_and_argmax():
    p = HT.walk_forward(with_ht(), "E0", ["2526"], CFG)
    assert len(p) == 30
    s = p[["p_ht_home", "p_ht_draw", "p_ht_away"]].sum(axis=1)
    assert np.allclose(s, 1, atol=1e-6)
    best = p[["p_ht_home", "p_ht_draw", "p_ht_away"]].to_numpy().argmax(axis=1)
    assert (p["pred_ht_result"] == np.array(["H", "D", "A"])[best]).all()


def test_does_not_see_its_own_week():
    fd = with_ht()
    last = fd["match_date"].max()
    fd2 = fd.copy()
    fd2.loc[fd2["match_date"] == last, ["home_goals_ht", "away_goals_ht"]] = [7, 0]
    a = HT.walk_forward(fd, "E0", ["2526"], CFG)
    b = HT.walk_forward(fd2, "E0", ["2526"], CFG)
    pd.testing.assert_frame_equal(a, b)


def test_missing_ht_scores_are_not_invented():
    fd = with_ht()
    fd.loc[fd["season"] == "2526", ["home_goals_ht", "away_goals_ht"]] = np.nan
    freq = HT.league_frequencies(fd, "2526")
    assert np.isclose(freq.loc["E0"].sum(), 1)
    r = HT.brier_rows(fd, HT.walk_forward(fd, "E0", ["2526"], CFG), freq)
    assert r.empty                                    # sonuc yoksa olcum de yok


def test_eval_and_card_show_halftime():
    from footintel.models import backtest as B
    from footintel.models import history as H
    from tests.test_history import CCFG
    from tests.test_history import CFG as GCFG

    fd = with_ht()
    bt = B.walk_forward(fd, "E0", ["2526"], GCFG, CCFG)
    ht = HT.walk_forward(fd, "E0", ["2526"], CFG)
    h = H.assemble(fd[fd["season"] == "2526"], bt, None, pd.Timestamp("2026-11-12"),
                   halftime=ht)
    ev = E.evaluate_matches(h, E.match_results(fd), E.league_priors(fd))
    actual = HT.ht_outcome(fd.set_index("match_id").loc[ev["match_id"]]).to_numpy()
    assert (ev["ht_result_actual"].to_numpy() == actual).all()
    assert (ev["ht_result_ok"] == (ev["ht_result_pred"] == ev["ht_result_actual"])).all()
    m = E.prob_metrics(ev, ["season"]).set_index("metric")
    assert m.loc["brier_ht", "n"] == 30
    card = h.merge(ev[["match_id", *[c for c in ev if c not in h]]], on="match_id").iloc[0]
    row = {r["label"]: r for r in V.card_rows(card)}["Ilk yari sonucu"]
    assert row["pred"].startswith(("1 (", "X (", "2 (")) and row["mark"] is not None
