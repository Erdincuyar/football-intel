"""Gol/korner modeli testleri. Veriler bilinen parametrelerden URETILMIS sentetik verilerdir."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from footintel.models import backtest as B
from footintel.models import strength as S


def simulate(n_teams=12, rounds=40, seed=0, home=0.25, c=0.1):
    rng = np.random.default_rng(seed)
    teams = [f"t{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    rows = []
    d0 = pd.Timestamp("2023-08-01")
    for rnd in range(rounds):
        perm = rng.permutation(n_teams)
        for k in range(n_teams // 2):
            h, a = perm[2 * k], perm[2 * k + 1]
            lam = np.exp(c + home + att[h] + dfn[a])
            mu = np.exp(c + att[a] + dfn[h])
            rows.append({
                "match_id": f"m{rnd}-{k}", "league": "X", "season": "2324",
                "match_date": d0 + pd.Timedelta(days=4 * rnd),
                "home_team_id": teams[h], "away_team_id": teams[a],
                "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                "home_corners": rng.negative_binomial(8, 8 / 13.5),
                "away_corners": rng.negative_binomial(8, 8 / 12.5),
            })
    return pd.DataFrame(rows), teams, att, dfn


@pytest.fixture(scope="module")
def sim():
    return simulate()


@pytest.mark.parametrize("cfg,cols", [
    (S.ModelConfig(xi=0.003, penalty=2.0), ("home_goals", "away_goals")),
    (S.ModelConfig(family="negbin", dixon_coles=False, xi=0.003, penalty=2.0),
     ("home_corners", "away_corners")),
])
def test_analytic_gradient_matches_numeric(sim, cfg, cols):
    df, teams, *_ = sim
    obj, p0, _ = S.fit(df, pd.Timestamp("2024-12-01"), cfg, home_col=cols[0], away_col=cols[1],
                       promoted={"t00"}, relegated={"t01", "t02"}, _return_problem=True)
    rng = np.random.default_rng(1)
    p = p0 + rng.normal(0, 0.1, len(p0))
    err = check_grad(lambda q: obj(q)[0], lambda q: obj(q)[1], p)
    assert err / np.linalg.norm(obj(p)[1]) < 1e-4


def test_recovers_known_parameters(sim):
    df, teams, att, dfn = sim
    m = S.fit(df, pd.Timestamp("2024-12-01"), S.ModelConfig(xi=0.0, penalty=0.5))
    order = [m.index[t] for t in teams]
    assert m.success
    assert np.corrcoef(m.attack[order], att)[0, 1] > 0.8
    assert np.corrcoef(m.defence[order], dfn)[0, 1] > 0.8
    assert m.home == pytest.approx(0.25, abs=0.12)


def test_no_future_data_used(sim):
    df, *_ = sim
    ref = pd.Timestamp("2023-10-01")
    m1 = S.fit(df, ref)
    changed = df.copy()
    changed.loc[pd.to_datetime(changed["match_date"]) >= ref, "home_goals"] = 9
    m2 = S.fit(changed, ref)
    assert np.allclose(m1.attack, m2.attack)


def test_score_matrix_and_markets_consistent(sim):
    df, *_ = sim
    m = S.fit(df, pd.Timestamp("2024-12-01"))
    mat = S.score_matrix(m, "t00", "t01")
    assert mat.sum() == pytest.approx(1)
    g = S.goal_markets(mat)
    assert g["p_home"] + g["p_draw"] + g["p_away"] == pytest.approx(1)
    assert 1 > g["p_over15"] > g["p_over25"] > g["p_over35"] > 0
    assert g["p_home_over05"] > g["p_home_over15"]
    assert S.correct_score_top(mat, 3)[0][1] == pytest.approx(mat.max())


def test_dixon_coles_tau_direction():
    base = S.FittedModel(S.ModelConfig(), ["a", "b"], 0.1, 0.2, np.zeros(2), np.zeros(2))
    neg = S.FittedModel(S.ModelConfig(), ["a", "b"], 0.1, 0.2, np.zeros(2), np.zeros(2),
                        rho=-0.1)
    m0, m1 = S.score_matrix(base, "a", "b"), S.score_matrix(neg, "a", "b")
    lam, mu = base.rates("a", "b")
    indep = np.outer(S.count_pmf(lam, 10), S.count_pmf(mu, 10))
    assert np.allclose(m0, indep / indep.sum())
    assert m1[0, 0] > m0[0, 0] and m1[1, 1] > m0[1, 1]   # rho < 0: 0-0 ve 1-1 artar
    assert m1[1, 0] < m0[1, 0] and m1[0, 1] < m0[0, 1]


def test_negbin_tends_to_poisson():
    p = S.count_pmf(5.0, 20)
    nb = S.count_pmf(5.0, 20, "negbin", np.log(1e6))
    assert np.allclose(p, nb, atol=1e-5)
    wide = S.count_pmf(5.0, 20, "negbin", np.log(3.0))
    assert wide[0] > p[0]  # asiri dagilim: uclar daha olasi


def test_promoted_team_uses_relegated_prior(sim):
    df, *_ = sim
    m = S.fit(df, pd.Timestamp("2024-12-01"), S.ModelConfig(penalty=8.0),
              promoted={"new"}, relegated={"t00", "t01"})
    i = m.index["new"]
    rel = [m.index["t00"], m.index["t01"]]
    assert m.attack[i] == pytest.approx(m.attack[rel].mean(), abs=1e-3)
    assert m.newcomer[0] == pytest.approx(m.attack[rel].mean())
    assert m.rates("never-seen", "t02") == m.rates("never-seen", "t02")  # bilinmeyen takim calisir


def test_walk_forward_predicts_with_past_only(sim):
    df, *_ = sim
    early = df.assign(season="2223", match_date=pd.to_datetime(df["match_date"])
                      - pd.Timedelta(days=365), match_id="e" + df["match_id"])
    full = pd.concat([early, df], ignore_index=True)
    cfg = S.ModelConfig()
    ccfg = S.ModelConfig(family="negbin", dixon_coles=False)
    p = B.walk_forward(full, "X", ["2324"], cfg, ccfg)
    assert len(p) == len(df)
    dates = df.set_index("match_id")["match_date"]
    assert (pd.to_datetime(p["fit_date"]) <= pd.to_datetime(p["match_id"].map(dates))).all()
    rates = B.base_rates(full, "2324")
    ev = B.evaluate(full.assign(odds_home_close=2.5, odds_draw_close=3.4, odds_away_close=3.0,
                                odds_over25_close=1.9, odds_under25_close=1.9), p, rates)
    assert ev.loc["X", "n"] == len(df)
    assert ev.loc["X", "1x2_logloss_model"] < np.log(3) + 0.1


def test_shot_blend_keeps_goal_scale_and_handles_missing():
    d = pd.DataFrame({"home_shots_on_target": [10.0, np.nan], "away_shots_on_target": [0.0, 4.0]})
    x, y = np.array([2.0, 1.0]), np.array([1.0, 0.0])
    bx, by = S._shot_blend(d, x, y, 0.5)
    c = 3.0 / 10.0  # gol / isabetli sut (yalnizca sut verisi olan mac)
    assert bx[0] == pytest.approx(0.5 * 2 + 0.5 * c * 10)
    assert by[0] == pytest.approx(0.5 * 1 + 0.5 * c * 0)
    assert (bx[1], by[1]) == (1.0, 0.0)  # sut verisi eksik: yalnizca gol


def test_shot_weight_disables_rho_and_fits(sim):
    df, *_ = sim
    rng = np.random.default_rng(3)
    df = df.assign(home_shots_on_target=df["home_goals"] * 3 + rng.integers(0, 3, len(df)),
                   away_shots_on_target=df["away_goals"] * 3 + rng.integers(0, 3, len(df)))
    m = S.fit(df, pd.Timestamp("2024-12-01"), S.ModelConfig(shot_weight=0.5))
    assert m.success and m.rho == 0.0
    assert S.score_matrix(m, "t00", "t01").sum() == pytest.approx(1)
