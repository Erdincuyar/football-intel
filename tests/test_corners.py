"""Faz 5 korner regresyonu testleri (bilinen parametrelerden uretilmis sentetik veri)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from footintel.models import corners as C


def synth(n=4000, seed=0, beta=(1.6, 0.08, 0.15, -0.05), r=9.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "is_home": rng.integers(0, 2, n).astype(float),
        "corner_rate": rng.normal(1.6, 0.15, n),
        "supremacy": rng.normal(0, 0.6, n),
    })
    x = df[["is_home", "corner_rate", "supremacy"]].to_numpy()
    xs = (x - x.mean(0)) / x.std(0)
    m = np.exp(beta[0] + xs @ np.array(beta[1:]))
    df["corners"] = rng.negative_binomial(r, r / (r + m))
    return df


def test_nb_regression_recovers_coefficients():
    m = C.NBRegression(["is_home", "corner_rate", "supremacy"]).fit(synth())
    assert m.success
    assert np.allclose(m.beta, [1.6, 0.08, 0.15, -0.05], atol=0.03)
    assert np.exp(m.log_r) == pytest.approx(9.0, rel=0.35)


def test_nb_regression_gradient():
    m = C.NBRegression(["is_home", "corner_rate"]).fit(synth(500))
    p = np.r_[m.beta, m.log_r] + 0.05
    err = check_grad(lambda q: m._obj(q)[0], lambda q: m._obj(q)[1], p)
    assert err / np.linalg.norm(m._obj(p)[1]) < 1e-4


def test_side_table_and_total_distribution():
    feat = pd.DataFrame({
        "match_id": ["a"], "league": ["X"], "season": ["2526"],
        "exp_goals_home": [1.8], "exp_goals_away": [0.9],
        "corner_rate_home": [np.log(6.0)], "corner_rate_away": [np.log(4.0)],
        "shot_rate_home": [np.log(15.0)], "shot_rate_away": [np.log(9.0)],
    })
    fd = pd.DataFrame({"match_id": ["a"], "home_corners": [7], "away_corners": [3]})
    s = C.side_table(feat, fd)
    home = s[s["side"] == "home"].iloc[0]
    away = s[s["side"] == "away"].iloc[0]
    assert home["supremacy"] == pytest.approx(0.9) and away["supremacy"] == pytest.approx(-0.9)
    assert away["opp_shot_rate"] == pytest.approx(np.log(15.0))
    assert (home["corners"], away["corners"]) == (7, 3)
    dist = C.total_distribution(6.0, 4.0, np.log(8.0))
    assert dist.sum() == pytest.approx(1)
    assert (np.arange(len(dist)) * dist).sum() == pytest.approx(10.0, rel=1e-3)
    ev = C.evaluate_sides(s, np.array([6.0, 4.0]), np.log(8.0)).iloc[0]
    assert ev["p_corners_over85"] > ev["p_corners_over95"] > ev["p_corners_over105"]
    assert ev["ll_corners"] == pytest.approx(np.log(dist[10]))


def test_bootstrap_diff_detects_real_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(2.60, 0.3, 800)
    mean, lo, hi = C.bootstrap_diff(a, a + 0.05)
    assert mean == pytest.approx(-0.05) and hi < 0
    mean, lo, hi = C.bootstrap_diff(a, a + rng.normal(0, 0.3, 800))
    assert lo < 0 < hi   # sadece gurultu: anlamli degil
