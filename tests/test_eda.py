"""EDA fonksiyonlarinin testleri (sentetik veri, gercek veri degil)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.analysis import eda


@pytest.fixture
def df():
    return eda.played(pd.DataFrame({
        "match_id": ["a", "b", "c", "d", "e"],
        "league": ["X"] * 5,
        "season": ["2526"] * 5,
        "home_goals": pd.array([1, 0, 2, 1, None], dtype="Int64"),
        "away_goals": pd.array([0, 0, 2, 3, None], dtype="Int64"),
        "odds_home_close": [2.0, 2.0, 2.0, 2.0, 2.0],
        "odds_draw_close": [4.0, 4.0, 4.0, 4.0, 4.0],
        "odds_away_close": [4.0, 4.0, 4.0, 4.0, 4.0],
    }))


def test_played_drops_unplayed_and_adds_total(df):
    assert len(df) == 4
    assert df["total_goals"].tolist() == [1, 0, 4, 4]


def test_dispersion_ratio(df):
    d = eda.dispersion(df, ["total_goals"]).iloc[0]
    assert d["mean"] == pytest.approx(2.25)
    assert d["var_mean_ratio"] == pytest.approx(np.var([1, 0, 4, 4], ddof=1) / 2.25)


def test_poisson_fit_sums_to_one(df):
    t = eda.poisson_fit(df["total_goals"], max_k=3)
    assert t["observed"].sum() == pytest.approx(1)
    assert t["poisson"].sum() == pytest.approx(1)
    assert t["goals"].iloc[-1] == "3+"


def test_market_rates(df):
    r = eda.market_rates(df).loc["X"]
    assert r["over25"] == pytest.approx(0.5)
    assert r["btts"] == pytest.approx(0.5)


def test_implied_probs_normalized_and_overround(df):
    p = eda.implied_1x2(df)
    assert p[["p_home", "p_draw", "p_away"]].sum(axis=1).tolist() == pytest.approx([1] * 4)
    assert p["overround"].iloc[0] == pytest.approx(0.0)  # 1/2 + 1/4 + 1/4 = 1


def test_brier_and_log_loss():
    y = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    assert eda.brier_1x2(y, y) == 0.0
    uniform = np.full((2, 3), 1 / 3)
    assert eda.brier_1x2(uniform, y) == pytest.approx(2 / 3)
    assert eda.log_loss_1x2(uniform, y) == pytest.approx(np.log(3))


def test_low_score_ratio(df):
    t = eda.low_score_check(df).set_index("score")
    # 0-0 bir kez (1/4); P(ev=0)=1/4, P(dep=0)=2/4 -> beklenen 1/8 -> oran 2
    assert t.loc["0-0", "ratio"] == pytest.approx(2.0)


def test_calibration_table():
    t = eda.calibration_table(pd.Series([0.1, 0.15, 0.9]), pd.Series([0, 1, 1]), bins=2)
    assert t["n"].tolist() == [2, 1]
    assert t["observed"].tolist() == [0.5, 1.0]
