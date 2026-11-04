"""Elo algoritmasinin birim testleri (sentetik maclar, gercek veri degil)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.features.elo import (
    BASE_RATING,
    EloParams,
    evaluate,
    expected_home,
    goal_diff_multiplier,
    run_elo,
)


def make(rows):
    """rows: (season, date, home, away, hg, ag)"""
    df = pd.DataFrame(rows, columns=["season", "match_date", "home_team_id", "away_team_id",
                                     "home_goals", "away_goals"])
    df["league"] = "X"
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["match_id"] = [f"m{i:03d}" for i in range(len(df))]
    return df


def test_expected_home_symmetry_and_hfa():
    assert expected_home(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home(1500, 1500, 60) > 0.5
    # Ev sahibinin beklentisi + deplasmanin beklentisi = 1
    assert expected_home(1600, 1450, 0) + expected_home(1450, 1600, 0) == pytest.approx(1.0)
    # 400 puan fark -> 10:1 oran -> 10/11
    assert expected_home(1900, 1500, 0) == pytest.approx(10 / 11)


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(-1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(-3) == pytest.approx(14 / 8)
    assert goal_diff_multiplier(5) == pytest.approx(2.0)


def test_first_match_uses_base_and_update_is_zero_sum():
    m = make([
        ("2122", "2021-08-01", "a", "b", 2, 0),
        ("2122", "2021-08-08", "a", "b", 1, 1),
    ])
    p = EloParams(k=20, home_advantage=0, season_regression=0)
    elo = run_elo(m, p).set_index("match_id")
    first, second = elo.loc["m000"], elo.loc["m001"]
    # Mac oncesi puan: ilk macta herkes 1500, o macin sonucu kullanilmamis
    assert first["home_elo_pre"] == first["away_elo_pre"] == BASE_RATING
    # 2-0 galibiyet: delta = 20 * 1.5 * (1 - 0.5) = 15
    assert second["home_elo_pre"] == pytest.approx(1515)
    assert second["away_elo_pre"] == pytest.approx(1485)
    assert second["home_elo_pre"] + second["away_elo_pre"] == pytest.approx(2 * BASE_RATING)


def test_unplayed_match_does_not_update():
    m = make([
        ("2122", "2021-08-01", "a", "b", np.nan, np.nan),
        ("2122", "2021-08-08", "a", "b", 1, 0),
    ])
    elo = run_elo(m, EloParams(home_advantage=0)).set_index("match_id")
    assert elo.loc["m001", "home_elo_pre"] == BASE_RATING


def test_newcomer_starts_at_relegated_mean_and_regression():
    m = make([
        ("2122", "2021-08-01", "a", "b", 3, 0),
        ("2122", "2021-08-08", "c", "a", 0, 1),
        ("2122", "2021-08-15", "b", "c", 1, 0),
        # c kume duser, d yukselir
        ("2223", "2022-08-01", "d", "a", 0, 0),
        ("2223", "2022-08-08", "b", "d", 0, 0),
    ])
    p = EloParams(k=20, home_advantage=0, season_regression=0.5)
    full = run_elo(m, p).set_index("match_id")

    # Ilk sezon sonu puanlarini elle hesapla
    one = run_elo(m[m["season"] == "2122"], EloParams(k=20, home_advantage=0,
                                                       season_regression=0.5))
    assert len(one) == 3
    r = {"a": BASE_RATING, "b": BASE_RATING, "c": BASE_RATING}
    for h, a, hg, ag in [("a", "b", 3, 0), ("c", "a", 0, 1), ("b", "c", 1, 0)]:
        e = expected_home(r[h], r[a], 0)
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        d = 20 * goal_diff_multiplier(hg - ag) * (s - e)
        r[h] += d
        r[a] -= d

    second = full.loc["m003"]
    assert second["home_elo_pre"] == pytest.approx(r["c"])  # d = kume dusen c'nin puani
    mean_ab = (r["a"] + r["b"]) / 2
    assert second["away_elo_pre"] == pytest.approx(mean_ab + (r["a"] - mean_ab) * 0.5)


def test_leagues_are_independent():
    m = make([("2122", "2021-08-01", "a", "b", 4, 0)])
    other = make([("2122", "2021-08-01", "c", "d", 0, 0)]).assign(league="Y", match_id="y0")
    elo = run_elo(pd.concat([m, other]), EloParams())
    assert len(elo) == 2
    assert (elo[["home_elo_pre", "away_elo_pre"]] == BASE_RATING).all().all()


def test_evaluate_perfect_and_coin():
    m = make([("2122", "2021-08-01", "a", "b", 1, 1)])
    elo = pd.DataFrame({"match_id": ["m000"], "home_elo_pre": [1500.0],
                        "away_elo_pre": [1500.0], "home_expected": [0.5]})
    assert evaluate(m, elo, ["2122"]) == 0.0
