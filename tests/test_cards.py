"""Faz 9 asama 1 testleri: hafta numarasi ve kart modeli (sentetik veri)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.features.matchweek import add_matchweek
from footintel.models import cards as K
from footintel.models.strength import ModelConfig

TEAMS = list("ABCDEF")


def round_robin(season: str, start: str, rng, yellow_bias: dict | None = None) -> pd.DataFrame:
    """6 takim, cift devre; her hafta 3 mac. Kart sayilari takim egilimiyle Poisson."""
    yellow_bias = yellow_bias or {}
    rows, d0 = [], pd.Timestamp(start)
    teams = TEAMS[:]
    for rnd in range(10):
        pairs = [(teams[i], teams[-1 - i]) for i in range(3)]
        if rnd % 2:
            pairs = [(b, a) for a, b in pairs]
        for j, (h, a) in enumerate(pairs):
            rows.append({
                "match_id": f"{season}-{rnd}-{j}", "league": "E0", "season": season,
                "match_date": (d0 + pd.Timedelta(days=7 * rnd)).date(),
                "home_team_id": h, "away_team_id": a,
                "home_goals": rng.poisson(1.4), "away_goals": rng.poisson(1.1),
                "home_yellow": rng.poisson(1.6 * yellow_bias.get(h, 1.0)),
                "away_yellow": rng.poisson(1.9 * yellow_bias.get(a, 1.0)),
                "home_red": rng.binomial(1, 0.06), "away_red": rng.binomial(1, 0.07)})
        teams = [teams[0], teams[-1], *teams[1:-1]]          # cember rotasyonu
    return pd.DataFrame(rows)


def test_matchweek_counts_and_postponed():
    rng = np.random.default_rng(0)
    fd = round_robin("2526", "2025-08-01", rng)
    # 2. haftanin ilk macini 6. haftadan sonraya ertele
    i = fd.index[fd["match_id"] == "2526-1-0"][0]
    fd.loc[i, "match_date"] = (pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * 5 + 2)).date()
    w = add_matchweek(fd).set_index("match_id")["matchweek"]
    assert w["2526-0-0"] == 1 and w["2526-2-1"] == 3
    assert w["2526-1-0"] >= 5                                # oynandigi siraya gore
    assert w.max() == 10


def test_red_rates_shrink_to_league():
    hist = pd.DataFrame({"home_team_id": ["A"] * 2 + ["B"] * 40,
                         "away_team_id": ["C"] * 42,
                         "home_red": [1, 1] + [0] * 40, "away_red": [0] * 42})
    rates, league = K.red_rates(hist, k=30)
    assert league == pytest.approx(2 / 84)
    assert league < rates["A"] < 0.5                          # 2 macta 2 kirmizi: buzuldu
    assert rates["B"] < league


def test_card_model_learns_team_tendency_without_leakage():
    rng = np.random.default_rng(1)
    bias = {"A": 2.0, "F": 0.5}
    fd = pd.concat([round_robin("2425", "2024-08-01", rng, bias),
                    round_robin("2526", "2025-08-01", rng, bias)], ignore_index=True)
    cfg = ModelConfig(family="negbin", dixon_coles=False, xi=0.0, penalty=4.0)
    p = K.walk_forward(fd, "E0", ["2526"], cfg)
    assert (pd.to_datetime(p["fit_date"]) <= pd.to_datetime(
        p["match_id"].map(fd.set_index("match_id")["match_date"]))).all()
    pa = p.merge(fd, on="match_id")
    a_home = pa.loc[pa["home_team_id"] == "A", "exp_home_yellow"].mean()
    f_home = pa.loc[pa["home_team_id"] == "F", "exp_home_yellow"].mean()
    assert a_home > 1.5 * f_home
    # sizinti: 2526'nin son haftasinin kartlarini degistir, o haftanin tahmini degismemeli
    last = fd["match_date"].max()
    fd2 = fd.copy()
    fd2.loc[fd2["match_date"] == last, ["home_yellow", "away_yellow"]] = 15
    p2 = K.walk_forward(fd2, "E0", ["2526"], cfg)
    pd.testing.assert_frame_equal(p, p2)


def test_evaluate_skips_missing_cards():
    fd = pd.DataFrame({"match_id": ["m1", "m2"], "league": "E0", "home_yellow": [2.0, np.nan],
                       "away_yellow": [1.0, np.nan], "home_red": [0, np.nan],
                       "away_red": [1, np.nan]})
    preds = pd.DataFrame({"match_id": ["m1", "m2"], "league": "E0", "exp_home_yellow": 2.0,
                          "exp_away_yellow": 2.0, "p_red_any": 0.2})
    base = pd.DataFrame({"home_yellow": [1.0], "away_yellow": [1.0], "p_red_any": [0.1]},
                        index=["E0"])
    ev = K.evaluate(fd, preds, base).loc["E0"]
    assert ev["matches"] == 1
    assert ev["yellow_mae_model"] == pytest.approx(0.5)
    assert ev["red_brier_model"] == pytest.approx(0.64)
