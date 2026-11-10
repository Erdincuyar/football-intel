"""Faz 7 testleri: sentetik veri (gercek degil), bilinen cevaplarla."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.analysis import rules as R
from footintel.features import style as ST


def side_rows(fouls: list[float], season="2223", league="E0", team="A") -> pd.DataFrame:
    n = len(fouls)
    return pd.DataFrame({
        "match_id": [f"{team}{season}{i}" for i in range(n)], "league": league,
        "season": season, "team": team, "is_home": 1,
        "match_date": pd.date_range("2022-08-01", periods=n, freq="7D").date,
        "shots": 10.0, "shots_against": 10.0, "sot": 4.0, "sot_against": 4.0,
        "corners": 5.0, "corners_against": 5.0, "fouls": fouls,
        "fouls_against": 12.0, "goals": 1, "goals_against": 1, "yellows": 2.0,
        "yellows_against": 2.0,
    })


def test_pre_match_profile_uses_only_previous_matches():
    p = ST.pre_match_profile(side_rows([10, 20, 30, 40, 50, 60]), window=3, min_prev=2)
    pre = p.sort_values("match_date")["pre_fouls"].tolist()
    assert np.isnan(pre[0]) and np.isnan(pre[1])          # en az 2 onceki mac gerekiyor
    assert pre[2] == pytest.approx(15)                     # (10+20)/2, bugunku 30 dahil degil
    assert pre[5] == pytest.approx(40)                     # son 3: 30, 40, 50


def test_flags_use_previous_seasons_only():
    a = pd.concat([side_rows(list(np.arange(10.0, 30.0)), "2122"),
                   side_rows([100.0] * 12, "2223")], ignore_index=True)
    p = ST.add_flags(ST.attach_opponent(ST.pre_match_profile(a)))
    first = p[p["season"] == "2122"]
    assert first["fouls_high"].isna().all()                # ilk sezonun esigi yok
    late = p[(p["season"] == "2223") & p["pre_fouls"].ge(60)]
    assert late["fouls_high"].astype(bool).all()           # 2021/22 esiginin cok ustunde


def test_kmeans_finds_separated_groups():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(m, 0.2, (30, 2)) for m in ([0, 0], [5, 5], [0, 5])])
    lab, centers, _ = ST.kmeans(x, 3, n_init=5)
    assert len(set(lab[:30])) == 1 and len(set(lab)) == 3
    assert ST.silhouette(x, lab) > 0.8


def _rule_frame(effect: float, n: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hi = rng.random(n) < 0.25
    opp_lo = rng.random(n) < 0.25
    fired = hi & opp_lo
    return pd.DataFrame({
        "match_id": np.arange(n) // 2, "season": "2223",
        "corner_share_high": hi, "opp_corner_share_low": opp_lo,
        "opp_corner_share_high": ~opp_lo,
        "res_corners": rng.normal(0, 2, n) + effect * fired,
    })


def test_rule_detects_planted_effect_and_not_noise():
    rule = R.RULES[0]                                       # korner baskisi
    assert rule.key == "corner_pressure"
    real = R.test_rule(_rule_frame(1.0), rule, n_boot=500)
    assert real["diff"] == pytest.approx(1.0, abs=0.3) and real["p_two"] < 0.001
    null = R.test_rule(_rule_frame(0.0), rule, n_boot=500)
    assert null["lo"] < 0 < null["hi"]


def test_benjamini_hochberg():
    q = R.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.5]))
    # sirali p: .01 .03 .04 .5 -> p*n/rank: .04 .06 .0533 .5 -> kumulatif min: .04 .0533 .0533 .5
    assert q.tolist() == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5])


def test_evaluate_rules_requires_confirmation():
    disc = _rule_frame(1.0, seed=1)
    conf_same = _rule_frame(1.0, seed=2).assign(season="2425")
    conf_none = _rule_frame(0.0, seed=3).assign(season="2425")
    rules = [R.RULES[0]]
    kept = R.evaluate_rules(pd.concat([disc, conf_same]), ["2223"], ["2425"], rules)
    dropped = R.evaluate_rules(pd.concat([disc, conf_none]), ["2223"], ["2425"], rules)
    assert kept["kept"].iloc[0] and not dropped["kept"].iloc[0]


def test_pair_rows_mirror_profiles():
    prof = pd.DataFrame({"team": ["A", "B"], **{f"pre_{c}": [1.0, 2.0]
                                                for c in ST.PROFILE_COLS}})
    rows = ST.pair_rows(prof, "A", "B")
    assert rows.loc[0, "pre_fouls"] == 1 and rows.loc[0, "opp_pre_fouls"] == 2
    assert rows.loc[1, "team"] == "B" and rows.loc[1, "opp_pre_fouls"] == 1
