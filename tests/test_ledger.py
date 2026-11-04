"""Tahmin defteri testleri (sentetik fikstur ve sonuclar, gercek veri degil)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footintel import ledger
from footintel.clean.fixtures import parse_fixtures
from footintel.models import strength as S

FIX = Path(__file__).parent / "fixtures" / "fixtures_sample.csv"
NOW = pd.Timestamp("2026-11-05 06:00")


@pytest.fixture
def fixtures():
    return parse_fixtures(FIX, season="2627")


def fake_predictions(fx):
    model = S.FittedModel(S.ModelConfig(), ["x"], 0.2, 0.25, np.zeros(1), np.zeros(1), rho=-0.05)
    cmodel = S.FittedModel(S.ModelConfig(family="negbin"), ["x"], 1.6, 0.1, np.zeros(1),
                           np.zeros(1), log_r=2.5)
    out = {}
    for r in fx.itertuples():
        m = S.score_matrix(model, r.home_team_id, r.away_team_id)
        corners = S.corner_markets(cmodel, r.home_team_id, r.away_team_id)
        corners |= {f"p_{side}_corners_over{line}": 0.5 for side in ("home", "away")
                    for line in ("35", "45", "55")}
        out[r.match_key] = (S.goal_markets(m), m, corners)
    return out


def test_parse_fixtures_filters_leagues_and_maps_odds(fixtures):
    assert list(fixtures["league"]) == ["E0", "T1"]
    assert fixtures.loc[0, "match_key"] == "E0-2627-team-alpha-team-beta"
    assert fixtures.loc[0, "odds_home_avg"] == 2.0
    assert np.isnan(fixtures.loc[1, "odds_over25_avg"])
    assert str(fixtures.loc[0, "match_date"]) == "2026-11-07"


def test_build_rows_sum_to_one_and_market_probs(fixtures):
    rows = ledger.build(fixtures, fake_predictions(fixtures), "v-test", NOW, "abc")
    sums = rows.groupby(["match_key", "market"])["prob"].sum()
    assert np.allclose(sums, 1)
    assert {"1x2", "ou25", "btts", "correct_score", "corners_ou95", "home_goals_ou15"} <= set(
        rows["market"])
    e0 = rows[(rows["league"] == "E0") & (rows["market"] == "1x2")].set_index("selection")
    inv = np.array([1 / 2, 1 / 3.5, 1 / 4])
    assert e0.loc["home", "market_prob"] == pytest.approx(inv[0] / inv.sum())
    t1_ou = rows[(rows["league"] == "T1") & (rows["market"] == "ou25")]
    assert t1_ou["market_prob"].isna().all()   # oran yoksa piyasa olasiligi bos
    assert rows["prediction_id"].is_unique


def test_append_is_append_only(tmp_path, fixtures):
    rows = ledger.build(fixtures, fake_predictions(fixtures), "v-test", NOW, "abc")
    path = ledger.append(tmp_path, rows, "2026-11-05", "v-test")
    assert path.exists()
    assert ledger.already_predicted(tmp_path, "v-test") == set(fixtures["match_key"])
    with pytest.raises(ledger.LedgerError, match="zaten tahmin"):
        ledger.append(tmp_path, rows, "2026-11-06", "v-test")
    # Baska bir surum ayni maci tahmin edebilir (surumler kiyaslanabilsin)
    ledger.append(tmp_path, rows.assign(model_version="v2"), "2026-11-05", "v2")
    # Ayni gun ayni surumle yeni maclar gelse bile var olan dosyanin uzerine yazilmaz
    with pytest.raises(ledger.LedgerError, match="uzerine yazilmaz"):
        ledger.append(tmp_path, rows.assign(match_key="yeni-mac"), "2026-11-05", "v-test")


def test_append_rejects_past_matches(tmp_path, fixtures):
    rows = ledger.build(fixtures, fake_predictions(fixtures), "v-test",
                        pd.Timestamp("2026-11-09 06:00"), "abc")
    with pytest.raises(ledger.LedgerError, match="Oynanmis"):
        ledger.append(tmp_path, rows, "2026-11-09", "v-test")


def test_evaluate_outcomes(fixtures):
    rows = ledger.build(fixtures, fake_predictions(fixtures), "v-test", NOW, "abc")
    results = pd.DataFrame({
        "league": ["E0"], "season": ["2627"], "home_team_id": ["team-alpha"],
        "away_team_id": ["team-beta"], "home_goals": [2], "away_goals": [1],
        "home_corners": [6], "away_corners": [5],
    })
    ev = ledger.evaluate(rows, results).merge(rows, on="prediction_id")
    got = ev.set_index(["market", "selection"])["outcome"]
    assert got[("1x2", "home")] == 1 and got[("1x2", "draw")] == 0
    assert got[("ou25", "over")] == 1 and got[("ou35", "over")] == 0
    assert got[("btts", "yes")] == 1
    assert got[("correct_score", "2-1")] == 1 and got[("correct_score", "1-1")] == 0
    assert got[("corners_ou105", "over")] == 1 and got[("corners_ou115", "over")] == 0
    assert got[("home_corners_ou55", "over")] == 1 and got[("away_corners_ou55", "over")] == 0
    assert got[("away_goals_ou05", "over")] == 1 and got[("away_goals_ou15", "over")] == 0
    assert (ev["league"] == "E0").all()   # T1 maci oynanmadi: degerlendirmede yok
    summary = ledger.market_summary(rows, ledger.evaluate(rows, results)).set_index("market")
    p_home = rows[(rows["market"] == "1x2") & (rows["selection"] == "home")
                  & (rows["league"] == "E0")]["prob"].iloc[0]
    assert summary.loc["1x2", "logloss_model"] == pytest.approx(-np.log(p_home))
    assert summary.loc["1x2", "n_matches"] == 1


def test_snapshot_hash_changes_with_data():
    a = pd.DataFrame({"match_id": ["1", "2"], "home_goals": [1, 2], "away_goals": [0, 0]})
    assert ledger.snapshot_hash(a) == ledger.snapshot_hash(a.iloc[::-1])
    assert ledger.snapshot_hash(a) != ledger.snapshot_hash(a.assign(home_goals=[1, 3]))
