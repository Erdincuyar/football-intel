"""Faz 10: piyasa reytingi, form ozellikleri ve dc-mr-v3 sonuc modeli (sentetik veri)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footintel.features import form as F
from footintel.features import market_rating as MR
from footintel.models import backtest as B
from footintel.models import live as L
from footintel.models import outcome as O
from footintel.models import strength as S
from footintel.models.ml import paired_bootstrap
from tests.test_history import CFG, league

STRONG = "A"     # sentetik ligde piyasanin favori gordugu takim


def with_odds(fd: pd.DataFrame) -> pd.DataFrame:
    """Sentetik kapanis oranlari: STRONG her macta favori, digerleri esit."""
    fd = fd.copy()
    ph = np.where(fd["home_team_id"] == STRONG, 0.7,
                  np.where(fd["away_team_id"] == STRONG, 0.2, 0.45))
    pa = np.where(fd["away_team_id"] == STRONG, 0.55,
                  np.where(fd["home_team_id"] == STRONG, 0.1, 0.28))
    pdraw = 1 - ph - pa
    for k, p in (("home", ph), ("draw", pdraw), ("away", pa)):
        fd[f"odds_{k}_close"] = 1 / (p * 1.05)
    return fd


def test_rating_uses_only_previous_matches():
    fd = with_odds(league())
    r1 = MR.match_ratings(fd).set_index("match_id")
    last = fd["match_date"].max()
    fd2 = fd.copy()
    fd2.loc[fd2["match_date"] == last, ["odds_home_close", "odds_away_close"]] = [1.01, 50.0]
    r2 = MR.match_ratings(fd2).set_index("match_id")
    pd.testing.assert_frame_equal(r1, r2)            # son haftanin orani hicbir satiri etkilemez
    assert r1.iloc[:3].isna().all().all()             # ilk hafta: gecmis yok -> NaN, uydurma yok


def test_rating_ranks_market_favourite_first():
    fd = with_odds(league())
    t = MR.team_ratings(fd)
    last = t.sort_values("match_id").groupby("team")["mr6"].last()
    assert last.idxmax() == STRONG


def test_form_features_do_not_see_own_match():
    fd = league()
    fd["home_shots_on_target"], fd["away_shots_on_target"] = 5.0, 3.0
    x1 = F.match_features(fd).set_index("match_id")
    fd2 = fd.copy()
    last = fd2["match_date"] == fd2["match_date"].max()
    fd2.loc[last, ["home_goals", "home_shots_on_target"]] = [9, 30.0]
    x2 = F.match_features(fd2).set_index("match_id")
    pd.testing.assert_frame_equal(x1, x2)


def _data(seed: int = 0) -> pd.DataFrame:
    fd = with_odds(league(seed))
    dc = B.walk_forward(fd, "E0", ["2526"], CFG)[["match_id", "p_home", "p_draw", "p_away"]]
    return fd.merge(dc, on="match_id").merge(MR.match_ratings(fd), on="match_id", how="left")


def test_model_probabilities_and_json_roundtrip():
    d = _data()
    m = O.fit(d)
    p = m.predict(d)
    assert p.shape == (len(d), 3) and np.allclose(p.sum(axis=1), 1)
    m2 = O.OutcomeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(d), p)
    # eksik reyting medyanla doldurulur, tahmin yine gecerli
    q = m.predict(d.assign(d_mr3=np.nan, d_mr6=np.nan))
    assert np.isfinite(q).all()


def test_walk_forward_ignores_future_results():
    d = pd.concat([_data(0), _data(1).assign(
        season="2627", match_id=lambda x: "b" + x["match_id"],
        match_date=lambda x: pd.to_datetime(x["match_date"]) + pd.DateOffset(years=1))],
        ignore_index=True)
    d["match_date"] = pd.to_datetime(d["match_date"])
    p1 = O.walk_forward(d, ["2627"], "2526", min_train=20)
    d2 = d.copy()
    late = d2["match_date"] >= d2.loc[d2["season"] == "2627", "match_date"].max() - pd.Timedelta(
        days=6)
    d2.loc[late, ["home_goals", "away_goals"]] = [9, 0]
    p2 = O.walk_forward(d2, ["2627"], "2526", min_train=20)
    pd.testing.assert_frame_equal(p1, p2)            # son haftanin sonucu o haftayi etkilemez
    assert len(p1) > 0


def test_live_fixture_outcome_overrides_1x2_only():
    fd = with_odds(league())
    d = _data()
    model = O.fit(d)
    gm = S.fit(fd, pd.Timestamp("2026-01-01"), CFG)
    fx = pd.DataFrame({"league": "E0", "match_date": pd.Timestamp("2026-01-10"),
                       "home_team_id": [STRONG, "B"], "away_team_id": ["B", STRONG],
                       "match_key": ["k1", "k2"]})
    out = L.fixture_outcome(fd, fx, gm, model)
    assert set(out) == {"k1", "k2"}
    for v in out.values():
        assert sum(v.values()) == pytest.approx(1.0)
    assert out["k1"]["p_home"] > out["k2"]["p_home"]   # favori evde daha olasi kazanir


def test_paired_bootstrap_detects_clear_difference():
    a = np.full(500, 0.9)
    b = np.full(500, 1.0) + np.random.default_rng(0).normal(0, 0.01, 500)
    m, lo, hi = paired_bootstrap(a, b)
    assert m < 0 and hi < 0
