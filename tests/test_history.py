"""Faz 9 asama 2: gecmis tahmin tablosu (sentetik veri)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from footintel import store
from footintel.models import backtest as B
from footintel.models import history as H
from footintel.models.strength import ModelConfig
from tests.test_cards import round_robin

TEST_URL = os.environ.get("FOOTINTEL_TEST_DATABASE_URL")
CFG = ModelConfig(family="poisson", dixon_coles=False, xi=0.0, penalty=4.0)
CCFG = ModelConfig(family="negbin", dixon_coles=False, xi=0.0, penalty=16.0)


def league(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fd = pd.concat([round_robin("2425", "2024-08-01", rng),
                    round_robin("2526", "2025-08-01", rng)], ignore_index=True)
    fd["home_team"], fd["away_team"] = fd["home_team_id"], fd["away_team_id"]
    fd["home_corners"] = rng.poisson(5, len(fd)).astype(float)
    fd["away_corners"] = rng.poisson(4, len(fd)).astype(float)
    return fd


def test_goal_backtest_does_not_see_its_own_week():
    fd = league()
    p1 = B.walk_forward(fd, "E0", ["2526"], CFG, CCFG)
    last = fd["match_date"].max()
    fd2 = fd.copy()
    wk = fd2["match_date"] == last
    fd2.loc[wk, ["home_goals", "away_goals", "home_corners", "away_corners"]] = [9, 0, 20, 0]
    p2 = B.walk_forward(fd2, "E0", ["2526"], CFG, CCFG)
    ids = fd.loc[wk, "match_id"]
    pred_cols = [c for c in p1 if not c.startswith("ll_")]   # ll_* = sonuca bagli skor
    a = p1[p1["match_id"].isin(ids)][pred_cols].reset_index(drop=True)
    b = p2[p2["match_id"].isin(ids)][pred_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert p1["pred_score"].str.match(r"^\d+-\d+$").all()


def test_assemble_has_no_result_columns_and_picks_argmax():
    fd = league()
    bt = B.walk_forward(fd, "E0", ["2526"], CFG, CCFG)
    h = H.assemble(fd[fd["season"] == "2526"], bt, None, pd.Timestamp("2026-11-12"))
    assert len(h) == 30 and h["matchweek"].between(1, 10).all()
    allowed = H.PRED_COLS + H.CARD_COLS + H.HT_COLS + ["exp_total_goals", "pred_result"]
    leaked = [c for c in h if c not in allowed and any(w in c for w in H.RESULT_WORDS)]
    assert leaked == []
    best = h[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)
    assert (h["pred_result"] == np.array(["H", "D", "A"])[best]).all()
    assert h["exp_home_yellow"].isna().all()                  # kart yoksa uydurma yok


@pytest.mark.skipif(not TEST_URL, reason="FOOTINTEL_TEST_DATABASE_URL yok")
def test_append_immutable_never_overwrites():
    df = pd.DataFrame({"match_id": ["m1", "m2"], "model_version": "v1", "p_home": [0.5, 0.4]})
    with store._connect(TEST_URL) as c:
        c.execute("DROP TABLE IF EXISTS app.hist_test")
    assert store.append_immutable(TEST_URL, "hist_test", df, ["match_id", "model_version"]) == 2
    changed = df.assign(p_home=[0.9, 0.9])
    more = pd.concat([changed, pd.DataFrame({"match_id": ["m3"], "model_version": "v1",
                                             "p_home": [0.3]})])
    assert store.append_immutable(TEST_URL, "hist_test", more, ["match_id", "model_version"]) == 1
    back = store.read_table("hist_test", url=TEST_URL).set_index("match_id")["p_home"]
    assert back.to_dict() == {"m1": 0.5, "m2": 0.4, "m3": 0.3}  # eski tahminler korundu
    # Yeni surum yeni kolon getirirse eski satirlar degismez, yeni kolon eskilerde NULL
    v2 = pd.DataFrame({"match_id": ["m1"], "model_version": "v2", "p_home": [0.6],
                       "p_ht_home": [0.3]})
    assert store.append_immutable(TEST_URL, "hist_test", v2, ["match_id", "model_version"]) == 1
    back = store.read_table("hist_test", url=TEST_URL)
    old = back[back["model_version"] == "v1"]
    assert old["p_home"].tolist() == [0.5, 0.4, 0.3] and old["p_ht_home"].isna().all()
