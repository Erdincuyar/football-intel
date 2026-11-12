"""Faz 9 asama 3: tahmin-sonuc eslestirme ve degerlendirme (elle kurulmus kucuk ornekler)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.analysis import history_eval as E
from footintel.models import backtest as B
from footintel.models import history as H
from tests.test_history import CCFG, CFG, league


def _fd() -> pd.DataFrame:
    """Iki sezon, tek lig. 2425 baseline icin, 2526 degerlendirme icin."""
    rows = [
        # match_id, season, date, hg, ag, hc, ac, hy, ay, hr, ar
        (1, "2425", "2024-08-10", 1, 1, 5, 5, 2, 2, 0, 0),
        (2, "2425", "2024-08-17", 3, 0, 6, 4, 1, 3, 1, 0),
        (10, "2526", "2025-08-09", 2, 1, 6, 3, 1, 2, 0, 0),
        (11, "2526", "2025-08-09", 0, 0, np.nan, np.nan, 2, 2, 0, 1),   # korner verisi yok
        (12, "2526", "2025-08-16", np.nan, np.nan, 4, 4, 1, 1, 0, 0),   # henuz oynanmadi
    ]
    fd = pd.DataFrame(rows, columns=["match_id", "season", "match_date", "home_goals",
                                     "away_goals", "home_corners", "away_corners",
                                     "home_yellow", "away_yellow", "home_red", "away_red"])
    fd["league"] = "E0"
    fd["match_date"] = pd.to_datetime(fd["match_date"])
    fd["home_team"], fd["away_team"] = "Ev", "Dep"
    fd["odds_home_close"], fd["odds_draw_close"], fd["odds_away_close"] = 2.0, 3.5, 4.0
    fd["odds_over25_close"], fd["odds_under25_close"] = 1.8, 2.0
    return fd


def _preds() -> pd.DataFrame:
    p = pd.DataFrame({
        "match_id": [10, 11, 12], "league": "E0", "season": "2526",
        "matchweek": [1, 1, 2], "match_date": pd.to_datetime(["2025-08-09"] * 2 + ["2025-08-16"]),
        "home_team": "Ev", "away_team": "Dep",
        "p_home": [0.5, 0.3, 0.4], "p_draw": [0.3, 0.4, 0.3], "p_away": [0.2, 0.3, 0.3],
        "pred_result": ["H", "D", "H"], "pred_score": ["1-0", "0-0", "1-1"],
        "p_pred_score": 0.1, "exp_home_goals": [1.6, 0.9, 1.2],
        "exp_away_goals": [0.9, 0.7, 1.0], "p_over25": [0.55, 0.3, 0.5],
        "p_btts": [0.6, 0.4, 0.5], "exp_home_corners": [5.4, 5.0, 5.0],
        "exp_away_corners": [3.6, 4.0, 4.0], "exp_corners": [9.0, 9.0, 9.0],
        "p_corners_over95": 0.4, "exp_home_yellow": [1.4, 2.6, 2.0],
        "exp_away_yellow": [2.2, np.nan, 2.0], "p_red_any": 0.2,
        "model_version": "hist-v1", "created_at": pd.Timestamp("2026-11-12"),
    })
    p["exp_total_goals"] = p["exp_home_goals"] + p["exp_away_goals"]
    return p


def _eval() -> pd.DataFrame:
    fd = _fd()
    return E.evaluate_matches(_preds(), E.match_results(fd), E.league_priors(fd))


def test_results_only_played_matches():
    r = E.match_results(_fd())
    assert set(r["match_id"]) == {1, 2, 10, 11}
    assert r.set_index("match_id").loc[10, "result"] == "H"


def test_priors_use_previous_season_only():
    pr = E.league_priors(_fd()).set_index("season")
    assert np.isnan(pr.loc["2425", "base_goals_total"])          # oncesi yok -> N/A
    assert pr.loc["2526", "base_goals_total"] == pytest.approx(2.5)   # (2 + 3) / 2
    assert pr.loc["2526", "base_p_red"] == pytest.approx(0.5)
    fd = _fd()
    fd.loc[fd["season"] == "2425", "season"] = "2324"            # arada sezon bosluk
    assert E.league_priors(fd).set_index("season")["base_goals_total"].isna().all()


def test_unplayed_match_is_not_evaluated():
    assert set(_eval()["match_id"]) == {10, 11}


def test_marks_match_hand_computed_values():
    ev = _eval().set_index("match_id")
    a, b = ev.loc[10], ev.loc[11]
    assert a["result_ok"] and b["result_ok"]                      # H==H, D==D
    assert not a["score_ok"] and b["score_ok"]                    # 1-0 vs 2-1; 0-0
    # toplam gol: 2.5 -> 3 (yarim yukari), gercek 3 -> tam; 1.6 -> 2, gercek 0 -> ikisi de yanlis
    assert a["goals_total_pred"] == 3 and a["goals_total_ok"]
    assert b["goals_total_pred"] == 2 and not b["goals_total_ok"] and not b["goals_total_ok_pm1"]
    assert a["ou25_ok"] and b["ou25_ok"]                          # 3>2.5 evet; 0 alt
    assert a["btts_ok"] and b["btts_ok"]
    # korner: 5.4->5 vs 6 (+-1 icinde), 3.6->4 vs 3 (+-1), toplam 9 vs 9
    assert not a["corners_home_ok"] and a["corners_home_ok_pm1"]
    assert a["corners_total_ok"]
    assert pd.isna(b["corners_home_ok"]) and pd.isna(b["corners_total_ok_pm1"])   # veri yok
    # sari: dep beklentisi yok -> toplam da N/A (uydurma yok)
    assert pd.isna(b["yellow_away_ok"]) and pd.isna(b["yellow_total_ok"])
    assert b["yellow_home_ok"] is not pd.NA and not b["yellow_home_ok"]   # 2.6->3 vs 2
    # kirmizi: p=0.2 -> "yok"; mac 11'de kirmizi var -> yanlis
    assert a["red_ok"] and not b["red_ok"]
    # piyasa favorisi ev sahibi (oran 2.0)
    assert a["result_base_pred"] == "H" and a["result_base_ok"] and not b["result_base_ok"]


def test_summary_counts_and_never_invents_percentages():
    s = E.summarize(_eval(), ["season"]).set_index("market")
    assert s.loc["result", "n"] == 2 and s.loc["result", "pct"] == 1.0
    assert s.loc["corners_home", "n"] == 1                       # korner verisi olmayan haric
    assert s.loc["yellow_total", "n"] == 1
    assert s.loc["result", "base_pct"] == 0.5
    assert s.loc["goals_total", "pct_pm1"] == 0.5
    empty = _eval().assign(corners_home_ok=pd.array([pd.NA, pd.NA], dtype="boolean"))
    e = E.summarize(empty, ["season"]).set_index("market")
    assert e.loc["corners_home", "n"] == 0 and np.isnan(e.loc["corners_home", "pct"])


def test_prob_metrics_same_matches_for_model_and_baseline():
    m = E.prob_metrics(_eval(), ["season"]).set_index("metric")
    assert m.loc["brier_1x2", "n"] == 2 and m.loc["brier_1x2", "baseline_name"] == "piyasa"
    y = np.array([[1, 0, 0], [0, 1, 0]])
    p = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    assert m.loc["brier_1x2", "model"] == pytest.approx(np.mean(((p - y) ** 2).sum(axis=1)))
    assert m.loc["mae_corners_total", "n"] == 1
    assert m.loc["mae_goals_total", "model"] == pytest.approx((0.5 + 1.6) / 2)
    assert m.loc["mae_goals_total", "baseline"] == pytest.approx((0.5 + 2.5) / 2)


def test_end_to_end_on_synthetic_walk_forward():
    fd = league()
    bt = B.walk_forward(fd, "E0", ["2526"], CFG, CCFG)
    h = H.assemble(fd[fd["season"] == "2526"], bt, None, pd.Timestamp("2026-11-12"))
    ev = E.evaluate_matches(h, E.match_results(fd[fd["season"] == "2526"]),
                            E.league_priors(fd))
    assert len(ev) == len(h)
    s = E.summarize(ev, ["season", "matchweek"])
    r = s[s["market"] == "result"]
    assert r["n"].sum() == len(h) and r["pct"].between(0, 1).all()
    # kart modeli yoksa kart isabeti uydurulmaz
    assert (s.loc[s["market"] == "yellow_total", "n"] == 0).all()
