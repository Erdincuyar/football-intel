"""Streamlit sayfalarinin hata vermeden calistigini dogrular (sentetik veri)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import footintel.config as cfg  # noqa: E402
from footintel import ledger  # noqa: E402
from footintel.clean.fixtures import parse_fixtures  # noqa: E402
from tests.test_ledger import FIX, fake_predictions  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    clean, led_dir = tmp_path / "clean", tmp_path / "predictions"
    clean.mkdir()
    fx = parse_fixtures(FIX, season="2627")
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    fx["match_date"] = future.date()
    rows = ledger.build(fx, fake_predictions(fx), "v-test", pd.Timestamp.today(), "abc")
    ledger.append(led_dir, rows, "2099-01-01", "v-test")
    rng = np.random.default_rng(0)
    n = 40
    matches = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)], "league": "E0", "season": "2526",
        "home_team_id": "a", "away_team_id": "b",
        "home_goals": rng.poisson(1.5, n), "away_goals": rng.poisson(1.1, n),
        "home_corners": rng.poisson(5, n), "away_corners": rng.poisson(4, n),
        "odds_home_close": 2.2, "odds_draw_close": 3.4, "odds_away_close": 3.3,
        "odds_over25_close": 1.9, "odds_under25_close": 1.9,
    })
    matches.to_parquet(clean / "matches_football_data.parquet")
    pd.DataFrame({"match_id": matches["match_id"], "league": "E0", "season": "2526",
                  "p_home": 0.45, "p_draw": 0.27, "p_away": 0.28, "p_over25": 0.52,
                  "p_btts": 0.5, "p_corners_over95": 0.48, "ll_score": -2.9,
                  "ll_corners": -2.6}).to_parquet(clean / "backtest_predictions.parquet")
    # Faz 6: taktik sayfasi icin kucuk sentetik ciktilar
    import json

    from footintel.analysis import tactics as A
    from footintel.features import tactics as T
    from tests.test_tactics import MATCHES, synthetic_match
    T.team_match_metrics(synthetic_match(), MATCHES, np.full((16, 12), 0.01)).to_parquet(
        clean / "tactics_team_match_1516.parquet")
    (clean / "xt_grid.json").write_text(json.dumps([[0.01, 0.02], [0.05, 0.2]]))
    proxy = {"target": "field_tilt", "features": ["shot_share"], "beta": [0.0, 1.0],
             "r2_leave_league_out": 0.8, "rmse_leave_league_out": 0.03, "target_sd": 0.07}
    (clean / "tactics_proxy.json").write_text(json.dumps([proxy]))
    (clean / "tactics_summary.json").write_text(json.dumps({
        "reliability": [{"metric": "ppda", "split_half_r": 0.7, "season_reliability": 0.8}],
        "predictive": [{"metric": "xg_share", "r": 0.6, "r2": 0.36}]}))
    sides = matches.assign(match_date=pd.Timestamp("2026-08-01").date(), home_team="A",
                           away_team="B", home_shots=12, away_shots=8,
                           home_shots_on_target=4, away_shots_on_target=3, home_fouls=11,
                           away_fouls=13, season="2627")
    A.style_profile(sides, [proxy], min_matches=1).to_parquet(clean / "style_profile.parquet")
    # Faz 7: eslesme sayfasi
    from footintel.analysis import rules as R
    from footintel.features import style as ST
    cols = [f"pre_{c}" for c in ST.PROFILE_COLS]
    pd.DataFrame({"league": "E0", "team": ["A", "B"], "n_prev": [10, 6],
                  **{c: [0.6, 0.4] for c in cols}}).to_parquet(clean / "style_now.parquet")
    th = pd.DataFrame({c: [0.45, 0.55] for c in cols}, index=[0.25, 0.75])
    (clean / "style_thresholds.json").write_text(
        json.dumps({"E0": th.reset_index().to_dict("records")}))
    (clean / "matchup_rules.json").write_text(json.dumps([
        {"rule": r.key, "title": r.title, "target": r.target, "n_fired_disc": 50,
         "diff_disc": 0.3, "q": 0.01, "n_fired_conf": 40, "diff_conf": 0.2,
         "p_conf_dir": 0.02, "kept": i == 0} for i, r in enumerate(R.RULES)]))
    (clean / "style_clusters.json").write_text(json.dumps({
        "k": 2, "labels": ["+sut payi", "-sut payi"],
        "centers": [[1.0] * len(ST.CLUSTER_FEATURES), [-1.0] * len(ST.CLUSTER_FEATURES)]}))
    monkeypatch.setattr(cfg, "CLEAN_DIR", clean)
    monkeypatch.setattr(cfg, "LEDGER_DIR", led_dir)
    return tmp_path


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Model_Performance.py",
                                  "pages/2_Taktik_Profil.py", "pages/3_Eslesme.py",
                                  "pages/4_Mac_Merkezi.py", "pages/5_Performans_Paneli.py"])
def test_page_runs(data_dir, page):
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception, at.exception


def test_home_without_upcoming_matches(data_dir, monkeypatch):
    # Defterde yalnizca gecmis maclar varken (milli mac arasi) ana sayfa cokmemeli.
    led_dir = data_dir / "past_predictions"
    fx = parse_fixtures(FIX, season="2627")
    fx["match_date"] = (pd.Timestamp.today().normalize() - pd.Timedelta(days=5)).date()
    rows = ledger.build(fx, fake_predictions(fx), "v-test", pd.Timestamp.today(), "abc")
    led_dir.mkdir()
    # ledger.append gecmis maclari (dogru olarak) reddeder; burada maclar yazildiktan sonra
    # oynanmis gibi davranmak icin dosyayi dogrudan yaziyoruz.
    rows.to_parquet(led_dir / "2099-01-01_v-test.parquet", index=False)
    monkeypatch.setattr(cfg, "LEDGER_DIR", led_dir)
    import streamlit as st
    st.cache_data.clear()                  # onceki testin defteri onbellekte kalmasin
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("yaklasan mac yok" in i.value for i in at.info)


def test_match_center_with_history(data_dir):
    from footintel.analysis import history_eval as E
    from footintel.models import backtest as B
    from footintel.models import history as H
    from tests.test_history import CCFG, CFG, league

    fd = league()
    bt = B.walk_forward(fd, "E0", ["2526"], CFG, CCFG)
    h = H.assemble(fd[fd["season"] == "2526"], bt, None, pd.Timestamp("2026-11-12"))
    ev = E.evaluate_matches(h, E.match_results(fd), E.league_priors(fd))
    clean = data_dir / "clean"
    h.to_parquet(clean / "historical_predictions.parquet")
    ev.to_parquet(clean / "prediction_evaluation.parquet")
    E.summarize(ev, ["model_version", "league", "season", "matchweek"]).assign(
        scope="week").to_parquet(clean / "evaluation_summary.parquet")
    import streamlit as st
    st.cache_data.clear()
    at = AppTest.from_file(str(APP / "pages/4_Mac_Merkezi.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 6 and at.metric[0].value.startswith("%")
    cards = [m.value for m in at.markdown if m.value.startswith("<div class=\"fi-card")]
    assert len(cards) == 3                                    # son haftanin 3 maci
    assert "N/A" in cards[0]                                  # kart modeli yok -> N/A

    # Performans Paneli ayni veriyle
    st.cache_data.clear()
    at = AppTest.from_file(str(APP / "pages/5_Performans_Paneli.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 7 and at.metric[0].value.startswith("%")
    assert at.metric[1].value == "N/A"                         # ilk yari modeli yok
    assert len(at.dataframe) == 2
