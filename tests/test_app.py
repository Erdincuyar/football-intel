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
    monkeypatch.setattr(cfg, "CLEAN_DIR", clean)
    monkeypatch.setattr(cfg, "LEDGER_DIR", led_dir)
    return tmp_path


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Model_Performance.py"])
def test_page_runs(data_dir, page):
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception, at.exception
