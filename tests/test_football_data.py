"""football-data temizleme ve yukleme testleri.

Fixture dosyalari elle yazilmis sentetik orneklerdir (tests/fixtures/README.md).
"""

from pathlib import Path

import pandas as pd
import pytest

from footintel import db
from footintel.clean.football_data import (
    OUTPUT_COLUMNS,
    clean_matches,
    coverage_report,
    read_raw_csv,
    validate,
)
from footintel.clean.teams import slugify

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def e0() -> pd.DataFrame:
    return clean_matches(read_raw_csv(FIXTURES / "sample_E0.csv"), "E0", "2526")


def test_empty_rows_and_unnamed_columns_dropped(e0):
    assert len(e0) == 2
    assert list(e0.columns) == OUTPUT_COLUMNS


def test_dates_are_parsed_day_first(e0):
    assert str(e0.loc[0, "match_date"]) == "2025-08-16"


def test_match_id_is_deterministic(e0):
    assert e0.loc[0, "match_id"] == "E0-2526-20250816-team-alpha-team-beta"


def test_missing_stat_columns_become_na_not_zero():
    t1 = clean_matches(read_raw_csv(FIXTURES / "sample_T1_no_stats.csv"), "T1", "2526")
    assert t1["home_corners"].isna().all()
    assert t1.loc[0, "away_goals"] == 3


def test_validation_catches_impossible_shots(e0):
    broken = e0.copy()
    broken.loc[0, "home_shots_on_target"] = 99
    with pytest.raises(ValueError, match="isabetli sut"):
        validate(broken)


def test_validation_catches_wrong_result(e0):
    broken = e0.copy()
    broken.loc[0, "result"] = "A"
    with pytest.raises(ValueError, match="FTR"):
        validate(broken)


@pytest.mark.parametrize(
    "name, expected",
    [("Man United", "man-united"), ("Göztepe", "goztepe"), ("Kasımpaşa", "kasimpasa"),
     ("İstanbulspor", "istanbulspor")],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_coverage_report_shows_missing_stats():
    t1 = clean_matches(read_raw_csv(FIXTURES / "sample_T1_no_stats.csv"), "T1", "2526")
    report = coverage_report(t1)
    assert report.loc[("T1", "2526"), "home_corners"] == 0.0
    assert report.loc[("T1", "2526"), "odds_home_avg"] == 100.0


def test_load_is_idempotent(e0, tmp_path):
    con = db.connect(tmp_path / "test.duckdb")
    db.load_leagues(con)
    db.load_matches(con, e0)
    db.load_matches(con, e0)  # ikinci yukleme yinelenen satir uretmemeli
    assert con.execute("SELECT count(*) FROM fact_match").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM fact_odds").fetchone()[0] == 20
    assert con.execute("SELECT count(*) FROM dim_team").fetchone()[0] == 4
