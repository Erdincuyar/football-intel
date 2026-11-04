"""Understat ve takim eslestirme testleri. Fixture'lar sentetiktir."""

from pathlib import Path

import pandas as pd
import pytest

from footintel.clean.football_data import clean_matches, read_raw_csv
from footintel.clean.team_matching import match_teams, similarity
from footintel.clean.understat import link_to_football_data, parse_league_season

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "a, b",
    [("Man City", "Manchester City"), ("Man United", "Manchester United"),
     ("Newcastle", "Newcastle United"), ("Milan", "AC Milan"), ("Betis", "Real Betis"),
     ("Parma", "Parma Calcio 1913"), ("Espanol", "Espanyol"), ("Clermont", "Clermont Foot")],
)
def test_similar_names_score_high(a, b):
    assert similarity(a, b) >= 0.72


def test_matching_is_one_to_one_and_avoids_traps():
    # Gercek kulup isimleriyle bir tuzak: "Man City" yalnizca "Manchester City" ile eslesmeli.
    cands = {"man-city": "Man City", "man-united": "Man United", "real-madrid": "Real Madrid",
             "ath-madrid": "Ath Madrid"}
    names = ["Manchester City", "Manchester United", "Real Madrid", "Atletico Madrid"]
    res = {r.source_name: r for r in match_teams(names, cands, {"Atletico Madrid": "ath-madrid"})}
    assert res["Manchester City"].team_id == "man-city"
    assert res["Manchester United"].team_id == "man-united"
    assert res["Real Madrid"].team_id == "real-madrid"
    assert res["Atletico Madrid"].method == "alias"


def test_low_similarity_is_left_unmatched():
    res = match_teams(["Wolverhampton Wanderers"], {"wolves": "Wolves"})
    assert res[0].team_id is None


def test_parse_understat_keeps_only_played_matches():
    df = parse_league_season(FIXTURES / "understat_sample.json", "E0", "2526")
    assert len(df) == 2
    first = df.iloc[0]
    assert first["home_xg"] == pytest.approx(1.85)
    assert first["home_ppda"] == pytest.approx(220 / 25)
    assert pd.isna(df.iloc[1]["home_ppda"])  # def=0 -> PPDA tanimsiz, sifira bolme yok


def test_link_understat_to_football_data():
    fd = clean_matches(read_raw_csv(FIXTURES / "sample_E0.csv"), "E0", "2526")
    us = parse_league_season(FIXTURES / "understat_sample.json", "E0", "2526")
    name_to_id = {"Team Alpha": "team-alpha", "Team Beta": "team-beta",
                  "Team Gamma FC": "team-gamma", "Team Delta": "team-delta"}
    linked, unlinked = link_to_football_data(us, fd, name_to_id)
    assert len(linked) == 2 and unlinked.empty
    assert linked.loc[0, "match_id"] == "E0-2526-20250816-team-alpha-team-beta"


def test_link_detects_wrong_team_mapping():
    fd = clean_matches(read_raw_csv(FIXTURES / "sample_E0.csv"), "E0", "2526")
    us = parse_league_season(FIXTURES / "understat_sample.json", "E0", "2526")
    us.loc[0, "us_home_goals"] = 5  # skor tutmazsa esleme yanlis demektir
    name_to_id = {"Team Alpha": "team-alpha", "Team Beta": "team-beta",
                  "Team Gamma FC": "team-gamma", "Team Delta": "team-delta"}
    with pytest.raises(ValueError, match="skoru farkli"):
        link_to_football_data(us, fd, name_to_id)
