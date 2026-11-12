"""On tahmin: openfootball fiksturu, hafta secimi, takim eslestirme."""

from __future__ import annotations

import pandas as pd

from footintel.analysis import preview as P
from footintel.ingest import openfootball as OF
from tests.test_ledger import fake_predictions

MATCHES = [
    {"round": "Matchday 10", "date": "2026-11-08", "team1": "Arsenal FC", "team2": "Chelsea FC",
     "score": {"ft": [1, 0]}},
    {"round": "Matchday 9", "date": "2026-11-10", "team1": "Everton FC",     # ertelenmis,
     "team2": "Fulham FC"},                                                  # gecmiste kaldi
    {"round": "Matchday 11", "date": "2026-11-22", "team1": "Chelsea FC",
     "team2": "Manchester City FC", "time": "16:30"},
    {"round": "Matchday 11", "date": "2026-11-21", "team1": "Nottingham Forest FC",
     "team2": "Arsenal FC", "time": "15:00"},
    {"round": "Matchday 11", "date": "2026-11-21", "team1": "Bilinmeyen FC",
     "team2": "Everton FC"},
    {"round": "Matchday 12", "date": "2026-11-28", "team1": "Arsenal FC",
     "team2": "Everton FC"},
]
CANDS = {"arsenal": "Arsenal", "chelsea": "Chelsea", "man-city": "Man City",
         "nott-m-forest": "Nott'm Forest", "everton": "Everton", "fulham": "Fulham"}


def test_season_dir():
    assert OF.season_dir("2627") == "2026-27"


def test_next_round_is_first_unplayed_round_after_today():
    rnd = P.next_round(MATCHES, pd.Timestamp("2026-11-12"))
    assert {m["round"] for m in rnd} == {"Matchday 11"} and len(rnd) == 3
    assert P.next_round([MATCHES[0]], pd.Timestamp("2026-11-12")) == []


def test_unknown_team_is_skipped_not_invented():
    rnd = P.next_round(MATCHES, pd.Timestamp("2026-11-12"))
    fx, unmatched = P.to_fixtures(rnd, "E0", "2627", CANDS,
                                  aliases={"Nottingham Forest FC": "nott-m-forest"})
    assert unmatched == ["Bilinmeyen FC"]
    assert set(fx["match_key"]) == {"E0-2627-chelsea-man-city",
                                    "E0-2627-nott-m-forest-arsenal"}
    assert fx.set_index("home_team_id").loc["chelsea", "away_team"] == "Man City"


def test_prediction_rows_marks_official_and_probabilities():
    rnd = P.next_round(MATCHES, pd.Timestamp("2026-11-12"))
    fx, _ = P.to_fixtures(rnd, "E0", "2627", CANDS,
                          aliases={"Nottingham Forest FC": "nott-m-forest"})
    out = P.prediction_rows(fx, fake_predictions(fx), {"E0-2627-chelsea-man-city"}, "v-test",
                            pd.Timestamp("2026-11-12 16:00"))
    assert len(out) == 2
    assert out.set_index("match_key")["in_ledger"].to_dict() == {
        "E0-2627-chelsea-man-city": True, "E0-2627-nott-m-forest-arsenal": False}
    s = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert ((s - 1).abs() < 1e-6).all()
    assert out["pred_score"].str.match(r"^\d+-\d+$").all()


def test_fixturedownload_is_converted_to_openfootball_shape():
    rows = [{"RoundNumber": 13, "DateUtc": "2026-11-22 16:00:00Z", "HomeTeam": "Galatasaray",
             "AwayTeam": "Çaykur Rizespor", "HomeTeamScore": None, "AwayTeamScore": None},
            {"RoundNumber": 12, "DateUtc": "2026-11-08 13:30:00Z", "HomeTeam": "Rizespor",
             "AwayTeam": "Galatasaray", "HomeTeamScore": 0, "AwayTeamScore": 2}]
    m = OF.from_fixturedownload(rows)
    assert m[0] == {"round": "Hafta 13", "date": "2026-11-22", "time": "16:00 UTC",
                    "team1": "Galatasaray", "team2": "Çaykur Rizespor"}
    assert m[1]["score"] == {"ft": [0, 2]}
    rnd = P.next_round(m, pd.Timestamp("2026-11-12"))
    fx, unmatched = P.to_fixtures(rnd, "T1", "2627",
                                  {"galatasaray": "Galatasaray", "rizespor": "Rizespor"},
                                  aliases={})
    assert unmatched == [] and fx["match_key"].tolist() == ["T1-2627-galatasaray-rizespor"]
