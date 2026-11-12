"""Faz 6 testleri: sentetik olaylar (gercek veri degil), bilinen cevaplarla."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.analysis import tactics as A
from footintel.features import tactics as T
from footintel.ingest.statsbomb import slim_events


def ev(team, type_, x, y=40.0, **kw):
    row = {"match_id": 1, "team": team, "type": type_, "x": x, "y": y, "end_x": None,
           "end_y": None, "pass_type": None, "pass_outcome": None, "pass_length": None,
           "pass_cross": False, "shot_xg": None, "shot_outcome": None, "shot_type": None,
           "duel_type": None, "play_pattern": "Regular Play", "counterpress": False}
    row.update(kw)
    return row


MATCHES = pd.DataFrame([{"match_id": 1, "league": "E0", "season": "1516",
                         "match_date": pd.Timestamp("2015-08-08").date(), "home_team": "A",
                         "away_team": "B", "home_score": 1, "away_score": 0}])


def synthetic_match() -> pd.DataFrame:
    rows = []
    # A: 6 pas, 4'u son ucte (x>=80); B: 2 pas, ikisi kendi yari sahasinda (x<72)
    rows += [ev("A", "Pass", 90, end_x=95, end_y=40) for _ in range(4)]
    rows += [ev("A", "Pass", 30, end_x=50, end_y=40) for _ in range(2)]
    rows += [ev("B", "Pass", 20, end_x=25, end_y=40) for _ in range(2)]
    # A'nin ileri bolgede (x>=48) 1 savunma hareketi -> A icin PPDA = 2 / 1 = 2
    rows.append(ev("A", "Interception", 70))
    rows.append(ev("A", "Shot", 108, shot_xg=0.3, shot_outcome="Goal", shot_type="Open Play"))
    rows.append(ev("A", "Shot", 110, shot_xg=0.76, shot_outcome="Goal", shot_type="Penalty"))
    rows.append(ev("B", "Shot", 100, shot_xg=0.1, shot_outcome="Saved", shot_type="Open Play"))
    rows.append(ev("A", "Pass", 119, end_x=110, end_y=40, pass_type="Corner"))
    return pd.DataFrame(rows)


def test_team_match_metrics_known_values():
    tm = T.team_match_metrics(synthetic_match(), MATCHES).set_index("team")
    assert tm.loc["A", "pass_share"] == pytest.approx(7 / 9)
    assert tm.loc["A", "field_tilt"] == pytest.approx(1.0)       # B'nin son ucte pasi yok
    assert tm.loc["A", "ppda"] == pytest.approx(2.0)
    assert np.isnan(tm.loc["B", "ppda"])                          # B hic savunma yapmadi
    assert tm.loc["A", "npxg"] == pytest.approx(0.3)              # penalti haric
    assert tm.loc["A", "xg_share"] == pytest.approx(0.3 / 0.4)
    assert tm.loc["A", "sot"] == 2 and tm.loc["B", "sot"] == 1
    assert tm.loc["A", "corners"] == 1 and tm.loc["B", "corners_against"] == 1
    assert tm.loc["A", "points"] == 3 and tm.loc["B", "points"] == 0


def test_xt_increases_towards_goal():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(6000):     # ileri paslar; %30'u top kaybi -> uzaktaki top daha az degerli
        x = rng.uniform(0, 110)
        rows.append(ev("A", "Pass", x, rng.uniform(0, 80), end_x=min(x + 15, 119.9),
                       end_y=rng.uniform(0, 80),
                       pass_outcome="Incomplete" if rng.random() < 0.3 else None))
    for _ in range(400):                        # sutlar yalnizca ceza sahasinda
        rows.append(ev("A", "Shot", rng.uniform(102, 119), rng.uniform(25, 55),
                       shot_type="Open Play",
                       shot_outcome="Goal" if rng.random() < 0.15 else "Saved"))
    grid = T.fit_xt(pd.DataFrame(rows))
    box = grid[14:, 4:8].mean()          # ceza sahasinin ortasi (16x12 izgara)
    assert box > grid[8].mean() > grid[0].mean()
    assert grid.max() < 1


def test_reliability_and_proxy():
    rng = np.random.default_rng(1)
    rows = []
    for t in range(20):
        style = rng.normal()
        for k in range(30):
            rows.append({"league": "E0" if t < 10 else "SP1", "season": "1516",
                         "team": f"T{t}", "match_id": t * 100 + k,
                         "match_date": pd.Timestamp("2015-08-01") + pd.Timedelta(days=k),
                         "stable": style + rng.normal(0, 0.1), "noise": rng.normal(),
                         "points": 1})
    tm = pd.DataFrame(rows)
    rel = A.reliability(tm, ["stable", "noise"])
    assert rel.loc["stable", "season_reliability"] > 0.95
    assert abs(rel.loc["noise", "split_half_r"]) < 0.5

    ts = pd.DataFrame({"league": ["E0"] * 10 + ["SP1"] * 10, "x": np.arange(20.0)})
    ts["y"] = 2 * ts["x"] + 1
    fit = A.proxy_fit(ts, "y", ["x"])
    assert fit["r2_leave_league_out"] == pytest.approx(1.0)
    assert fit["beta"] == pytest.approx([1.0, 2.0])


def test_style_profile_from_football_data():
    fd = pd.DataFrame({
        "match_id": [1, 2], "league": "T1", "season": "2627",
        "match_date": pd.to_datetime(["2026-08-01", "2026-08-08"]).date,
        "home_team": ["A", "B"], "away_team": ["B", "A"],
        "home_shots": [15, 5], "away_shots": [5, 15], "home_shots_on_target": [6, 2],
        "away_shots_on_target": [2, 6], "home_corners": [8, 2], "away_corners": [2, 8],
        "home_fouls": [10, 14], "away_fouls": [14, 10], "home_goals": [2, 0],
        "away_goals": [0, 1]})
    proxy = {"target": "field_tilt", "features": ["shot_share"], "beta": [0.0, 1.0]}
    p = A.style_profile(fd, [proxy], min_matches=1).set_index("team")
    assert p.loc["A", "shot_share"] == pytest.approx(0.75)
    assert p.loc["A", "field_tilt_est"] == pytest.approx(0.75)
    assert p.loc["A", "ppg"] == 3 and p.loc["B", "ppg"] == 0
    assert p.loc["A", "fouls"] == pytest.approx(10)


def test_slim_events_flattens_nested_json():
    raw = [
        {"index": 1, "period": 1, "minute": 0, "second": 1, "type": {"name": "Pass"},
         "team": {"name": "A"}, "player": {"name": "p"}, "location": [60, 40],
         "possession": 1, "possession_team": {"name": "A"}, "play_pattern": {"name": "x"},
         "pass": {"end_location": [70, 30], "length": 14.1, "type": {"name": "Corner"},
                  "outcome": {"name": "Incomplete"}, "cross": True}},
        {"index": 2, "period": 1, "minute": 0, "second": 2, "type": {"name": "Ball Receipt*"}},
        {"index": 3, "period": 1, "minute": 0, "second": 3, "type": {"name": "Shot"},
         "team": {"name": "A"}, "location": [110, 40],
         "shot": {"statsbomb_xg": 0.2, "end_location": [120, 40, 1.0],
                  "outcome": {"name": "Goal"}, "type": {"name": "Open Play"}}},
    ]
    df = slim_events(raw, 7)
    assert list(df["type"]) == ["Pass", "Shot"]                  # Ball Receipt atlandi
    p, s = df.iloc[0], df.iloc[1]
    assert (p["end_x"], p["pass_type"], p["pass_outcome"], p["pass_cross"]) == (
        70, "Corner", "Incomplete", True)
    assert (s["shot_xg"], s["end_x"], s["shot_outcome"]) == (0.2, 120, "Goal")
    assert (df["match_id"] == 7).all()
