"""Faz 11: FotMob ayristirma ve ozellikleri (sentetik JSON ve tablolar)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.features import fotmob as FF
from footintel.ingest import fotmob as FM


def _player(pid: int, value: float, **kw) -> dict:
    return {"id": pid, "name": f"P{pid}", "marketValue": value, "usualPlayingPositionId": 2,
            **kw}


def _detail(finished: bool = True) -> dict:
    team = {"id": 10, "name": "Ev", "formation": "4-3-3", "coach": {"id": 7, "name": "Hoca"},
            "totalStarterMarketValue": 1e8, "averageStarterAge": 27.5,
            "starters": [_player(1, 5e7)], "subs": [_player(2, 1e7)],
            "unavailable": [_player(3, 2e7, unavailability={"type": "injury",
                                                           "expectedReturn": "Late May"})]}
    away = {"id": 20, "name": "Dep", "unavailable": []}
    stats = {"Periods": {"All": {"stats": [{"stats": [
        {"key": "expected_goals", "stats": ["1.50", "0.70"]},
        {"key": "accurate_passes", "stats": ["438 (87%)", "300 (80%)"]},
        {"key": "opposition_half_passes", "stats": [200, 100]}]}]}}}
    return {"general": {"matchId": "99", "finished": finished},
            "content": {"stats": stats if finished else None,
                        "lineup": {"lineupType": "standard", "homeTeam": team,
                                   "awayTeam": away}}}


def test_parse_match_reads_stats_lineup_and_missing_fields():
    row, players = FM.parse_match(_detail())
    assert row["fm_id"] == 99 and row["home_xg"] == 1.5 and row["away_xg"] == 0.7
    assert row["home_opp_half_passes"] == 200 and np.isnan(row["home_npxg"])  # yok -> NaN
    assert row["home_coach"] == "Hoca" and row["away_coach"] is None
    p = pd.DataFrame(players)
    assert p["role"].tolist() == ["starter", "sub", "unavailable"]
    assert p.loc[p["role"] == "unavailable", "unavailable_type"].item() == "injury"


def test_parse_upcoming_match_has_no_stats():
    row, players = FM.parse_match(_detail(finished=False))
    assert np.isnan(row["home_xg"]) and not row["finished"]
    assert len(players) == 3


def test_num_parses_fotmob_strings():
    assert FM._num("438 (87%)") == 438 and FM._num("3.55") == 3.55
    assert np.isnan(FM._num(None)) and np.isnan(FM._num("-"))
    assert FM.fm_season("2526") == "2025/2026"


# --- Ozellikler -------------------------------------------------------------------------

TEAMS = [10, 20, 30, 40]


def _season(n_rounds: int = 8):
    """4 takimli sentetik lig: fikstur, mac istatistikleri, kadrolar."""
    rng = np.random.default_rng(0)
    fx, ms, ps = [], [], []
    fm_id = 1000
    day = pd.Timestamp("2025-08-01")
    for r in range(n_rounds):
        pairs = [(10, 20), (30, 40)] if r % 2 == 0 else [(20, 30), (40, 10)]
        for h, a in pairs:
            fm_id += 1
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 3))
            fx.append({"fm_id": fm_id, "league": "E0", "season": "2526",
                       "kickoff": day + pd.Timedelta(days=7 * r), "home_fm_id": h,
                       "away_fm_id": a, "finished": True, "cancelled": False,
                       "fm_home_goals": float(hg), "fm_away_goals": float(ag)})
            m = {"fm_id": fm_id}
            for side, t in (("home", h), ("away", a)):
                m |= {f"{side}_npxg": 1.0 + (t == 10), f"{side}_opp_half_passes": 100.0 + t,
                      f"{side}_coach_id": 1 if (t != 20 or r < 5) else 2,
                      f"{side}_xi_value": 1e8}
                for k in range(11):
                    ps.append({"fm_id": fm_id, "team_fm_id": t, "player_id": t * 100 + k,
                               "role": "starter", "market_value": 1e7})
            ms.append(m)
    return pd.DataFrame(fx), pd.DataFrame(ms), pd.DataFrame(ps)


def test_features_use_only_previous_matches():
    fx, ms, ps = _season()
    x1 = FF.match_features(fx, ms, ps).set_index("fm_id")
    last = fx["kickoff"] == fx["kickoff"].max()
    ms2 = ms.copy()
    ms2.loc[ms2["fm_id"].isin(fx.loc[last, "fm_id"]), ["home_npxg", "away_npxg"]] = [9.0, 0.0]
    x2 = FF.match_features(fx, ms2, ps).set_index("fm_id")
    pd.testing.assert_frame_equal(x1, x2)
    first = fx.loc[fx["kickoff"] == fx["kickoff"].min(), "fm_id"]
    assert x1.loc[first, "d_xg_for"].isna().all()          # gecmis yok -> NaN, uydurma yok
    later = x1.loc[fx.loc[fx["home_fm_id"] == 10, "fm_id"].iloc[-1]]
    assert later["d_xg_for"] > 0                             # 10 numara daha cok xG uretiyor


def test_missing_regular_lowers_and_unavailable_list_counts():
    fx, ms, ps = _season()
    last_10 = fx.loc[fx["home_fm_id"].eq(10) | fx["away_fm_id"].eq(10), "fm_id"].iloc[-1]
    # 10 numaranin iki duzenli oyuncusu son macta kadroda yok
    ps2 = ps[~((ps["fm_id"] == last_10) & ps["player_id"].isin([1000, 1001]))]
    x = FF.match_features(fx, ms, ps2).set_index("fm_id")
    side = "h" if fx.set_index("fm_id").loc[last_10, "home_fm_id"] == 10 else "a"
    assert x.loc[last_10, f"{side}_missing"] == 0.2          # 2e7 / 1e8
    # yaklasan mac: sakat listesindeki duzenli oyuncu eksik sayilir
    nxt = fx.iloc[[-1]].assign(fm_id=5000, kickoff=fx["kickoff"].max() + pd.Timedelta(days=7),
                               finished=False, home_fm_id=10, away_fm_id=20)
    up = pd.DataFrame({"fm_id": [5000], "team_fm_id": [10], "player_id": [1003],
                       "role": ["unavailable"]})
    y = FF.match_features(pd.concat([fx, nxt]), ms, ps, up).set_index("fm_id")
    assert y.loc[5000, "h_missing"] == 0.1 and y.loc[5000, "a_missing"] == 0


def test_new_coach_flag():
    fx, ms, ps = _season(12)
    long = FF.team_form(FF.team_long(ms, fx))
    t20 = long[long["team"] == 20].reset_index(drop=True)
    assert t20["new_coach"].iloc[:5].isna().all()             # veri basi: bilinmiyor
    switch = t20.index[t20["coach"] == 2][0]
    assert t20.loc[switch, "new_coach"] == 1 and t20.loc[switch, "coach_games"] == 0


def test_team_map_links_by_date_and_score():
    fx, _, _ = _season()
    names = {10: "a", 20: "b", 30: "c", 40: "d"}
    fd = pd.DataFrame({"match_id": [f"m{i}" for i in range(len(fx))], "league": "E0",
                       "match_date": fx["kickoff"].dt.normalize(),
                       "home_team_id": fx["home_fm_id"].map(names),
                       "away_team_id": fx["away_fm_id"].map(names),
                       "home_goals": fx["fm_home_goals"], "away_goals": fx["fm_away_goals"]})
    tm = FF.team_map(fd, fx).set_index("team")["fm_team"].to_dict()
    assert tm == {v: k for k, v in names.items()}
    ln = FF.link(fd, fx)
    assert len(ln) == len(fx) and (ln["fm_id"].to_numpy() == fx["fm_id"].to_numpy()).all()
