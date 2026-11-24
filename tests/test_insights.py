"""Faz 11: tahmin yazilari ve haftanin kuponu (sentetik veri)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footintel.analysis import insights as I
from tests.test_history import league
from tests.test_outcome import STRONG, with_odds


def _fixtures(fd: pd.DataFrame) -> pd.DataFrame:
    day = pd.to_datetime(fd["match_date"]).max() + pd.Timedelta(days=7)
    return pd.DataFrame({"match_key": ["k1", "k2"], "league": "E0", "season": "2526",
                         "match_date": day, "home_team_id": [STRONG, "C"],
                         "away_team_id": ["B", "D"]})


def _pred(key: str, league: str, ph: float, po: float = 0.5, pb: float = 0.5) -> dict:
    return {"match_key": key, "league": league, "p_home": ph, "p_draw": (1 - ph) / 2,
            "p_away": (1 - ph) / 2, "p_over25": po, "p_btts": pb}


def test_context_uses_only_past_matches():
    fd = with_odds(league())
    fd["home_shots_on_target"], fd["away_shots_on_target"] = 5.0, 3.0
    c1 = I.context(fd, _fixtures(fd))
    assert set(c1["match_key"]) == {"k1", "k2"}
    assert c1["h_last_n"].eq(6).all()                  # son 6 oynanmis mac sayildi
    # oynanmis maclar degismeden fikstur eklemek/cikarmak baglami degistirmez
    c2 = I.context(fd, _fixtures(fd).iloc[:1])
    pd.testing.assert_frame_equal(c1.iloc[:1].reset_index(drop=True),
                                  c2.reset_index(drop=True), check_like=True)
    # piyasa reytingi favoriyi gosterir: STRONG'un maci esit takimlarin macindan buyuk
    d = c1.set_index("match_key")["d_mr6"]
    assert d["k1"] > 0.3 > abs(d["k2"])


def test_note_has_three_lines_and_no_fake_reasons():
    r = pd.Series({"home_team": "Ev", "away_team": "Dep", "p_home": 0.6, "p_draw": 0.25,
                   "p_away": 0.15, "p_over25": 0.55, "p_btts": 0.4, "pred_score": "2-0",
                   "d_mr6": 0.8, "h_last_pts": 15, "a_last_pts": 4, "h_last_n": 6,
                   "a_last_n": 6})
    lines = I.note(r)
    assert len(lines) == 3
    assert lines[0].startswith("Tahmin: Ev kazanir (%60)")
    assert "Piyasa" in lines[1] and "15 puan" in lines[1]
    assert lines[2].startswith("Dikkat")
    # veri yoksa neden uydurulmaz
    bare = r.drop(["d_mr6", "h_last_pts", "a_last_pts", "h_last_n", "a_last_n"])
    assert I.note(bare)[1].startswith("Iki takimin")


def test_reasons_never_support_the_other_side():
    r = pd.Series({"home_team": "Ev", "away_team": "Dep", "d_mr6": -0.9,
                   "h_last_pts": 3, "a_last_pts": 16, "h_last_n": 6, "a_last_n": 6})
    assert I._reasons(r, side=1) == []                 # gostergeler deplasmani destekliyor
    assert len(I._reasons(r, side=-1)) == 2


def test_coupon_rules():
    preds = pd.DataFrame([_pred("a", "E0", 0.9), _pred("b", "E0", 0.85),
                          _pred("c", "E0", 0.8), _pred("d", "SP1", 0.55, po=0.7),
                          _pred("e", "D1", 0.5)])
    cp = I.pick_coupon(preds)
    assert list(cp["match_key"]) == ["a", "b", "d"]      # lig basina en fazla 2
    assert cp.groupby("match_key").size().max() == 1
    assert cp.loc[cp["match_key"] == "d", "selection"].item() == "2.5 Ust"
    assert (cp["prob"] >= I.MIN_PROB).all()
    weak = pd.DataFrame([_pred("x", "E0", 0.4, po=0.55, pb=0.52)])
    assert I.pick_coupon(weak).empty                    # esigi gecen yoksa kupon yok


@pytest.mark.parametrize(("market", "sel", "hg", "ag", "won"), [
    ("1x2", "1", 2, 0, 1.0), ("1x2", "X", 1, 1, 1.0), ("1x2", "2", 1, 1, 0.0),
    ("ou25", "2.5 Ust", 2, 1, 1.0), ("ou25", "2.5 Alt", 2, 1, 0.0),
    ("btts", "KG Var", 1, 0, 0.0), ("btts", "KG Yok", 0, 0, 1.0),
])
def test_selection_won(market, sel, hg, ag, won):
    assert I.selection_won(market, sel, hg, ag) == won


def test_selection_unplayed_is_nan():
    assert np.isnan(I.selection_won("1x2", "1", np.nan, np.nan))


def test_coupon_backtest_skips_unfinished_weeks():
    days = pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-10"])
    hist = pd.DataFrame([{**_pred(f"m{i}", "E0", 0.8), "match_id": f"m{i}",
                          "match_date": d} for i, d in enumerate(days)]).drop(
        columns="match_key")
    res = pd.DataFrame({"match_id": ["m0", "m1", "m2"], "home_goals": [2, 0, np.nan],
                        "away_goals": [0, 1, np.nan]})
    bt = I.coupon_backtest(hist, res)
    assert len(bt) == 1                                   # m2 oynanmadi, o hafta yok
    row = bt.iloc[0]
    assert row["n_picks"] == 2 and row["picks_won"] == 1 and not row["coupon_won"]
    assert row["combined_prob"] == pytest.approx(0.64)


def test_caution_names_missing_regulars_first():
    r = pd.Series({"home_team": "Ev", "away_team": "Dep", "p_home": 0.6, "p_draw": 0.25,
                   "p_away": 0.15, "p_over25": 0.5, "p_btts": 0.5, "h_missing": 0.05,
                   "a_missing": 0.3, "a_absent": "A, B, C", "a_n_absent": 4})
    line = I.note(r)[2]
    assert line.startswith("Dikkat: Dep takiminda") and "A, B, C ve 1 oyuncu daha" in line
    assert "%30" in line
    # esigin altindaki eksik uyari sayilmaz
    assert "takiminda" not in I.note(r.drop(["a_absent"]))[2]
    assert "takiminda" not in I.note(pd.concat([r.drop("a_missing"),
                                                pd.Series({"a_missing": 0.05})]))[2]
