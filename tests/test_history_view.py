"""Faz 9 asama 4: Mac Merkezi kart metinleri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.analysis import history_view as V
from tests.test_history_eval import _eval, _preds


def _rows(m):
    return {r["label"]: r for r in V.card_rows(m)}


def test_played_match_card():
    ev = _eval()
    m = _preds().merge(ev[[c for c in ev if c not in _preds() or c == "match_id"]],
                       on="match_id", how="left").set_index("match_id")
    r = _rows(m.loc[10])
    assert r["Mac sonucu (1X2)"]["pred"] == "1 (%50)" and r["Mac sonucu (1X2)"]["mark"]
    assert r["Skor"]["actual"] == "2-1" and r["Skor"]["mark"] is False
    assert r["Toplam gol"]["pred"] == "2.5 → 3" and r["Toplam gol"]["mark"] is True
    assert r["Ilk golu atan"]["pred"] == "N/A"
    r11 = _rows(m.loc[11])
    assert r11["Korner (ev)"]["actual"] == "N/A" and r11["Korner (ev)"]["mark"] is None
    assert r11["Sari kart (toplam)"]["pred"] == "N/A"         # dep beklentisi yok


def test_unplayed_match_has_no_marks():
    m = _preds().set_index("match_id").loc[12]                # degerlendirme kolonu yok
    rows = V.card_rows(m)
    assert all(r["mark"] is None for r in rows)
    assert {r["actual"] for r in rows} <= {"—", "N/A"}
    assert "Sonuc bekleniyor" in V.card_html(m.rename(None).to_frame().T.assign(
        match_id=12).iloc[0])


def test_badge_is_stable_and_escaped():
    assert V.badge("Manchester United") == V.badge("Manchester United")
    assert V.badge("Manchester United")[0] == "MU"
    assert V.badge("Arsenal")[0] == "ARS"
    m = _preds().iloc[2].copy()
    m["home_team"] = "<script>x</script>"
    assert "<script>" not in V.card_html(m)


def test_missing_probabilities_show_na():
    m = _preds().iloc[0].copy()
    m["p_over25"] = np.nan
    assert _rows(m)["2.5 Ust/Alt"]["pred"] == "N/A"
    assert pd.isna(m["p_over25"])
