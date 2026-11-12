"""Faz 9 asama 2: gecmis maclar icin mac oncesi tahmin tablosu (sonuclardan bagimsiz).

Kaynaklar (hepsi walk-forward; her hafta yalnizca o haftadan onceki maclarla fit edilir):
- backtest_predictions (Faz 3): 1X2, skor matrisi, beklenen goller, 2.5 ust, KG, korner.
- cards_backtest (Faz 9 asama 1): sari kart beklentisi, kirmizi kart olasiligi.

Bu modul gercek sonuclara HIC dokunmaz: maclardan yalnizca kimlik kolonlarini alir
(IDENTITY_COLS). Sonuclarla eslestirme ayri bir adimdir (asama 3). Boylece bir tahminin
yanlislikla sonuctan etkilenmesi yapisal olarak imkansiz.

Korner neden Faz 5 regresyonu degil? O regresyonun katsayilari 2022-25 verisiyle egitildi;
2025/26 icin sorun yok ama sonraki surumlerde gecmis sezonlara uygulanirsa sizinti olur.
Takim korner guc modeli (Faz 3) her hafta yeniden fit edildigi icin her sezonda temiz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footintel.features.matchweek import add_matchweek

HISTORY_VERSION = "hist-v2"            # v1 + ilk yari sonucu modeli
IDENTITY_COLS = ["match_id", "league", "season", "match_date", "home_team", "away_team",
                 "home_team_id", "away_team_id"]
RESULT_WORDS = ("goals", "corners", "yellow", "red", "shots", "fouls", "ht_", "result",
                "odds")
PRED_COLS = ["p_home", "p_draw", "p_away", "pred_score", "p_pred_score", "exp_home_goals",
             "exp_away_goals", "p_over25", "p_btts", "exp_home_corners", "exp_away_corners",
             "exp_corners", "p_corners_over95"]
CARD_COLS = ["exp_home_yellow", "exp_away_yellow", "p_red_any"]
HT_COLS = ["p_ht_home", "p_ht_draw", "p_ht_away", "pred_ht_result"]


def assemble(fd: pd.DataFrame, bt: pd.DataFrame, cards: pd.DataFrame | None,
             created_at: pd.Timestamp, version: str = HISTORY_VERSION,
             halftime: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mac kimligi + hafta + model tahminleri. Gercek sonuc kolonu ICERMEZ."""
    ids = add_matchweek(fd[IDENTITY_COLS])                  # yalnizca kimlik kolonlari
    keep = ["match_id", "fit_date", *[c for c in PRED_COLS if c in bt]]
    out = ids.merge(bt[keep], on="match_id", how="inner")
    if cards is not None and not cards.empty:
        out = out.merge(cards[["match_id", *CARD_COLS]], on="match_id", how="left")
    else:
        for c in CARD_COLS:
            out[c] = np.nan
    if halftime is not None and not halftime.empty:
        out = out.merge(halftime[["match_id", *HT_COLS]], on="match_id", how="left")
    else:
        for c in HT_COLS:
            out[c] = np.nan
    probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
    out["pred_result"] = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
    out["exp_total_goals"] = out["exp_home_goals"] + out["exp_away_goals"]
    out["model_version"] = version
    out["created_at"] = pd.Timestamp(created_at).tz_localize(None)
    return out.drop(columns=["home_team_id", "away_team_id"]).sort_values(
        ["league", "season", "matchweek", "match_date", "match_id"]).reset_index(drop=True)
