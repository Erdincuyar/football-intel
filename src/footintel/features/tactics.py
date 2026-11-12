"""Faz 6: olay verisinden takim-mac bazinda taktik metrikler.

Koordinat kurali (StatsBomb): saha 120 x 80 yard, her olay onu YAPAN takimin soldan
saga hucum ettigi yone gore kaydedilir. x=0 kendi kalemiz, x=120 rakip kale.
Bu yuzden bir takimin "rakip yari sahadaki" savunma hareketi x >= 60 demektir; rakibin
"kendi yari sahasindaki" pasi da (rakibin kendi koordinatinda) x < 60'tir.

Metrikler (hepsi bir takim, bir mac):
- pass_share     : macin paslarindan bu takimin payi. Topa sahip olmanin basit olcusu.
- ppda           : Passes Per Defensive Action. Rakibin kendi %60'lik alanindaki (x < 72)
                   acik oyun paslari / bizim o bolgedeki (x >= 48) savunma hareketlerimiz.
                   KUCUK = yogun pres (rakibe az pas izni veriyoruz).
- def_height     : savunma hareketlerinin (pres, top kapma, mudahale, araya girme)
                   ortalama x'i. Takimin topu nerede geri kazanmaya calistigi.
- high_press_share: preslerin rakip son ucte (x >= 80) yapilan orani.
- field_tilt     : iki takimin son ucte (x >= 80) baslayan paslarindan bu takimin payi.
                   "Saha kimin lehine egik?"
- xg, npxg       : StatsBomb xG (np = penaltisiz). xg_share = npxg / (npxg + rakip npxg).
- npxg_per_shot  : sut kalitesi. Az ama iyi pozisyon mu, cok ama uzak sut mu?
- setpiece_xg_share: korner ve serbest vurus akislarindan gelen xG orani. (Tac atislari
                   dahil degil: StatsBomb tac sonrasi tum hucumu "From Throw In" sayiyor.)
- long_share     : 35 yard ustu acik oyun paslarinin orani (direkt oyun).
- cross_share    : son uctaki acik oyun paslarinin ortaya donusme orani.
- prog_passes    : en az 10 yard ileri giden ve rakip yari sahanin oncesine (x>=60)
                   ulasan basarili acik oyun paslari.
- xt             : expected threat (Karun Singh): basarili pas ve top surmelerin, topu
                   gol olasiligi daha yuksek bolgelere tasimasiyla urettigi tehdit.
- shots, sot, corners, fouls: football-data.co.uk'un da verdigi basit sayilar. Bunlari
                   ayni olay verisinden hesapliyoruz ki zengin metriklerle karsilastirabilelim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LENGTH, WIDTH = 120.0, 80.0
OPEN_PLAY_PASS_TYPES = ["Recovery", "Interception"]   # (+ tipsiz) duran top olmayan paslar
DEF_ACTIONS = {"Pressure", "Interception", "Ball Recovery", "Foul Committed"}
SOT_OUTCOMES = {"Goal", "Saved", "Saved to Post"}
SETPIECE_PATTERNS = {"From Corner", "From Free Kick"}


def _open_play_pass(ev: pd.DataFrame) -> pd.Series:
    pt = ev["pass_type"]
    return (ev["type"] == "Pass") & (pt.isna() | pt.isin(OPEN_PLAY_PASS_TYPES))


def _ppda_actions(ev: pd.DataFrame) -> pd.Series:
    """PPDA payindaki savunma hareketleri: top kapma (tackle), araya girme, faul,
    ve rakibin bizi gectigi ikili mucadele ("Dribbled Past" = denenmis ama kaybedilmis
    mudahale). Pres (Pressure) dahil DEGIL: klasik PPDA tanimi pres verisi olmayan
    saglayicilardan geldigi icin yalnizca topa temas eden hareketleri sayar."""
    t = ev["type"]
    return ((t == "Duel") & (ev["duel_type"] == "Tackle")) | t.isin(
        ["Interception", "Foul Committed", "Dribbled Past"])


# --- Expected threat (xT) ------------------------------------------------------------

def _cells(x: pd.Series, y: pd.Series, nx: int, ny: int) -> np.ndarray:
    cx = np.clip((x.to_numpy(float) / LENGTH * nx).astype(int), 0, nx - 1)
    cy = np.clip((y.to_numpy(float) / WIDTH * ny).astype(int), 0, ny - 1)
    return cx * ny + cy


def fit_xt(ev: pd.DataFrame, nx: int = 16, ny: int = 12, iters: int = 50) -> np.ndarray:
    """Karun Singh'in expected threat (xT) izgarasi.

    Fikir: sahayi nx*ny hucreye bol. Topun bir hucrede oldugu anda takimin gol atma
    olasiligi = (simdi sut atma olasiligi * o hucreden sutun gol olma olasiligi)
               + (topu tasima olasiligi * topun gidecegi hucrelerin xT'sinin ortalamasi).
    Bu ozyinelemeli denklemi tekrar tekrar uygulayarak (deger iterasyonu) cozeriz.
    Basarisiz pas, topu kaybettigimiz icin 0 tehditle sonuclanan bir hareket sayilir.
    Ciktisi: hucre basina "o noktada topa sahip olmanin degeri".
    """
    n = nx * ny
    op = _open_play_pass(ev)
    moves = ev[(op | (ev["type"] == "Carry")) & ev["x"].notna() & ev["end_x"].notna()]
    shots = ev[(ev["type"] == "Shot") & (ev["shot_type"] == "Open Play") & ev["x"].notna()]
    ok = moves["pass_outcome"].isna().to_numpy()
    start = _cells(moves["x"], moves["y"], nx, ny)
    end = _cells(moves["end_x"], moves["end_y"], nx, ny)
    shot_cell = _cells(shots["x"], shots["y"], nx, ny)
    goal = (shots["shot_outcome"] == "Goal").to_numpy(float)

    n_moves = np.bincount(start, minlength=n).astype(float)
    n_shots = np.bincount(shot_cell, minlength=n).astype(float)
    n_goals = np.bincount(shot_cell, weights=goal, minlength=n)
    total = n_moves + n_shots
    with np.errstate(invalid="ignore", divide="ignore"):
        p_shot = np.where(total > 0, n_shots / total, 0.0)
        p_move = np.where(total > 0, n_moves / total, 0.0)
        p_goal = np.where(n_shots > 0, n_goals / n_shots, 0.0)
    trans = np.zeros((n, n))
    np.add.at(trans, (start[ok], end[ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        trans = np.where(n_moves[:, None] > 0, trans / n_moves[:, None], 0.0)
    xt = np.zeros(n)
    for _ in range(iters):
        xt = p_shot * p_goal + p_move * (trans @ xt)
    return xt.reshape(nx, ny)


def xt_added(ev: pd.DataFrame, grid: np.ndarray) -> pd.Series:
    """Basarili acik oyun pasi ve top surmenin xT(bitis) - xT(baslangic) katkisi."""
    nx, ny = grid.shape
    flat = grid.ravel()
    mv = ((_open_play_pass(ev) & ev["pass_outcome"].isna()) | (ev["type"] == "Carry")) & \
        ev["x"].notna() & ev["end_x"].notna()
    out = pd.Series(0.0, index=ev.index)
    m = ev[mv]
    out.loc[m.index] = (flat[_cells(m["end_x"], m["end_y"], nx, ny)]
                        - flat[_cells(m["x"], m["y"], nx, ny)])
    return out


# --- Takim-mac metrikleri ------------------------------------------------------------

def team_match_metrics(ev: pd.DataFrame, matches: pd.DataFrame,
                       xt_grid: np.ndarray | None = None) -> pd.DataFrame:
    """Bir satir = bir takimin bir mactaki metrikleri (her mac icin iki satir)."""
    ev = ev.copy()
    op = _open_play_pass(ev)
    is_pass = ev["type"] == "Pass"
    shot = ev["type"] == "Shot"
    np_shot = shot & (ev["shot_type"] != "Penalty")
    ev["xg"] = ev["shot_xg"].where(shot, 0.0).fillna(0.0)
    ev["npxg"] = ev["xg"].where(np_shot, 0.0)
    flags = pd.DataFrame({
        "match_id": ev["match_id"], "team": ev["team"],
        "passes": is_pass, "passes_ok": is_pass & ev["pass_outcome"].isna(),
        "op_passes": op,
        "op_passes_own60": op & (ev["x"] < 72),          # PPDA payi (rakip icin)
        "ppda_actions": _ppda_actions(ev) & (ev["x"] >= 48),
        "def_n": ev["type"].isin(DEF_ACTIONS) | ((ev["type"] == "Duel")
                                                 & (ev["duel_type"] == "Tackle")),
        "pressures": ev["type"] == "Pressure",
        "pressures_high": (ev["type"] == "Pressure") & (ev["x"] >= 80),
        "counterpress": (ev["type"] == "Pressure") & ev["counterpress"],
        "final3_passes": is_pass & (ev["x"] >= 80),
        "long_passes": op & (ev["pass_length"] >= 35),
        "final3_op_passes": op & (ev["x"] >= 80),
        "crosses": op & ev["pass_cross"] & (ev["x"] >= 80),
        "prog_passes": op & ev["pass_outcome"].isna() & (ev["end_x"] - ev["x"] >= 10)
        & (ev["end_x"] >= 60),
        "shots": shot, "np_shots": np_shot,
        "sot": shot & ev["shot_outcome"].isin(SOT_OUTCOMES),
        "corners": is_pass & (ev["pass_type"] == "Corner"),
        "fouls": ev["type"] == "Foul Committed",
    })
    flags["xg"] = ev["xg"]
    flags["npxg"] = ev["npxg"]
    flags["setpiece_npxg"] = ev["npxg"].where(ev["play_pattern"].isin(SETPIECE_PATTERNS), 0.0)
    flags["def_x_sum"] = ev["x"].where(flags["def_n"], 0.0).fillna(0.0)
    if xt_grid is not None:
        flags["xt"] = xt_added(ev, xt_grid)
    g = flags.groupby(["match_id", "team"]).sum(numeric_only=True)

    # Rakip satirini ayni mac icindeki diger takim olarak eslestir
    side = pd.concat([
        matches[["match_id", "home_team", "away_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "away_team": "opponent", "home_score": "goals",
                     "away_score": "goals_against"}).assign(is_home=1),
        matches[["match_id", "away_team", "home_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "home_team": "opponent", "away_score": "goals",
                     "home_score": "goals_against"}).assign(is_home=0),
    ], ignore_index=True)
    meta = matches[["match_id", "league", "season", "match_date"]]
    t = side.merge(meta, on="match_id").merge(
        g.reset_index(), on=["match_id", "team"], how="left")
    o = g.add_prefix("opp_").reset_index().rename(columns={"team": "opponent"})
    t = t.merge(o, on=["match_id", "opponent"], how="left")

    def share(a, b):
        tot = t[a] + t[b]
        return (t[a] / tot).where(tot > 0)

    out = t[["match_id", "league", "season", "match_date", "team", "opponent", "is_home",
             "goals", "goals_against"]].copy()
    out["points"] = np.select([t["goals"] > t["goals_against"],
                               t["goals"] == t["goals_against"]], [3, 1], 0)
    out["pass_share"] = share("passes", "opp_passes")
    out["pass_completion"] = t["passes_ok"] / t["passes"]
    out["ppda"] = t["opp_op_passes_own60"] / t["ppda_actions"].replace(0, np.nan)
    out["def_height"] = t["def_x_sum"] / t["def_n"]
    out["high_press_share"] = t["pressures_high"] / t["pressures"]
    out["counterpress"] = t["counterpress"]
    out["field_tilt"] = share("final3_passes", "opp_final3_passes")
    out["long_share"] = t["long_passes"] / t["op_passes"]
    out["cross_share"] = t["crosses"] / t["final3_op_passes"].replace(0, np.nan)
    out["prog_passes"] = t["prog_passes"]
    for c in ["xg", "npxg", "shots", "sot", "corners", "fouls"]:
        out[c] = t[c]
        out[f"{c}_against"] = t[f"opp_{c}"]
    out["xg_share"] = share("npxg", "opp_npxg")
    out["npxg_per_shot"] = t["npxg"] / t["np_shots"].replace(0, np.nan)
    out["setpiece_xg_share"] = t["setpiece_npxg"] / t["npxg"].replace(0, np.nan)
    out["shot_share"] = share("shots", "opp_shots")
    out["sot_share"] = share("sot", "opp_sot")
    out["corner_share"] = share("corners", "opp_corners")
    if xt_grid is not None:
        out["xt"] = t["xt"]
        out["xt_share"] = share("xt", "opp_xt")
    return out.sort_values(["match_date", "match_id", "is_home"],
                           ascending=[True, True, False]).reset_index(drop=True)


STYLE_METRICS = ["pass_share", "pass_completion", "ppda", "def_height", "high_press_share",
                 "field_tilt", "long_share", "cross_share", "prog_passes", "npxg_per_shot",
                 "setpiece_xg_share", "xt_share"]
OUTCOME_METRICS = ["xg_share", "shot_share", "sot_share", "corner_share", "fouls"]
