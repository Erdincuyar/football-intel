"""Futbol icin Elo puanlama sistemi (sifirdan).

Fikir: Her takimin bir puani (R) vardir. Mactan once ev sahibinin beklenen skoru

    E = 1 / (1 + 10 ** ((R_deplasman - R_ev - HFA) / 400))

ile hesaplanir. E, "ev sahibinin bu mactan alacagi beklenen puan payidir"
(galibiyet = 1, beraberlik = 0.5, maglubiyet = 0). 400 sabiti olcektir: 400 puan
fark, guclu takimin beklenen skorunun yaklasik 10 kat daha iyi olmasi demektir.

Mactan sonra:

    R_ev' = R_ev + K * G * (S - E)
    R_dep' = R_dep - K * G * (S - E)      (puan toplami korunur)

- S: gercek sonuc (1 / 0.5 / 0).
- K: ogrenme hizi. Buyuk K son maclara cok, kucuk K uzun gecmise agirlik verir.
- G: gol farki carpani. 4-0 kazanmak 1-0 kazanmaktan daha fazla bilgi tasir
  (World Football Elo Ratings'in formulu).
- HFA: ev sahibi avantaji, Elo puani cinsinden.

Iki futbola ozgu ekleme:
- Sezon basinda ortalamaya donus: Yaz transferleri takimlari degistirir; gecen
  sezonun puanina tamamen guvenmek yerine puani lig ortalamasina biraz yaklastiririz.
- Yukselen takimlar: Ligde ilk kez gorunen takim, bir onceki sezon kume dusen
  takimlarin son puanlarinin ortalamasiyla baslar. (Mantik: kume dusenlerle
  yukselenler kabaca ayni seviyededir.)

Veri sizintisi yok: her mac icin MACTAN ONCEKI puanlar kaydedilir. Bu puanlar
yalnizca o tarihten onceki maclardan hesaplanmistir; tahmin ozelligi olarak
guvenle kullanilabilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_RATING = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 60.0
    season_regression: float = 0.2  # 0 = donus yok, 1 = herkes ortalamaya
    use_goal_diff: bool = True


def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - r_home - hfa) / 400.0))


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def run_elo(matches: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """Maclari tarih sirasiyla isler, her mac icin mac oncesi puanlari dondurur.

    Gerekli kolonlar: match_id, league, season, match_date, home_team_id,
    away_team_id, home_goals, away_goals. Her lig ayri bir puan havuzudur.
    """
    params = params or EloParams()
    out = []
    for _league, lg in matches.groupby("league", sort=False):
        lg = lg.sort_values(["match_date", "match_id"])
        ratings: dict[str, float] = {}
        prev_teams: set[str] = set()
        for _season, s in lg.groupby("season", sort=True):
            teams = set(s["home_team_id"]) | set(s["away_team_id"])
            relegated = prev_teams - teams
            newcomer_rating = (
                float(np.mean([ratings[t] for t in relegated])) if relegated else BASE_RATING
            )
            # Sezon basi: once ortalamaya donus (ligde kalanlar), sonra yeni gelenler.
            stayers = teams & prev_teams
            if stayers:
                mean = float(np.mean([ratings[t] for t in stayers]))
                for t in stayers:
                    ratings[t] = mean + (ratings[t] - mean) * (1 - params.season_regression)
            for t in teams - prev_teams:
                ratings[t] = newcomer_rating if prev_teams else BASE_RATING

            for row in s.itertuples(index=False):
                h, a = row.home_team_id, row.away_team_id
                rh, ra = ratings[h], ratings[a]
                e = expected_home(rh, ra, params.home_advantage)
                out.append((row.match_id, rh, ra, e))
                if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                    continue  # oynanmamis mac: puan guncellenmez
                gd = int(row.home_goals) - int(row.away_goals)
                score = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
                g = goal_diff_multiplier(gd) if params.use_goal_diff else 1.0
                delta = params.k * g * (score - e)
                ratings[h] = rh + delta
                ratings[a] = ra - delta
            prev_teams = teams
    return pd.DataFrame(out, columns=["match_id", "home_elo_pre", "away_elo_pre", "home_expected"])


def evaluate(matches: pd.DataFrame, elo: pd.DataFrame, seasons: list[str]) -> float:
    """Beklenen skorun ortalama karesel hatasi (Brier benzeri), secilen sezonlarda.

    Dusuk = iyi. Referans: her zaman 0.5 tahmin eden model ~0.17-0.18 civari verir.
    """
    df = matches[matches["season"].isin(seasons)].merge(elo, on="match_id")
    df = df.dropna(subset=["home_goals", "away_goals"])
    gd = df["home_goals"].astype(int) - df["away_goals"].astype(int)
    score = np.select([gd > 0, gd == 0], [1.0, 0.5], 0.0)
    return float(np.mean((df["home_expected"] - score) ** 2))


def tune(
    matches: pd.DataFrame,
    train_seasons: list[str],
    grid_k: list[float],
    grid_hfa: list[float],
    grid_reg: list[float],
) -> pd.DataFrame:
    """Izgara aramasi: her parametre kombinasyonu icin egitim sezonlarindaki hata.

    Ilk sezon "isinma" sezonudur: herkes 1500'den basladigi icin o sezonun
    tahminleri degerlendirmeye katilmaz (train_seasons bunu disarida birakmali).
    """
    rows = []
    for k in grid_k:
        for hfa in grid_hfa:
            for reg in grid_reg:
                p = EloParams(k=k, home_advantage=hfa, season_regression=reg)
                elo = run_elo(matches, p)
                rows.append({"k": k, "hfa": hfa, "reg": reg,
                             "mse": evaluate(matches, elo, train_seasons)})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
