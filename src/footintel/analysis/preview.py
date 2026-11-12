"""On tahmin: henuz resmi fikstur listesine (ve oranlara) girmemis bir sonraki hafta.

Resmi tahmin defteri (ledger) degismezdir ve football-data fiksturu yayinlaninca yazilir.
On tahmin ise her calistirmada BASTAN uretilir; arada oynanan maclar ve sakatliklar degil,
yalnizca yeni sonuclar onu degistirir. Bu yuzden basarisi olculmez ve deftere girmez;
arayuzde "on tahmin" etiketiyle gosterilir.
"""

from __future__ import annotations

import pandas as pd

from footintel.clean.fixtures import match_key
from footintel.clean.team_matching import load_aliases, match_teams


def next_round(matches: list[dict], today: pd.Timestamp) -> list[dict]:
    """Bugunden sonraki ilk oynanmamis haftanin maclari (ertelenmis eski maclar haric)."""
    upcoming = [m for m in matches if "score" not in m and pd.Timestamp(m["date"]) >= today]
    if not upcoming:
        return []
    first = min(upcoming, key=lambda m: (m["date"], m.get("time", "")))
    rnd = first.get("round")
    return [m for m in upcoming if m.get("round") == rnd]


def to_fixtures(matches: list[dict], league: str, season: str,
                candidates: dict[str, str], aliases: dict[str, str] | None = None
                ) -> tuple[pd.DataFrame, list[str]]:
    """openfootball maclarini football-data takim kimliklerine esler.

    candidates: {team_id: football-data adi}, bu sezonun takimlari. Eslesmeyen isimli maclar
    atilir (uydurma yok) ve isimler ikinci donus degerinde raporlanir.
    """
    aliases = load_aliases("openfootball") if aliases is None else aliases
    names = sorted({m["team1"] for m in matches} | {m["team2"] for m in matches})
    res = {r.source_name: r.team_id for r in match_teams(names, candidates, aliases)}
    unmatched = [n for n, t in res.items() if t is None]
    rows = []
    for m in matches:
        h, a = res.get(m["team1"]), res.get(m["team2"])
        if h is None or a is None:
            continue
        rows.append({"match_key": match_key(league, season, h, a), "league": league,
                     "season": season, "round": m.get("round"),
                     "match_date": pd.Timestamp(m["date"]).date(),
                     "kickoff_time": m.get("time"), "home_team_id": h, "away_team_id": a,
                     "home_team": candidates[h], "away_team": candidates[a]})
    return pd.DataFrame(rows), unmatched


def prediction_rows(fixtures: pd.DataFrame, preds: dict[str, tuple], official: set[str],
                    version: str, created_at: pd.Timestamp) -> pd.DataFrame:
    """Genis tablo: mac basina 1X2, en olasi skor, gol, 2.5, KG ve korner beklentileri."""
    from footintel.models.strength import correct_score_top

    rows = []
    for fx in fixtures.itertuples(index=False):
        if fx.match_key not in preds:
            continue
        goals, matrix, corners = preds[fx.match_key]
        (score, p_score), = correct_score_top(matrix, 1)
        probs = {"H": goals["p_home"], "D": goals["p_draw"], "A": goals["p_away"]}
        row = {**fx._asdict(), **{k: goals[k] for k in (
            "p_home", "p_draw", "p_away", "exp_home_goals", "exp_away_goals", "p_over25",
            "p_btts")}, "pred_result": max(probs, key=probs.get), "pred_score": score,
            "p_pred_score": p_score, "in_ledger": fx.match_key in official}
        for k in ("exp_home_corners", "exp_away_corners", "p_corners_over95"):
            row[k] = (corners or {}).get(k, float("nan"))
        row["exp_corners"] = row["exp_home_corners"] + row["exp_away_corners"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exp_total_goals"] = out["exp_home_goals"] + out["exp_away_goals"]
        out["model_version"] = version
        out["created_at"] = pd.Timestamp(created_at).tz_localize(None)
    return out
