"""Faz 9: hafta numarasi.

football-data.co.uk'ta hafta (round) bilgisi yok. Her takimin sezondaki kacinci macini
oynadigini sayiyoruz; bir macin haftasi = iki takimin o maca kadar oynadigi mac sayisinin
buyugu. Normal bir haftada ikisi ayni sayidir. Ertelenen bir mac, resmi haftasinda degil
OYNANDIGI siraya gore yer alir (or. 8. haftanin maci 12. haftadan sonra oynandiysa 12-13.
hafta civarinda gorunur). Bu bir yaklasimdir ve arayuzde boyle yazilir.
"""

from __future__ import annotations

import pandas as pd


def add_matchweek(fd: pd.DataFrame) -> pd.DataFrame:
    """Her maca `matchweek` (1'den baslar) ekler. Oynanmamis maclar da sayilir (fikstur)."""
    d = fd.copy()
    order = d.assign(_date=pd.to_datetime(d["match_date"])).sort_values(
        ["_date", "match_id"]).index
    played: dict[tuple, int] = {}
    week = {}
    for i in order:
        lg, season = d.at[i, "league"], d.at[i, "season"]
        h = (lg, season, d.at[i, "home_team_id"])
        a = (lg, season, d.at[i, "away_team_id"])
        played[h] = played.get(h, 0) + 1
        played[a] = played.get(a, 0) + 1
        week[i] = max(played[h], played[a])
    d["matchweek"] = pd.Series(week).reindex(d.index).astype(int)
    return d
