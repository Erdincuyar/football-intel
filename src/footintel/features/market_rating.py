"""Faz 10: piyasa reytingi, yani takimin GECMIS maclarindaki oranlardan turetilen guc olcusu.

Neden? Bahis piyasasi sakatlik, kadro, transfer ve motivasyon bilgisini fiyata katar;
bizim skor tabanli modelimiz bunlari goremez. Bir macin KENDI oranini girdi yapmak modeli
piyasanin kopyasina cevirir. Bunun yerine takimin onceki maclarinda piyasanin ona bicitigi
beklenen puani kullaniyoruz. Bu bilgi mac gununden once zaten kamuya aciktir.

Hesap (her takim icin, mac sirasina gore):
    xpts = 3 * P(galibiyet) + P(beraberlik)      (kapanis oranlarindan, marj temizlenmis)
    xpts_adj = xpts - ligin ayni sahadaki (ev/deplasman) gecmis ortalamasi
    reyting = onceki maclarin xpts_adj ustel agirlikli ortalamasi (yari omur 3 ve 6 mac)

Saha ortalamasi da yalnizca onceki maclardan hesaplanir; hicbir adim gelecege bakmaz.
Oynanmamis fikstur satirlarinin orani yoktur, bu yuzden reytinge katki yapmazlar ama
kendi reytinglerini onceki maclardan alirlar. Canli tahmin bu sayede ayni fonksiyonla yapilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALFLIVES = (3, 6)
FEATURES = ["d_mr3", "d_mr6"]


def implied(fd: pd.DataFrame) -> pd.DataFrame:
    """Kapanis oranlari; yoksa ortalama (acilis) oranlar. Marj orantili temizlenir."""
    out = {}
    for k in ("home", "draw", "away"):
        close = fd.get(f"odds_{k}_close", pd.Series(np.nan, index=fd.index))
        avg = fd.get(f"odds_{k}_avg", pd.Series(np.nan, index=fd.index))
        out[k] = 1 / close.astype("float64").fillna(avg.astype("float64"))
    p = pd.DataFrame(out)
    return p.div(p.sum(axis=1), axis=0)


def team_ratings(fd: pd.DataFrame) -> pd.DataFrame:
    """(match_id, team, is_home) basina mac oncesi piyasa reytingleri mr3, mr6."""
    p = implied(fd)
    parts = []
    for side, win in (("home", "home"), ("away", "away")):
        r = fd[["match_id", "league", "match_date"]].copy()
        r["team"] = fd[f"{side}_team_id"]
        r["is_home"] = int(side == "home")
        r["xpts"] = 3 * p[win] + p["draw"]
        parts.append(r)
    t = pd.concat(parts, ignore_index=True)
    t["match_date"] = pd.to_datetime(t["match_date"])
    t = t.sort_values(["match_date", "match_id", "is_home"]).reset_index(drop=True)
    venue_mean = t.groupby(["league", "is_home"])["xpts"].transform(
        lambda s: s.shift(1).expanding().mean())
    t["xpts_adj"] = t["xpts"] - venue_mean
    g = t.groupby("team", sort=False)["xpts_adj"]
    for hl in HALFLIVES:
        t[f"mr{hl}"] = g.transform(
            lambda s, hl=hl: s.shift(1).ewm(halflife=hl, ignore_na=True).mean())
    return t[["match_id", "team", "is_home", *[f"mr{h}" for h in HALFLIVES]]]


def match_ratings(fd: pd.DataFrame) -> pd.DataFrame:
    """Mac basina ev - deplasman reyting farki (d_mr3, d_mr6)."""
    t = team_ratings(fd)
    cols = [f"mr{h}" for h in HALFLIVES]
    h = t[t["is_home"] == 1].set_index("match_id")[cols]
    a = t[t["is_home"] == 0].set_index("match_id")[cols]
    out = pd.DataFrame(index=h.index)
    for c in cols:
        out[f"d_{c}"] = h[c] - a[c]
    return out.reset_index()
