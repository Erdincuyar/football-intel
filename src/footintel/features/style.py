"""Faz 7: mac ONCESI stil profili ve stil gruplari (K-Means).

Neden "mac oncesi"? Bir kurali test ederken, o macin kendi istatistigini kullanirsak
gelecegi sizdirmis oluruz: "bu macta cok korner atan takim cok korner atti" bir kural
degil, totolojidir. Bu yuzden her takimin profili, YALNIZCA o mactan onceki son
WINDOW macindan hesaplanir (shift: bugunku mac dahil degil).

Esikler de gecmisten gelir: "korner payi yuksek" = ayni ligin ONCEKI sezonlarindaki
mac oncesi profillerin ust %25'i. Boylece 2023/24'teki bir kural, 2023/24'un
dagilimini bilmeden ateslenir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 10
MIN_PREV = 4
PROFILE_COLS = ["shot_share", "sot_share", "corner_share", "fouls", "fouls_drawn",
                "yellows", "sot_ratio", "sot_ratio_against"]


def add_match_shares(sides: pd.DataFrame) -> pd.DataFrame:
    s = sides.copy()
    for c, a in (("shot", "shots"), ("sot", "sot"), ("corner", "corners")):
        tot = s[a] + s[f"{a}_against"]
        s[f"{c}_share"] = (s[a] / tot).where(tot > 0)
    s["fouls_drawn"] = s["fouls_against"]
    s["sot_ratio"] = (s["sot"] / s["shots"]).where(s["shots"] > 0)
    s["sot_ratio_against"] = (s["sot_against"] / s["shots_against"]).where(
        s["shots_against"] > 0)
    if "yellows" not in s:
        s["yellows"] = np.nan
    return s


def pre_match_profile(sides: pd.DataFrame, window: int = WINDOW,
                      min_prev: int = MIN_PREV) -> pd.DataFrame:
    """Her takim-mac satirina, o mactan ONCEKI son `window` macin ortalamasini ekler.

    Sezon sinirini gecer (yeni sezonun ilk haftalarinda gecen sezonun son maclari kullanilir),
    ama lig sinirini gecmez: kume dusen/cikan takimin baska ligdeki maclari karsilastirilamaz.
    """
    s = add_match_shares(sides).sort_values(["league", "team", "match_date", "match_id"])
    g = s.groupby(["league", "team"])[PROFILE_COLS]
    prev = g.transform(lambda x: x.shift(1).rolling(window, min_periods=min_prev).mean())
    n_prev = s.groupby(["league", "team"]).cumcount()
    out = s.copy()
    for c in PROFILE_COLS:
        out[f"pre_{c}"] = prev[c]
    out["n_prev"] = n_prev.clip(upper=window)
    return out.sort_index()


def attach_opponent(profile: pd.DataFrame) -> pd.DataFrame:
    """Ayni macin rakip satirindaki mac oncesi profili opp_pre_* olarak ekler."""
    cols = [f"pre_{c}" for c in PROFILE_COLS]
    opp = profile[["match_id", "team", *cols]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in cols}})
    p = profile.copy()
    p["opponent"] = p.groupby("match_id")["team"].transform(lambda t: t.iloc[::-1].values)
    return p.merge(opp, on=["match_id", "opponent"], how="left")


def thresholds(profile: pd.DataFrame, season: str, league: str,
               qs: tuple[float, ...] = (0.25, 0.75)) -> pd.DataFrame:
    """Esikler: ayni ligin ONCEKI sezonlarindaki mac oncesi profillerden (sizinti yok)."""
    past = profile[(profile["league"] == league) & (profile["season"] < season)]
    cols = [f"pre_{c}" for c in PROFILE_COLS]
    return past[cols].quantile(list(qs))


def apply_thresholds(df: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    """pre_X ve opp_pre_X icin X_high (>= ust esik) ve X_low (<= alt esik) bayraklari."""
    out = df.copy()
    lo, hi = th.iloc[0], th.iloc[-1]
    for c in PROFILE_COLS:
        for who in ("", "opp_"):
            v, h, low = out[f"{who}pre_{c}"], hi[f"pre_{c}"], lo[f"pre_{c}"]
            ok = v.notna() & pd.notna(h)
            out[f"{who}{c}_high"] = (v >= h).where(ok)
            out[f"{who}{c}_low"] = (v <= low).where(ok)
    return out


def add_flags(profile: pd.DataFrame) -> pd.DataFrame:
    """Her lig-sezon icin esikleri o ligin ONCEKI sezonlarindan alip bayraklari ekler.

    Ilk sezonun gecmisi olmadigi icin bayraklari NaN kalir (kural orada test edilmez).
    """
    parts = []
    for (lg, season), g in profile.groupby(["league", "season"]):
        th = thresholds(profile, season, lg)
        if th.isna().all().all():
            parts.append(g)
            continue
        parts.append(apply_thresholds(g, th))
    return pd.concat(parts).sort_index()


def current_profile(sides: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Her takimin EN SON `window` macinin ortalamasi (bir sonraki mac icin mac oncesi profil)."""
    s = add_match_shares(sides).sort_values(["league", "team", "match_date", "match_id"])
    last = s.groupby(["league", "team"]).tail(window)
    prof = last.groupby(["league", "team"])[PROFILE_COLS].mean().add_prefix("pre_")
    prof["n_prev"] = last.groupby(["league", "team"]).size()
    prof["last_match"] = last.groupby(["league", "team"])["match_date"].max()
    prof["season"] = last.groupby(["league", "team"])["season"].max()
    return prof.reset_index()


def pair_rows(prof: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Iki takimin guncel profillerinden, bir mac icin iki taraf satiri (ev, deplasman)."""
    p = prof.set_index("team")
    cols = [f"pre_{c}" for c in PROFILE_COLS]
    rows = []
    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
        r = {"team": team, "opponent": opp, "is_home": is_home}
        r.update({c: p.at[team, c] for c in cols})
        r.update({f"opp_{c}": p.at[opp, c] for c in cols})
        rows.append(r)
    return pd.DataFrame(rows)


# --- K-Means --------------------------------------------------------------------------

def kmeans(x: np.ndarray, k: int, n_init: int = 20, iters: int = 200,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """K-Means (k-means++ baslangic, birden cok deneme; en dusuk icsel varyans kazanir).

    Algoritma: k merkez sec; her noktayi en yakin merkeze ata; merkezleri o gruplarin
    ortalamasina tasi; atama degismeyene kadar tekrarla. Sonuc baslangica bagli oldugu
    icin n_init kez farkli baslangicla calistirip en iyisini aliyoruz.
    """
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):                       # k-means++: uzak noktalar daha olasi
            d = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(x[rng.choice(len(x), p=d / d.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j]
                            for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((x - c[lab]) ** 2).sum())
        if inertia < best[2]:
            best = (lab, c, inertia)
    return best


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    """Ortalama siluet: bir nokta kendi grubuna, en yakin diger gruptan ne kadar yakin?
    1'e yakin = net ayrisan gruplar, 0 civari = gruplar ic ice."""
    d = np.sqrt(((x[:, None, :] - x[None]) ** 2).sum(-1))
    s = []
    for i in range(len(x)):
        own = lab == lab[i]
        if own.sum() < 2:
            s.append(0.0)
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


CLUSTER_FEATURES = ["shot_share", "sot_ratio", "sot_ratio_against", "corner_share",
                    "fouls", "fouls_drawn"]


def team_season_style(sides: pd.DataFrame) -> pd.DataFrame:
    """Takim-sezon ortalamalari, lig-sezon icinde standartlastirilmis (z-skor).

    Neden lig icinde? Serie A'da faul ortalamasi Premier League'den yuksek. Ham degerle
    gruplarsak "Italyan takimlari" diye bir grup cikar; stil degil lig bulmus oluruz.
    """
    s = add_match_shares(sides)
    ts = s.groupby(["league", "season", "team"])[CLUSTER_FEATURES].mean()
    ts["matches"] = s.groupby(["league", "season", "team"]).size()
    z = ts[CLUSTER_FEATURES].groupby(level=["league", "season"]).transform(
        lambda c: (c - c.mean()) / c.std(ddof=0))
    return ts.join(z.add_prefix("z_")).reset_index()
