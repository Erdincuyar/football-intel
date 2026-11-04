"""Degistirilemez tahmin defteri (prediction ledger).

Kurallar:
1. Tahmin MACTAN ONCE yazilir ve bir daha degistirilmez. Her calistirma yeni bir
   Parquet dosyasi ekler (data/predictions/YYYY-MM-DD_<surum>.parquet); var olan
   dosyanin uzerine yazmak hata verir.
2. Bir mac, ayni model surumuyle yalnizca bir kez tahmin edilir (ilk tahmin gecerlidir).
   Boylece "kotu giden tahmini sonradan duzeltmek" mumkun olmaz.
3. Dosyalar GitHub Actions tarafindan git'e commit'lenir. Commit zamani mac saatinden
   once oldugu icin herkes tahminin sonradan degismedigini dogrulayabilir: ucretsiz bir
   zaman damgasi (noter).
4. Degerlendirme (prediction_eval) ayri hesaplanir; defterin kendisi hic degismez.

Bicim: uzun (long) tablo, bir satir = bir pazar secenegi (or. 1x2/home, ou25/over).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.clean.fixtures import match_key

LEDGER_COLUMNS = [
    "prediction_id", "match_key", "league", "season", "match_date", "kickoff_time",
    "home_team", "away_team", "model_version", "market", "selection", "prob",
    "market_prob", "created_at", "data_snapshot_hash",
]
MAX_CORRECT_SCORE = 5   # 0-0 ... 5-5 tek tek, kalan olasilik "other"


class LedgerError(RuntimeError):
    pass


def snapshot_hash(matches: pd.DataFrame) -> str:
    """Modelin egitildigi verinin parmak izi: ayni veri -> ayni hash. Tekrar uretilebilirlik."""
    cols = ["match_id", "home_goals", "away_goals", "home_shots_on_target",
            "away_shots_on_target", "home_corners", "away_corners"]
    d = matches[[c for c in cols if c in matches]].sort_values("match_id").astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()
                          ).hexdigest()[:16]


def prediction_id(key: str, version: str, market: str, selection: str) -> str:
    return hashlib.sha1(f"{key}|{version}|{market}|{selection}".encode()).hexdigest()[:16]


def _market_1x2(fx) -> dict[str, float]:
    odds = [fx.get("odds_home_avg"), fx.get("odds_draw_avg"), fx.get("odds_away_avg")]
    if any(o is None or pd.isna(o) or o <= 1 for o in odds):
        return {}
    inv = np.array([1 / o for o in odds])
    p = inv / inv.sum()
    return {"home": p[0], "draw": p[1], "away": p[2]}


def _market_ou25(fx) -> dict[str, float]:
    o, u = fx.get("odds_over25_avg"), fx.get("odds_under25_avg")
    if any(x is None or pd.isna(x) or x <= 1 for x in (o, u)):
        return {}
    po = (1 / o) / (1 / o + 1 / u)
    return {"over": po, "under": 1 - po}


def match_rows(fx: dict, goals: dict, score_matrix: np.ndarray, corners: dict | None) -> list:
    """Bir mac icin (pazar, secenek, olasilik, piyasa_olasiligi) satirlari."""
    m1x2, mou = _market_1x2(fx), _market_ou25(fx)
    rows = [("1x2", s, goals[f"p_{s}"], m1x2.get(s)) for s in ("home", "draw", "away")]
    for line in ("15", "25", "35"):
        p = goals[f"p_over{line}"]
        mk = mou if line == "25" else {}
        rows += [(f"ou{line}", "over", p, mk.get("over")),
                 (f"ou{line}", "under", 1 - p, mk.get("under"))]
    rows += [("btts", "yes", goals["p_btts"], None), ("btts", "no", 1 - goals["p_btts"], None)]
    for side in ("home", "away"):
        for line in ("05", "15"):
            p = goals[f"p_{side}_over{line}"]
            rows += [(f"{side}_goals_ou{line}", "over", p, None),
                     (f"{side}_goals_ou{line}", "under", 1 - p, None)]
    k = MAX_CORRECT_SCORE + 1
    inner = score_matrix[:k, :k]
    rows += [("correct_score", f"{i}-{j}", inner[i, j], None)
             for i in range(k) for j in range(k)]
    rows.append(("correct_score", "other", max(0.0, 1 - inner.sum()), None))
    if corners:
        for line in ("85", "95", "105", "115"):
            p = corners[f"p_corners_over{line}"]
            rows += [(f"corners_ou{line}", "over", p, None),
                     (f"corners_ou{line}", "under", 1 - p, None)]
        for side in ("home", "away"):
            for line in ("35", "45", "55"):
                key = f"p_{side}_corners_over{line}"
                if key in corners:
                    rows += [(f"{side}_corners_ou{line}", "over", corners[key], None),
                             (f"{side}_corners_ou{line}", "under", 1 - corners[key], None)]
    return rows


def build(fixtures: pd.DataFrame, predictions: dict[str, tuple], version: str,
          created_at: pd.Timestamp, snapshot: str) -> pd.DataFrame:
    """predictions: match_key -> (goal_markets, score_matrix, corner_markets|None)."""
    out = []
    for fx in fixtures.to_dict("records"):
        key = fx["match_key"]
        if key not in predictions:
            continue
        goals, matrix, corners = predictions[key]
        for market, sel, p, mp in match_rows(fx, goals, matrix, corners):
            out.append({
                "prediction_id": prediction_id(key, version, market, sel),
                "match_key": key, "league": fx["league"], "season": fx["season"],
                "match_date": fx["match_date"], "kickoff_time": fx.get("kickoff_time"),
                "home_team": fx["home_team"], "away_team": fx["away_team"],
                "model_version": version, "market": market, "selection": sel,
                "prob": float(p), "market_prob": None if mp is None else float(mp),
                "created_at": created_at, "data_snapshot_hash": snapshot,
            })
    df = pd.DataFrame(out, columns=LEDGER_COLUMNS)
    if not df.empty:
        sums = df.groupby(["match_key", "market"])["prob"].sum()
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise LedgerError(f"Olasiliklar 1'e toplanmiyor: {sums[~np.isclose(sums, 1)].head()}")
    return df


def load(ledger_dir: Path) -> pd.DataFrame:
    files = sorted(ledger_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def already_predicted(ledger_dir: Path, version: str) -> set[str]:
    led = load(ledger_dir)
    return set(led.loc[led["model_version"] == version, "match_key"])


def append(ledger_dir: Path, rows: pd.DataFrame, run_date: str, version: str) -> Path | None:
    """Yeni bir defter dosyasi yazar. Var olan dosyaya ve var olan tahmine dokunmaz."""
    if rows.empty:
        return None
    dup = set(rows["match_key"]) & already_predicted(ledger_dir, version)
    if dup:
        raise LedgerError(f"Bu maclar {version} ile zaten tahmin edilmis: {sorted(dup)[:5]}")
    path = ledger_dir / f"{run_date}_{version}.parquet"
    if path.exists():
        raise LedgerError(f"{path.name} zaten var; defter dosyalarinin uzerine yazilmaz.")
    late = rows[pd.to_datetime(rows["match_date"]) < pd.Timestamp(rows["created_at"].iloc[0]
                                                                  ).normalize()]
    if len(late):
        raise LedgerError(f"Oynanmis maclar icin tahmin yazilamaz: {late['match_key'].unique()}")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    return path


# --- Degerlendirme -----------------------------------------------------------------------

def _outcome(market: str, sel: str, hg: int, ag: int, hc, ac) -> float | None:
    if market == "1x2":
        res = "home" if hg > ag else "draw" if hg == ag else "away"
        return float(sel == res)
    if market == "btts":
        return float((hg > 0 and ag > 0) == (sel == "yes"))
    if market == "correct_score":
        k = MAX_CORRECT_SCORE
        actual = f"{hg}-{ag}" if hg <= k and ag <= k else "other"
        return float(sel == actual)
    if "corners_ou" in market:
        if pd.isna(hc) or pd.isna(ac):
            return None
        side, line = market.split("corners_ou")
        value = {"": hc + ac, "home_": hc, "away_": ac}[side]
        line = int(line) / 10
    elif market.startswith("home_goals_ou"):
        value, line = hg, int(market[len("home_goals_ou"):]) / 10
    elif market.startswith("away_goals_ou"):
        value, line = ag, int(market[len("away_goals_ou"):]) / 10
    elif market.startswith("ou"):
        value, line = hg + ag, int(market[2:]) / 10
    else:
        return None
    return float((value > line) == (sel == "over"))


def evaluate(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """prediction_eval: oynanmis maclar icin sonuc, Brier ve log loss (secenek bazinda)."""
    if ledger.empty:
        return pd.DataFrame(columns=["prediction_id", "outcome", "brier", "log_loss"])
    r = results.dropna(subset=["home_goals", "away_goals"]).copy()
    r["match_key"] = [match_key(*x) for x in r[["league", "season", "home_team_id",
                                                 "away_team_id"]].itertuples(index=False)]
    cols = ["match_key", "home_goals", "away_goals", "home_corners", "away_corners"]
    d = ledger.merge(r[cols], on="match_key")
    d["outcome"] = [
        _outcome(m, s, int(hg), int(ag), hc, ac)
        for m, s, hg, ag, hc, ac in d[["market", "selection", "home_goals", "away_goals",
                                       "home_corners", "away_corners"]].itertuples(index=False)
    ]
    d = d.dropna(subset=["outcome"])
    p = d["prob"].clip(1e-12, 1 - 1e-12)
    d["brier"] = (d["prob"] - d["outcome"]) ** 2
    d["log_loss"] = -(d["outcome"] * np.log(p) + (1 - d["outcome"]) * np.log(1 - p))
    return d[["prediction_id", "outcome", "brier", "log_loss"]].reset_index(drop=True)


def market_summary(ledger: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Pazar bazinda: mac sayisi, modelin ve (varsa) piyasanin log loss'u.

    Cok secenekli pazarda (1X2, dogru skor) log loss = -log(gerceklesen secenegin olasiligi).
    """
    d = ledger.merge(ev, on="prediction_id")
    hit = d[d["outcome"] == 1]
    rows = []
    for market, g in hit.groupby("market"):
        mp = g["market_prob"].astype(float)
        rows.append({
            "market": market, "n_matches": g["match_key"].nunique(),
            "logloss_model": float(-np.log(g["prob"].clip(1e-12)).mean()),
            "logloss_market": float(-np.log(mp.clip(1e-12)).mean()) if mp.notna().all()
            else np.nan,
        })
    return pd.DataFrame(rows)
