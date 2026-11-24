"""Faz 11: yaklasan maclar icin tahmin yazilari ve haftanin kuponu.

Girdi: on tahmin tablosu (bir sonraki hafta, tum ligler), oynanmis maclar, tahmin defteri.
Cikti (data/clean, Neon'a Faz 8 yayin adimi tasir):
- match_notes: mac basina 3 satir (tahmin, nedenler, uyari).
- coupon_backtest: ayni kupon kuralinin gecmis haftalardaki sonucu.
- coupon_history: kaydedilmis kuponlarin sonucu.
Kupon kaydi: data/predictions/coupons/<ISO hafta>.parquet. Bir hafta icin kupon bir kez
yazilir ve degistirilmez; CI bu klasoru tahmin defteriyle birlikte commit'ler.

Kullanim:
    uv run python pipelines/build_insights.py
"""

from __future__ import annotations

import logging

import pandas as pd

from footintel import ledger
from footintel.analysis import history_eval as E
from footintel.analysis import insights as I
from footintel.analysis.eda import played
from footintel.config import CLEAN_DIR, CURRENT_SEASON, LEDGER_DIR, RAW_DIR
from footintel.features import fotmob as FF

log = logging.getLogger("build_insights")
COUPON_DIR = LEDGER_DIR / "coupons"
ROUND_DAYS = 4          # bir hafta sonu turu: ilk mactan itibaren Cuma-Pazartesi
FOTMOB_DIR = RAW_DIR / "fotmob"


def fotmob_context(results: pd.DataFrame, preview: pd.DataFrame) -> pd.DataFrame | None:
    """Bu sezonun FotMob verisinden xG formu ve eksik duzenli oyuncular (yoksa None)."""
    def load(kind: str) -> pd.DataFrame:
        files = sorted(FOTMOB_DIR.glob(f"{kind}_*_{CURRENT_SEASON}.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) \
            if files else pd.DataFrame()

    fx, ms, ps = load("fixtures"), load("matches"), load("players")
    if fx.empty or ms.empty:
        log.info("FotMob verisi yok; yazilar onsuz uretilir")
        return None
    up_path = FOTMOB_DIR / "upcoming.parquet"
    up = pd.read_parquet(up_path) if up_path.exists() else None
    try:
        out = FF.for_fixtures(results[results["season"] == CURRENT_SEASON], preview, fx, ms,
                              ps, up)
    except (KeyError, ValueError) as exc:      # resmi olmayan kaynak: bicim degisebilir
        log.warning("FotMob baglami uretilemedi (%s); yazilar onsuz uretilir", exc)
        return None
    log.info("FotMob: %d/%d mac eslesti, %d macta eksik duzenli oyuncu", out["h_xg_for"].notna()
             .sum(), len(preview), int((out[["h_absent", "a_absent"]].notna().any(axis=1)).sum()))
    return out


def notes_table(preview: pd.DataFrame, results: pd.DataFrame, led: pd.DataFrame) -> pd.DataFrame:
    ctx = I.context(results, preview)
    d = preview.merge(ctx, on="match_key", how="left")
    fm = fotmob_context(results, preview)
    if fm is not None:
        d = d.merge(fm, on="match_key", how="left")
    d = I.add_market(d, led)
    rows = []
    for _, r in d.iterrows():
        l1, l2, l3 = I.note(r)
        rows.append({"match_key": r["match_key"], "league": r["league"],
                     "match_date": r["match_date"], "home_team": r["home_team"],
                     "away_team": r["away_team"], "line1": l1, "line2": l2, "line3": l3})
    return pd.DataFrame(rows)


def save_coupon(preview: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame | None:
    """Bu haftanin kuponu yoksa olusturur; varsa dokunmaz (degistirilemez kayit)."""
    d = preview.assign(match_date=pd.to_datetime(preview["match_date"]))
    d = d[d["match_date"] >= today]
    if d.empty:
        return None
    start = d["match_date"].min()
    d = d[d["match_date"] <= start + pd.Timedelta(days=ROUND_DAYS)]
    week = start.strftime("%G-W%V")
    path = COUPON_DIR / f"{week}.parquet"
    if path.exists():
        log.info("Kupon %s zaten kayitli, degistirilmedi", week)
        return pd.read_parquet(path)
    cp = I.pick_coupon(d)
    if cp.empty:
        log.info("Kupon %s: esigi (%%%d) gecen secim yok", week, 100 * I.MIN_PROB)
        return None
    cols = ["match_key", "league", "match_date", "home_team", "away_team", "market",
            "selection", "prob"]
    cp = cp[cols].assign(week=week, combined_prob=cp["prob"].prod(),
                         model_version=d["model_version"].iloc[0],
                         created_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
    COUPON_DIR.mkdir(parents=True, exist_ok=True)
    cp.to_parquet(path, index=False)
    log.info("Kupon %s kaydedildi: %d secim, birlesik olasilik %.2f", week, len(cp),
             cp["combined_prob"].iloc[0])
    return cp


def coupon_history(fd: pd.DataFrame) -> pd.DataFrame:
    """Kaydedilmis kuponlarin sonucu. Mac kimligi defterdekiyle ayni anahtardir."""
    files = sorted(COUPON_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    cp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    from footintel.clean.fixtures import match_key
    res = played(fd).assign(match_key=lambda x: [
        match_key(*r) for r in x[["league", "season", "home_team_id", "away_team_id"]]
        .itertuples(index=False)])[["match_key", "home_goals", "away_goals"]]
    cp = cp.merge(res, on="match_key", how="left")
    cp["won"] = [I.selection_won(m, s, a, b) for m, s, a, b in
                 cp[["market", "selection", "home_goals", "away_goals"]].itertuples(index=False)]
    return cp


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    fd = pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet")
    results = played(fd)
    pv_path = CLEAN_DIR / "preview_predictions.parquet"
    preview = pd.read_parquet(pv_path) if pv_path.exists() else pd.DataFrame()
    led = ledger.load(LEDGER_DIR)

    if not preview.empty:
        notes = notes_table(preview, results, led)
        notes.to_parquet(CLEAN_DIR / "match_notes.parquet", index=False)
        log.info("Tahmin yazisi: %d mac", len(notes))
        warn = notes[notes["line3"].str.contains("sakat veya cezali")]
        log.info("Eksik oyuncu uyarisi: %d mac", len(warn))
        for r in pd.concat([notes.head(3), warn.head(3)]).itertuples(index=False):
            log.info("%s - %s\n  %s\n  %s\n  %s", r.home_team, r.away_team, r.line1, r.line2,
                     r.line3)
        save_coupon(preview, today)

    hist_path = CLEAN_DIR / "historical_predictions.parquet"
    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        hist = hist[hist["model_version"] == hist["model_version"].max()]
        bt = I.coupon_backtest(hist, E.match_results(fd))
        bt.to_parquet(CLEAN_DIR / "coupon_backtest.parquet", index=False)
        if not bt.empty:
            log.info("Kupon geriye donuk (%s): %d hafta, tutan %.1f%%, beklenen %.1f%%, "
                     "secim isabeti %.1f%%", hist["model_version"].iloc[0], len(bt),
                     100 * bt["coupon_won"].mean(), 100 * bt["combined_prob"].mean(),
                     100 * bt["picks_won"].sum() / bt["n_picks"].sum())
    ch = coupon_history(fd)
    if not ch.empty:          # kolonsuz bos tablo veritabanina yazilamaz
        ch.to_parquet(CLEAN_DIR / "coupon_history.parquet", index=False)


if __name__ == "__main__":
    main()
