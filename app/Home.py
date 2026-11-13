"""Football Intelligence Platform: Bu Hafta (yaklasan maclarin tahminleri).

Calistirma (once veriyi indir, README'deki "Veriyi almak"):
    uv sync --group app
    uv run streamlit run app/Home.py

Faz 10: resmi defter ve on tahmin tek tabloda, ustte dort KPI karti.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from footintel import ledger, store  # noqa: E402
from footintel.config import LEDGER_DIR  # noqa: E402

ui.page("Bu Hafta", "Alti lig icin yaklasan maclarin olasiliklari. Resmi tahminler mactan "
        "once degistirilemez deftere yazilir ve basarisi Performans sayfasinda olculur. "
        + ui.DISCLAIMER)

PROB_COLS = {"p_home": "1", "p_draw": "X", "p_away": "2", "p_over25": "2.5 Ust",
             "p_btts": "KG Var"}


@st.cache_data(ttl=3600)
def load():
    return (ledger.load(LEDGER_DIR), store.read_table("preview_predictions"),
            store.read_table("evaluation_summary"))


def ledger_wide(led: pd.DataFrame) -> pd.DataFrame:
    """Defterin uzun formatini mac basina tek satira cevirir (1X2, 2.5, KG, skor, korner)."""
    keys = ["match_key", "match_date", "league", "home_team", "away_team", "model_version"]
    want = {("1x2", "home"): "p_home", ("1x2", "draw"): "p_draw", ("1x2", "away"): "p_away",
            ("ou25", "over"): "p_over25", ("btts", "yes"): "p_btts",
            ("home_corners_ou45", "over"): "p_home_corners45",
            ("away_corners_ou45", "over"): "p_away_corners45",
            ("corners_ou95", "over"): "p_corners_over95"}
    sel = pd.Series([want.get((m, s)) for m, s in zip(led["market"], led["selection"],
                                                      strict=True)], index=led.index)
    w = led[sel.notna()].assign(col=sel.dropna()).pivot_table(
        index=keys, columns="col", values="prob").reset_index()
    cs = led[(led["market"] == "correct_score") & (led["selection"] != "other")]
    if not cs.empty:
        best = cs.loc[cs.groupby("match_key")["prob"].idxmax(), ["match_key", "selection"]]
        w = w.merge(best.rename(columns={"selection": "pred_score"}), on="match_key", how="left")
    return w.assign(status="Resmi")


def season_kpi(summary: pd.DataFrame | None) -> tuple[float | None, float | None, int]:
    """En son sezonun, en yeni model surumunun tum liglerdeki 1X2 isabeti ve piyasa kiyasi."""
    if summary is None or summary.empty:
        return None, None, 0
    s = summary[(summary["scope"] == "season") & (summary["market"] == "result")]
    if s.empty:
        return None, None, 0
    s = s[(s["season"] == s["season"].max())
          & (s["model_version"] == s["model_version"].max()) & (s["n"] > 0)]
    n = int(s["n"].sum())
    if n == 0:
        return None, None, 0
    model = float(s["hits"].sum() / n)
    b = s.dropna(subset=["base_pct"])
    base = float((b["base_pct"] * b["n"]).sum() / b["n"].sum()) if len(b) else None
    return model, base, n


led, pv, summary = load()
today = pd.Timestamp.today().normalize()
parts = []
if not led.empty:
    led = led.assign(match_date=pd.to_datetime(led["match_date"]))
    up = led[led["match_date"] >= today]
    if not up.empty:
        parts.append(ledger_wide(up))
if pv is not None and not pv.empty:
    p = pv[~pv["in_ledger"].astype(bool)]
    if not p.empty:
        parts.append(p.assign(match_date=pd.to_datetime(p["match_date"]), status="On tahmin"))
games = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

model_pct, base_pct, n_eval = season_kpi(summary)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Yaklasan mac", len(games), help="Resmi defter + on tahmin")
k2.metric("Resmi tahmin", int((games.get("status") == "Resmi").sum()) if len(games) else 0,
          help="Degistirilemez deftere yazilmis, sonucu olculecek maclar")
k3.metric("Sezon 1X2 isabeti", ui.pct(model_pct), ui.delta_points(model_pct, base_pct),
          help=f"Bu sezon oynanmis {n_eval} mac; fark piyasa favorisine gore (yuzde puan)",
          delta_color="off" if ui.delta_points(model_pct, base_pct) == "±0 puan" else "normal")
version = (led["model_version"].iloc[-1] if not led.empty else
           (pv["model_version"].iloc[0] if pv is not None and len(pv) else "N/A"))
k4.metric("Model surumu", version)

if games.empty:
    last = led["match_date"].max().date() if not led.empty else None
    st.info("Su an yaklasan mac yok." + (f" Son tahmin edilen maclar {last} tarihindeydi; "
                                          if last else " ")
            + "Yeni fikstur yayinlaninca CI tahminleri otomatik ekler.")
    st.stop()

league = ui.pick("Lig", sorted(games["league"].unique()), "league", all_label="Tum ligler",
                 format_func=ui.league_name)
g = games if league is None else games[games["league"] == league]
g = g.sort_values(["match_date", "league", "home_team"]).copy()
probs = g[["p_home", "p_draw", "p_away"]].to_numpy()
g["Tahmin"] = [f"{'1X2'[i]} · %{100 * row[i]:.0f}" for row, i in zip(probs, probs.argmax(1),
                                                                      strict=True)]
g["Mac"] = g["home_team"] + " – " + g["away_team"]
g["Lig"] = g["league"].map(ui.league_name)
# Telefonda once mac ve tahmin gorunsun; lig secildiyse lig kolonu gereksiz.
cols = ["Mac", "Tahmin", "match_date", *PROB_COLS, "pred_score", "status",
        *(["Lig"] if league is None else [])]
g[list(PROB_COLS)] = 100 * g[list(PROB_COLS)]          # cubuk etiketi %0-100
st.subheader(f"Maclar ({len(g)})")
st.caption("Cubuklar olasiligi gosterir. 'Tahmin' en olasi sonuc ve olasiligi.")
st.dataframe(
    g[[c for c in cols if c in g]].rename(columns=PROB_COLS | {
        "match_date": "Tarih", "pred_score": "Skor", "status": "Durum"}),
    hide_index=True, use_container_width=True,
    column_config={c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0,
                                                      max_value=100)
                   for c in PROB_COLS.values()} | {
        "Tarih": st.column_config.DateColumn("Tarih", format="DD.MM ddd")})
st.caption("Resmi: mactan once deftere yazildi, sonucu olculecek. On tahmin: ayni model, "
           "acik fikstur kaynagindan (openfootball, fixturedownload); her gun yenilenir ve "
           "olculmez. 1X2 dc-mr-v3 modelinden, diger pazarlar Dixon-Coles'tan gelir.")

corner = g[g["status"] == "Resmi"]
if {"p_home_corners45", "p_away_corners45"} <= set(corner.columns) and len(corner):
    with st.expander("Korner tahminleri (resmi maclar)"):
        cc = ["p_home_corners45", "p_away_corners45", "p_corners_over95"]
        st.dataframe(
            corner[["Mac", *cc]].assign(**{c: 100 * corner[c] for c in cc})
            .rename(columns={"p_home_corners45": "Ev 4.5 Ust", "p_away_corners45": "Dep 4.5 Ust",
                             "p_corners_over95": "Toplam 9.5 Ust*"}),
            hide_index=True, use_container_width=True,
            column_config={c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0,
                                                              max_value=100)
                           for c in ("Ev 4.5 Ust", "Dep 4.5 Ust", "Toplam 9.5 Ust*")})
        st.caption("* Dusuk guven: toplam korner modeli lig ortalamasindan anlamli olarak iyi "
                   "degil. Takim bazli korner tahminleri anlamli olarak daha iyi.")
