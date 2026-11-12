"""Performans Paneli: gecmis tahminlerin sezon ve hafta bazinda basarisi (Faz 9)."""

from __future__ import annotations

# Streamlit Cloud paketi kurmadan calistirir; src/ klasorunu yola ekliyoruz.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from footintel import store
from footintel.analysis import history_eval as E
from footintel.analysis import history_view as V
from footintel.config import LEAGUES

MODEL, BASE = "#2a78d6", "#eb6834"          # kategorik sira: 1 mavi, 2 turuncu

st.set_page_config(page_title="Performans Paneli", layout="wide")
st.title("Performans Paneli")
st.caption("Tum oranlar gercek, oynanmis maclardan hesaplanir ve her biri mac sayisiyla (n) "
           "verilir. Kiyas cizgisi olmadan bir yuzde yorumlanamaz: 1X2 ve 2.5 icin piyasa, "
           "digerleri icin ligin onceki sezon ortalamasi.")


@st.cache_data(ttl=3600)
def load():
    return store.read_table("prediction_evaluation")


ev = load()
if ev is None or ev.empty:
    st.info("Degerlendirme tablosu henuz yok. CI'daki 'build_history' adimi uretir.")
    st.stop()

c1, c2, c3 = st.columns([2, 1, 1])
names = {"Tum ligler": None} | {LEAGUES[k].name if k in LEAGUES else k: k
                                for k in sorted(ev["league"].unique())}
league = names[c1.selectbox("Lig", list(names))]
d = ev if league is None else ev[ev["league"] == league]
season = c2.selectbox("Sezon", sorted(d["season"].unique(), reverse=True),
                      format_func=lambda s: f"20{s[:2]}/{s[2:]}")
d = d[d["season"] == season]
version = c3.selectbox("Model surumu", sorted(d["model_version"].unique(), reverse=True))
d = d[d["model_version"] == version]
st.caption(f"{len(d)} oynanmis mac degerlendirildi.")

summary = E.summarize(d.assign(_all=1), ["_all"]).set_index("market")
tiles = [("result", "1X2", False), ("ht_result", "Ilk yari", False), ("ou25", "2.5 Ust/Alt", False),
         ("btts", "KG", False), ("goals_total", "Toplam gol ±1", True),
         ("corners_total", "Korner ±1", True), ("yellow_total", "Sari kart ±1", True)]
cols = st.columns(len(tiles))
for col, (m, label, pm1) in zip(cols, tiles, strict=True):
    if m not in summary.index or summary.loc[m, "n"] == 0:
        col.metric(label, V.NA, help="Bu secimde degerlendirilebilen mac yok")
        continue
    r = summary.loc[m]
    pct = r["pct_pm1"] if pm1 else r["pct"]
    base = r.get("base_pct_pm1" if pm1 else "base_pct")
    diff = 100 * (pct - base) if pd.notna(base) else None
    delta = None if diff is None else ("±0 puan" if abs(diff) < 0.05 else f"{diff:+.1f} puan")
    col.metric(label, f"%{100 * pct:.1f}", delta, help=f"n = {int(r['n'])} mac",
               delta_color="off" if delta == "±0 puan" else "normal")

st.caption("Yesil/kirmizi kutu: kiyasa gore fark (yuzde puan).")

st.subheader("Isabet oranlari")
acc = V.accuracy_table(summary.reset_index())
for c in ("Model %", "Model ±1 %", "Kiyas %", "Kiyas ±1 %"):
    acc[c] = acc[c].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
st.dataframe(acc, hide_index=True, use_container_width=True)
st.caption("– = o tahmin icin tanimli degil (or. sonuc tahmininde ±1) ya da hesaplanamadi. "
           "Kirmizi kartta model "
           "neredeyse hep 'Yok' der; yuksek isabet bu yuzden yaniltici, Brier'e bakin.")

st.subheader("Hafta hafta: sezon icinde kumulatif isabet")
mk = st.selectbox("Tahmin", [m for m in E.MARKETS if m != "score"],
                  format_func=V.MARKET_TR.get)
pm1 = f"{mk}_ok_pm1" in d and st.toggle("±1 tolerans", value=False)
cum = V.cumulative_by_week(d, mk, pm1)
if cum.empty:
    st.info("Bu tahmin icin degerlendirilebilen mac yok (N/A).")
else:
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Model", "Kiyas"], range=[MODEL, BASE]))
    base = alt.Chart(cum).encode(
        x=alt.X("matchweek:Q", title="Hafta", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("pct:Q", title="Kumulatif isabet (%)", scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip("series:N", title="Seri"), alt.Tooltip("matchweek:Q", title="Hafta"),
                 alt.Tooltip("pct:Q", title="%", format=".1f"),
                 alt.Tooltip("hits:Q", title="Isabet"), alt.Tooltip("n:Q", title="n")])
    st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=50, filled=True))
                    .properties(height=320), use_container_width=True)
    st.caption("Ilk haftalarda n kucuk oldugu icin cizgi oynaktir; sezon ilerledikce oturur.")

st.subheader("Olasilik kalitesi ve sayi hatasi")
pm = E.prob_metrics(d.assign(_all=1), ["_all"])
if pm.empty:
    st.info("Hesaplanabilen metrik yok (N/A).")
else:
    pm = pm.assign(Metrik=pm["metric"].map(V.METRIC_TR),
                   Fark=pm["model"] - pm["baseline"])
    pm["Sonuc"] = pm["Fark"].map(lambda x: "model daha iyi" if x < 0 else
                                 ("kiyas daha iyi" if x > 0 else "esit"))
    st.dataframe(pm[["Metrik", "n", "model", "baseline", "baseline_name", "Fark", "Sonuc"]]
                 .rename(columns={"model": "Model", "baseline": "Kiyas",
                                  "baseline_name": "Kiyas turu"}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                for c in ("Model", "Kiyas", "Fark")})
    st.caption("Hepsinde kucuk = iyi. Model ve kiyas ayni maclarda hesaplanir. Tek sezonluk "
               "kucuk farklar sans eseri olabilir; anlamlilik testleri CI raporlarinda.")

if league is None:
    st.subheader("Lig karsilastirmasi")
    lm = st.selectbox("Tahmin ", [m for m in E.MARKETS if m != "score"],
                      format_func=V.MARKET_TR.get, key="league_market")
    s = E.summarize(d, ["league"])
    s = s[(s["market"] == lm) & (s["n"] > 0)]
    if s.empty:
        st.info("Degerlendirilebilen mac yok (N/A).")
    else:
        s = s.assign(Lig=s["league"].map(lambda k: LEAGUES[k].name if k in LEAGUES else k))
        long = pd.concat([
            s.assign(series="Model", pct=100 * s["pct"]),
            s.assign(series="Kiyas", pct=100 * s.get("base_pct", float("nan")))],
            ignore_index=True).dropna(subset=["pct"])
        chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("series:N", title=None, axis=None, sort=["Model", "Kiyas"]),
            x=alt.X("pct:Q", title="Isabet (%)", scale=alt.Scale(domain=[0, 100])),
            row=alt.Row("Lig:N", title=None, header=alt.Header(labelAngle=0,
                                                                labelAlign="left")),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Model", "Kiyas"], range=[MODEL, BASE])),
            tooltip=[alt.Tooltip("Lig:N"), alt.Tooltip("series:N", title="Seri"),
                     alt.Tooltip("pct:Q", title="%", format=".1f"),
                     alt.Tooltip("n:Q", title="n")])
        st.altair_chart(chart.properties(height=40, width=560), use_container_width=False)
        if long["series"].nunique() == 1:
            st.caption("Bu tahmin icin kiyas verisi yok (N/A); yalnizca model gosteriliyor.")
