"""Performans: gecmis tahminlerin basarisi, canli defter, backtest ve model testi.

Faz 10: eski "Model Performansi" ve "Performans Paneli" sayfalari birlesti. Ilk ekranda
KPI kartlari, tek grafik ve tek tablo; ayrintilar acilir bolumlerde.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from footintel import ledger, store  # noqa: E402
from footintel.analysis import history_eval as E  # noqa: E402
from footintel.analysis import history_view as V  # noqa: E402
from footintel.analysis.eda import calibration_table, played  # noqa: E402
from footintel.config import LEDGER_DIR  # noqa: E402
from footintel.models import backtest as B  # noqa: E402

ui.page("Performans", "Tahminler gercekten tuttu mu? Tum oranlar oynanmis maclardan, mac "
        "sayisiyla (n) ve bir kiyasla birlikte verilir: 1X2 ve 2.5 icin piyasa, digerleri icin "
        "ligin onceki sezon ortalamasi.")


@st.cache_data(ttl=3600)
def load():
    return (store.read_table("prediction_evaluation"), store.read_table("matches_football_data"),
            store.read_table("backtest_predictions"), store.read_json("outcome_eval"))


ev, fd, bt, outcome_eval = load()
if ev is None or ev.empty:
    st.info("Degerlendirme tablosu henuz yok. CI'daki 'build_history' adimi uretir.")
    st.stop()

c1, c2, c3 = st.columns([2, 1, 1])
league = ui.pick("Lig", sorted(ev["league"].unique()), "league", c1, all_label="Tum ligler",
                 format_func=ui.league_name)
d = ev if league is None else ev[ev["league"] == league]
season = ui.pick("Sezon", sorted(d["season"].unique(), reverse=True), "season", c2,
                 format_func=ui.season_text)
d = d[d["season"] == season]
version = ui.pick("Model surumu", sorted(d["model_version"].unique(), reverse=True),
                  "version", c3)
d = d[d["model_version"] == version]

summary = E.summarize(d.assign(_all=1), ["_all"]).set_index("market")
pm = E.prob_metrics(d.assign(_all=1), ["_all"])
TILES = [("result", "1X2", False), ("ht_result", "Ilk yari", False),
         ("ou25", "2.5 Ust/Alt", False), ("btts", "KG", False),
         ("goals_total", "Toplam gol ±1", True), ("corners_total", "Korner ±1", True),
         ("yellow_total", "Sari kart ±1", True)]
cols = st.columns(4) + st.columns(4)
for col, (m, label, pm1) in zip(cols, TILES, strict=False):
    if m not in summary.index or summary.loc[m, "n"] == 0:
        col.metric(label, V.NA, help="Bu secimde degerlendirilebilen mac yok")
        continue
    r = summary.loc[m]
    val = r["pct_pm1"] if pm1 else r["pct"]
    delta = ui.delta_points(val, r.get("base_pct_pm1" if pm1 else "base_pct"))
    col.metric(label, ui.pct(val), delta, help=f"n = {int(r['n'])} mac",
               delta_color="off" if delta == "±0 puan" else "normal")
ll = pm.set_index("metric").loc["logloss_1x2"] if "logloss_1x2" in set(pm.get("metric", [])) \
    else None
if ll is not None and pd.notna(ll["model"]):
    diff = ll["model"] - ll["baseline"]
    cols[7].metric("Log loss 1X2", f"{ll['model']:.3f}",
                   None if pd.isna(diff) else f"{diff:+.3f} piyasaya gore",
                   delta_color="inverse", help=f"Kucuk = iyi. n = {int(ll['n'])} mac")
else:
    cols[7].metric("Log loss 1X2", V.NA)
st.caption(f"{len(d)} oynanmis mac. Kutudaki fark kiyasa gore (yuzde puan); yesil = model "
           "daha iyi.")

st.subheader("Sezon boyunca isabet")
mk = st.selectbox("Tahmin", [m for m in E.MARKETS if m != "score"],
                  format_func=V.MARKET_TR.get)
pm1 = f"{mk}_ok_pm1" in d and st.toggle("±1 tolerans", value=False)
cum = V.cumulative_by_week(d, mk, pm1)
if cum.empty:
    st.info("Bu tahmin icin degerlendirilebilen mac yok (N/A).")
else:
    st.caption("Her hafta, o haftaya kadarki tum maclarin isabeti. Mavi model, turuncu kiyas. "
               "Ilk haftalarda az mac oldugu icin cizgi oynaktir.")
    enc = alt.Chart(cum).encode(
        x=alt.X("matchweek:Q", title="Hafta", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("pct:Q", title="Kumulatif isabet (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Model", "Kiyas"], range=[ui.MODEL, ui.BASE])),
        tooltip=[alt.Tooltip("series:N", title="Seri"), alt.Tooltip("matchweek:Q", title="Hafta"),
                 alt.Tooltip("pct:Q", title="%", format=".1f"),
                 alt.Tooltip("hits:Q", title="Isabet"), alt.Tooltip("n:Q", title="n")])
    st.altair_chart((enc.mark_line(strokeWidth=2) + enc.mark_point(size=40, filled=True))
                    .properties(height=300), use_container_width=True)

st.subheader("Tum tahminler")
acc = V.accuracy_table(summary.reset_index())
for c in ("Model %", "Model ±1 %", "Kiyas %", "Kiyas ±1 %"):
    acc[c] = acc[c].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
st.dataframe(acc, hide_index=True, use_container_width=True)
st.caption("– = o tahmin icin tanimli degil ya da hesaplanamadi. Kirmizi kartta model "
           "neredeyse hep 'Yok' der; yuksek isabet bu yuzden yaniltici, Brier'e bakin.")

with st.expander("Olasilik kalitesi: Brier, log loss, sayi hatasi"):
    if pm.empty:
        st.info("Hesaplanabilen metrik yok (N/A).")
    else:
        t = pm.assign(Metrik=pm["metric"].map(V.METRIC_TR), Fark=pm["model"] - pm["baseline"])
        t["Sonuc"] = t["Fark"].map(lambda x: "model daha iyi" if x < 0 else
                                   ("kiyas daha iyi" if x > 0 else "esit"))
        st.dataframe(t[["Metrik", "n", "model", "baseline", "baseline_name", "Fark", "Sonuc"]]
                     .rename(columns={"model": "Model", "baseline": "Kiyas",
                                      "baseline_name": "Kiyas turu"}),
                     hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                    for c in ("Model", "Kiyas", "Fark")})
        st.caption("Hepsinde kucuk = iyi. Model ve kiyas ayni maclarda hesaplanir.")

if league is None:
    with st.expander("Lig karsilastirmasi"):
        lm = st.selectbox("Tahmin ", [m for m in E.MARKETS if m != "score"],
                          format_func=V.MARKET_TR.get, key="league_market")
        s = E.summarize(d, ["league"])
        s = s[(s["market"] == lm) & (s["n"] > 0)]
        if s.empty:
            st.info("Degerlendirilebilen mac yok (N/A).")
        else:
            t = pd.DataFrame({"Lig": s["league"].map(ui.league_name), "n": s["n"],
                              "Model %": 100 * s["pct"],
                              "Kiyas %": 100 * s.get("base_pct", float("nan"))})
            t["Fark (puan)"] = t["Model %"] - t["Kiyas %"]
            st.dataframe(t.sort_values("Model %", ascending=False), hide_index=True,
                         use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(c, format="%.1f")
                                        for c in ("Model %", "Kiyas %", "Fark (puan)")})

with st.expander("Faz 10 model testi: dc-mr-v3, Dixon-Coles ve piyasa"):
    if not outcome_eval:
        st.info("Model testi henuz yok. CI'daki 'build_outcome' adimi uretir.")
    else:
        o = outcome_eval
        a, b, c = st.columns(3)
        a.metric("Dixon-Coles", f"{o['logloss_dc']:.4f}", help="1X2 log loss, kucuk = iyi")
        b.metric("dc-mr-v3", f"{o['logloss_new']:.4f}", f"{o['diff_vs_dc'][0]:+.4f}",
                 delta_color="inverse")
        c.metric("Piyasa (kapanis)", f"{o['logloss_market']:.4f}")
        lo, hi = o["diff_vs_dc"][1], o["diff_vs_dc"][2]
        st.caption(f"Test sezonlari {', '.join(map(ui.season_text, o['test_seasons']))}, "
                   f"n = {o['n']} mac, egitimde gorulmedi. Fark %95 aralik [{lo:+.4f}, "
                   f"{hi:+.4f}]; kapi {'gecti' if o['gate_passed'] else 'GECMEDI'}. "
                   "Yeni model takimin onceki maclarindaki oranlardan bir 'piyasa reytingi' "
                   "kullanir, bu macin oranini kullanmaz.")

with st.expander("Canli defter (ileriye donuk tahminler)"):
    fdp = played(fd) if fd is not None else None
    led = ledger.load(LEDGER_DIR)
    lev = ledger.evaluate(led, fdp) if fdp is not None and not led.empty else pd.DataFrame()
    if lev.empty:
        st.info("Defterdeki maclarin hicbiri henuz oynanmadi.")
    else:
        st.dataframe(ledger.market_summary(led, lev).round(4), hide_index=True,
                     use_container_width=True)
        st.caption("Log loss: kucuk = iyi. Piyasa olasiligi tahmin anindaki ortalama orandan.")

with st.expander("Backtest ve kalibrasyon (Dixon-Coles, walk-forward)"):
    if fd is None or bt is None:
        st.info("Backtest verisi yok.")
    else:
        fdp = played(fd)
        bseason = st.selectbox("Sezon ", sorted(bt["season"].unique(), reverse=True),
                               format_func=ui.season_text)
        preds = bt[bt["season"] == bseason]
        ev_bt = B.evaluate(fdp, preds, B.base_rates(fdp, bseason))
        ev_bt.index = ev_bt.index.map(ui.league_name)
        names = {"1x2_logloss_model": "1X2 model", "1x2_logloss_market": "1X2 piyasa",
                 "ou25_logloss_model": "2.5 model", "ou25_logloss_market": "2.5 piyasa",
                 "btts_logloss_model": "KG model", "btts_logloss_base": "KG lig ort."}
        st.dataframe(ev_bt[[c for c in names if c in ev_bt]].rename(columns=names).round(4),
                     use_container_width=True)
        market = st.radio("Kalibrasyon", ["Ev galibiyeti", "2.5 Ust"], horizontal=True)
        m = preds.merge(fdp[["match_id", "home_goals", "away_goals"]], on="match_id")
        if market == "Ev galibiyeti":
            cal = calibration_table(m["p_home"], m["home_goals"] > m["away_goals"])
        else:
            cal = calibration_table(m["p_over25"], (m["home_goals"] + m["away_goals"]) > 2.5)
        cal = cal[cal["n"] >= 10]
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color=ui.MUTED).encode(x="x", y="y")
        pts = alt.Chart(cal).mark_line(point=True, color=ui.MODEL).encode(
            x=alt.X("mean_pred", title="Modelin olasiligi", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed", title="Gerceklesme orani", scale=alt.Scale(domain=[0, 1])),
            tooltip=["n", "mean_pred", "observed"])
        st.altair_chart((diag + pts).properties(height=280), use_container_width=True)
        st.caption("Noktalar kesik cizgiye yakinsa olasiliklar guvenilirdir.")
