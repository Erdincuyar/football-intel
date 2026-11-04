"""Model Performansi: backtest (gecmis) ve canli defter (ileriye donuk) sonuclari."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from footintel import ledger
from footintel.analysis.eda import calibration_table, played
from footintel.config import CLEAN_DIR, LEAGUES, LEDGER_DIR
from footintel.models import backtest as B

st.set_page_config(page_title="Model Performansi", layout="wide")
st.title("Model Performansi")


@st.cache_data
def load():
    fd = played(pd.read_parquet(CLEAN_DIR / "matches_football_data.parquet"))
    bt = pd.read_parquet(CLEAN_DIR / "backtest_predictions.parquet")
    return fd, bt


try:
    fd, bt = load()
except FileNotFoundError as exc:
    st.error(f"Veri bulunamadi: {exc}. README'deki 'Veriyi almak' adimini uygula.")
    st.stop()

st.header("Canli defter")
led = ledger.load(LEDGER_DIR)
ev = ledger.evaluate(led, fd)
if ev.empty:
    st.info("Defterdeki maclarin hicbiri henuz oynanmadi.")
else:
    st.dataframe(ledger.market_summary(led, ev).round(4), hide_index=True)
    st.caption("Log loss: kucuk = iyi. Piyasa olasiligi tahmin anindaki ortalama orandan.")

st.header("Backtest (walk-forward)")
season = st.selectbox("Sezon", sorted(bt["season"].unique(), reverse=True))
preds = bt[bt["season"] == season]
ev_bt = B.evaluate(fd, preds, B.base_rates(fd, season))
ev_bt.index = ev_bt.index.map(lambda k: LEAGUES[k].name)
cols = {"1x2_logloss_model": "1X2 model", "1x2_logloss_market": "1X2 piyasa (kapanis)",
        "ou25_logloss_model": "U/A 2.5 model", "ou25_logloss_market": "U/A 2.5 piyasa",
        "btts_logloss_model": "KG model", "btts_logloss_base": "KG lig ortalamasi",
        "corners95_logloss_model": "Korner 9.5 model",
        "corners95_logloss_base": "Korner 9.5 lig ortalamasi"}
st.dataframe(ev_bt[list(cols)].rename(columns=cols).round(4))
st.caption("Her tahmin, o haftadan once oynanan maclarla egitilmis modelden geldi.")

st.subheader("Kalibrasyon")
market = st.radio("Pazar", ["Ev galibiyeti", "2.5 Ust"], horizontal=True)
d = preds.merge(fd[["match_id", "home_goals", "away_goals"]], on="match_id")
if market == "Ev galibiyeti":
    cal = calibration_table(d["p_home"], d["home_goals"] > d["away_goals"])
else:
    cal = calibration_table(d["p_over25"], (d["home_goals"] + d["away_goals"]) > 2.5)
cal = cal[cal["n"] >= 10]
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
    strokeDash=[4, 4], color="gray").encode(x="x", y="y")
pts = alt.Chart(cal).mark_line(point=True).encode(
    x=alt.X("mean_pred", title="Modelin olasiligi", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("observed", title="Gerceklesme orani", scale=alt.Scale(domain=[0, 1])),
    tooltip=["n", "mean_pred", "observed"])
st.altair_chart(diag + pts, use_container_width=True)
st.caption("Noktalar kesik cizgiye yakinsa olasiliklar guvenilirdir.")
