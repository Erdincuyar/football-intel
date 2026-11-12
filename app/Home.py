"""Football Intelligence Platform: ana sayfa (yaklasan maclarin tahminleri).

Calistirma (once veriyi indir, README'deki "Veriyi almak"):
    uv sync --group app
    uv run streamlit run app/Home.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from footintel import ledger
from footintel.config import LEAGUES, LEDGER_DIR

st.set_page_config(page_title="Football Intelligence", layout="wide")
st.title("Football Intelligence Platform")
st.caption(
    "Premier League, La Liga, Super Lig, Serie A ve Ligue 1 icin mac olasiliklari. "
    "Tahminler mactan once degistirilemez deftere yazilir; sonuclar Model Performansi "
    "sayfasinda olculur. Bu bir portfoy projesidir, bahis tavsiyesi degildir."
)


@st.cache_data
def load_ledger() -> pd.DataFrame:
    return ledger.load(LEDGER_DIR)


led = load_ledger()
if led.empty:
    st.info("Defterde henuz tahmin yok. Ilk tahminler haftalik CI calistirmasiyla eklenir.")
    st.stop()

led["match_date"] = pd.to_datetime(led["match_date"])
upcoming = led[led["match_date"] >= pd.Timestamp.today().normalize()]
if upcoming.empty:
    # Milli mac arasi gibi haftalarda defterdeki tum maclar geride kalir. Bos tabloyu
    # filtrelemeye calismak yerine bunu acikca soyluyoruz.
    last = led["match_date"].max().date()
    st.info(f"Su an yaklasan mac yok. Son tahmin edilen maclar {last} tarihindeydi; "
            "yeni fikstur yayinlaninca CI tahminleri otomatik ekler.")
    st.stop()
leagues = {LEAGUES[k].name: k for k in sorted(upcoming["league"].unique())}
choice = st.multiselect("Lig", list(leagues), default=list(leagues))
upcoming = upcoming[upcoming["league"].isin([leagues[c] for c in choice])]

wanted = {("1x2", "home"): "1", ("1x2", "draw"): "X", ("1x2", "away"): "2",
          ("ou25", "over"): "2.5 Ust", ("btts", "yes"): "KG Var",
          ("home_corners_ou45", "over"): "Ev korner 4.5 Ust",
          ("away_corners_ou45", "over"): "Dep korner 4.5 Ust",
          ("corners_ou95", "over"): "Toplam korner 9.5 Ust*"}
keep = pd.Series([(m, s) in wanted for m, s in zip(upcoming["market"], upcoming["selection"],
                                                    strict=True)], index=upcoming.index,
                 dtype=bool)
view = upcoming[keep].copy()
view["col"] = [wanted[(m, s)] for m, s in zip(view["market"], view["selection"], strict=True)]
table = view.pivot_table(index=["match_date", "league", "home_team", "away_team"],
                         columns="col", values="prob").reset_index()
table["league"] = table["league"].map(lambda k: LEAGUES[k].name)
table = table[["match_date", "league", "home_team", "away_team", *wanted.values()]]

st.subheader(f"Yaklasan maclar ({len(table)})")
st.dataframe(
    table, hide_index=True, use_container_width=True,
    column_config={c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0,
                                                      max_value=1)
                   for c in wanted.values()} | {
        "match_date": st.column_config.DateColumn("Tarih"),
        "league": "Lig", "home_team": "Ev", "away_team": "Deplasman"},
)
version = led["model_version"].iloc[-1]
st.caption(f"Model surumu: {version}. Olasiliklar mac oncesi kaydedildi; sonradan degistirilmez.")
st.caption("* Dusuk guven: toplam korner modeli backtest'te lig ortalamasindan anlamli olarak "
           "iyi degil (Faz 5). Takim bazli korner tahminleri anlamli olarak daha iyi.")
