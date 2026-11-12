"""Football Intelligence Platform: ana sayfa (yaklasan maclarin tahminleri).

Calistirma (once veriyi indir, README'deki "Veriyi almak"):
    uv sync --group app
    uv run streamlit run app/Home.py
"""

from __future__ import annotations

# Streamlit Cloud paketi kurmadan calistirir; src/ klasorunu yola ekliyoruz.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from footintel import ledger, store
from footintel.config import LEAGUES, LEDGER_DIR

st.set_page_config(page_title="Football Intelligence", layout="wide")
st.title("Football Intelligence Platform")
st.caption(
    "Premier League, La Liga, Bundesliga, Super Lig, Serie A ve Ligue 1 icin mac olasiliklari. "
    "Tahminler mactan once degistirilemez deftere yazilir; sonuclar Model Performansi "
    "sayfasinda olculur. Bu bir portfoy projesidir, bahis tavsiyesi degildir."
)


@st.cache_data
def load_ledger() -> pd.DataFrame:
    return ledger.load(LEDGER_DIR)


@st.cache_data(ttl=3600)
def load_preview() -> pd.DataFrame | None:
    return store.read_table("preview_predictions")


def show_preview() -> None:
    """Bir sonraki haftanin on tahmini (resmi deftere henuz girmemis maclar)."""
    pv = load_preview()
    if pv is None or pv.empty:
        return
    pv = pv[~pv["in_ledger"].astype(bool)].copy()
    if pv.empty:
        return
    pv["league"] = pv["league"].map(lambda k: LEAGUES[k].name if k in LEAGUES else k)
    cols = {"match_date": "Tarih", "league": "Lig", "home_team": "Ev", "away_team": "Deplasman",
            "p_home": "1", "p_draw": "X", "p_away": "2", "pred_score": "En olasi skor",
            "p_over25": "2.5 Ust", "p_btts": "KG Var", "exp_corners": "Korner beklentisi"}
    pv = pv.sort_values(["match_date", "league", "home_team"])[list(cols)].rename(columns=cols)
    st.subheader(f"Onumuzdeki hafta: on tahmin ({len(pv)} mac)")
    st.dataframe(
        pv, hide_index=True, use_container_width=True,
        column_config={c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0,
                                                          max_value=1)
                       for c in ("1", "X", "2", "2.5 Ust", "KG Var")} | {
            "Tarih": st.column_config.DateColumn("Tarih"),
            "Korner beklentisi": st.column_config.NumberColumn(format="%.1f")})
    st.caption("On tahmin: ayni model, bugune kadarki sonuclarla. Fikstur kaynagi openfootball "
               "ve fixturedownload (Super Lig), ikisi de acik ve ucretsiz. Resmi tahmin, mac "
               "football-data fikstur listesine girince degistirilemez deftere yazilir ve "
               "basarisi olculur; on tahmin olculmez, her gun yenilenir.")


led = load_ledger()
if led.empty:
    st.info("Defterde henuz tahmin yok. Ilk tahminler haftalik CI calistirmasiyla eklenir.")
    show_preview()
    st.stop()

led["match_date"] = pd.to_datetime(led["match_date"])
upcoming = led[led["match_date"] >= pd.Timestamp.today().normalize()]
if upcoming.empty:
    # Milli mac arasi gibi haftalarda defterdeki tum maclar geride kalir. Bos tabloyu
    # filtrelemeye calismak yerine bunu acikca soyluyoruz.
    last = led["match_date"].max().date()
    st.info(f"Su an yaklasan mac yok. Son tahmin edilen maclar {last} tarihindeydi; "
            "yeni fikstur yayinlaninca CI tahminleri otomatik ekler.")
    show_preview()
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
show_preview()
