"""Taktik Profil: guncel sezon stil profili (football-data) + StatsBomb 2015/16 metrikleri."""

from __future__ import annotations

# Streamlit Cloud paketi kurmadan calistirir; src/ klasorunu yola ekliyoruz.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import altair as alt
import pandas as pd
import streamlit as st

from footintel import store
from footintel.config import LEAGUES, season_label

st.set_page_config(page_title="Taktik Profil", layout="wide")
st.title("Taktik Profil")

LABELS = {
    "shot_share": "Sut payi", "sot_share": "Isabetli sut payi", "corner_share": "Korner payi",
    "fouls": "Faul/mac", "fouls_against": "Rakip faul/mac", "goal_share": "Gol payi",
    "ppg": "Puan/mac", "xg_share_est": "xG payi (tahmini)", "xt_share_est": "xT payi (tahmini)",
    "field_tilt_est": "Field tilt (tahmini)", "pass_share_est": "Pas payi (tahmini)",
    "pass_share": "Pas payi", "ppda": "PPDA", "def_height": "Savunma yuksekligi (yd)",
    "high_press_share": "Ileri pres orani", "field_tilt": "Field tilt", "xg_share": "xG payi",
    "xt_share": "xT payi", "long_share": "Uzun pas orani", "cross_share": "Orta orani",
    "npxg_per_shot": "xG/sut", "pass_completion": "Pas isabeti", "points": "Puan/mac",
}


@st.cache_data(ttl=3600)
def load_json(name: str):
    return store.read_json(name.removesuffix(".json"))


@st.cache_data(ttl=3600)
def load_table(name: str) -> pd.DataFrame | None:
    return store.read_table(name)


tab_now, tab_sb = st.tabs(["Guncel stil profili", "StatsBomb 2015/16"])

with tab_now:
    st.caption(
        "Kaynak: football-data.co.uk mac istatistikleri (6 lig, Super Lig dahil). "
        "'Tahmini' kolonlar olay verisi olmadan, sut/korner/faul sayilarindan dogrusal bir "
        "modelle hesaplanir. Model StatsBomb 2015/16'da egitildi ve her seferinde bir ligi "
        "disarida birakarak test edildi; hata payi asagida."
    )
    prof = load_table("style_profile")
    if prof is None:
        st.info("Stil profili henuz yok. CI'daki build_tactics adimi uretir.")
    else:
        c1, c2 = st.columns(2)
        lg_names = {LEAGUES[k].name: k for k in sorted(prof["league"].unique()) if k in LEAGUES}
        lg = lg_names[c1.selectbox("Lig", list(lg_names))]
        seasons = sorted(prof["season"].unique(), reverse=True)
        season = c2.selectbox("Sezon", seasons, format_func=season_label)
        view = prof[(prof["league"] == lg) & (prof["season"] == season)]
        est = [c for c in view if c.endswith("_est")]
        cols = ["team", "matches", "ppg", *est, "shot_share", "sot_share", "corner_share",
                "fouls", "fouls_against"]
        table = view[cols].sort_values("ppg", ascending=False).rename(columns=LABELS)
        st.dataframe(table.round(3), hide_index=True, width="stretch")
        proxies = load_json("tactics_proxy.json") or []
        if proxies:
            st.markdown("**Tahmini metriklerin dogrulugu** (baska bir ligde test):")
            st.dataframe(pd.DataFrame([{
                "metrik": LABELS.get(p["target"], p["target"]),
                "R2 (dis lig)": round(p["r2_leave_league_out"], 3),
                "tipik hata": round(p["rmse_leave_league_out"], 3),
                "takimlar arasi yayilim (sd)": round(p["target_sd"], 3)} for p in proxies]),
                hide_index=True)
        if len(view) and est:
            x = st.selectbox("Grafik: x ekseni", est, format_func=lambda c: LABELS.get(c, c))
            chart = alt.Chart(view).mark_circle(size=80).encode(
                x=alt.X(x, title=LABELS.get(x, x), scale=alt.Scale(zero=False)),
                y=alt.Y("ppg", title="Puan/mac"), tooltip=["team", x, "ppg"])
            st.altair_chart(chart + chart.mark_text(dx=8, align="left").encode(text="team"),
                            width="stretch")

with tab_sb:
    st.caption(
        "Kaynak: StatsBomb acik verisi (2015/16 tam sezon: Premier League, La Liga, Serie A, "
        "Ligue 1). Her pas, pres ve sutun sahadaki konumundan hesaplandi. "
        "Data: StatsBomb (github.com/statsbomb/open-data)."
    )
    tm = load_table("tactics_team_match_1516")
    if tm is None:
        st.info("StatsBomb metrikleri henuz yok. CI'daki build_tactics adimi uretir.")
    else:
        metrics = ["pass_share", "ppda", "def_height", "high_press_share", "field_tilt",
                   "xg_share", "xt_share", "long_share", "cross_share", "npxg_per_shot",
                   "pass_completion", "points"]
        metrics = [m for m in metrics if m in tm]
        ts = tm.groupby(["league", "team"])[metrics].mean().reset_index()
        lg_names = {LEAGUES[k].name: k for k in sorted(ts["league"].unique()) if k in LEAGUES}
        lg = lg_names[st.selectbox("Lig ", list(lg_names))]
        view = ts[ts["league"] == lg].sort_values("points", ascending=False)
        st.dataframe(view.drop(columns="league").rename(columns=LABELS).round(3),
                     hide_index=True, width="stretch")
        st.markdown("**Pres ve topa sahip olma**: sag ust = topu tutan ve yogun pres yapan "
                    "takimlar (PPDA kucuk = yogun pres, eksen ters cevrildi).")
        chart = alt.Chart(view).mark_circle(size=80).encode(
            x=alt.X("pass_share", title="Pas payi", scale=alt.Scale(zero=False)),
            y=alt.Y("ppda", title="PPDA", scale=alt.Scale(zero=False, reverse=True)),
            color=alt.Color("points", title="Puan/mac"), tooltip=["team", *metrics])
        st.altair_chart(chart + chart.mark_text(dx=8, align="left").encode(text="team"),
                        width="stretch")

        grid = load_json("xt_grid.json")
        if grid:
            st.markdown("**Expected threat (xT) haritasi**: topun o bolgede olmasinin gol "
                        "degeri. Hucum soldan saga.")
            g = pd.DataFrame([(i, j, v) for i, row in enumerate(grid) for j, v in enumerate(row)],
                             columns=["x", "y", "xT"])
            st.altair_chart(alt.Chart(g).mark_rect().encode(
                x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None, sort="descending"),
                color=alt.Color("xT", scale=alt.Scale(scheme="blues")),
                tooltip=["xT"]).properties(height=300), width="stretch")

        summary = load_json("tactics_summary.json")
        if summary:
            c1, c2 = st.columns(2)
            c1.markdown("**Hangi metrik kalici bir 'stil'?** (sezon guvenilirligi, 1 = tamamen)")
            c1.dataframe(pd.DataFrame(summary["reliability"]).assign(
                metric=lambda d: d["metric"].map(lambda m: LABELS.get(m, m))).round(3),
                hide_index=True)
            c2.markdown("**Ilk yari metrik, ikinci yari puani ne kadar aciklar?** (R2)")
            c2.dataframe(pd.DataFrame(summary["predictive"]).assign(
                metric=lambda d: d["metric"].map(lambda m: LABELS.get(m, m))).round(3),
                hide_index=True)
