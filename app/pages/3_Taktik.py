"""Taktik: iki takimin eslesmesi, guncel stil profili ve StatsBomb 2015/16 metrikleri.

Faz 10: eski "Taktik Profil" ve "Taktiksel Eslesme" sayfalari ayni veriyi kullaniyordu;
tek sayfada uc sekme oldu. En cok kullanilan eslesme ilk sekmede.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from footintel import store  # noqa: E402
from footintel.analysis import rules as R  # noqa: E402
from footintel.config import LEAGUES, season_label  # noqa: E402
from footintel.features import style as ST  # noqa: E402

ui.page("Taktik", "Takimlar nasil oynuyor ve iki takim karsilastiginda ne beklenir? Profiller "
        "mac istatistiklerinden; eslesme kurallari gecmis verilerde test edildi.")

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


MATCH_LABELS = {"shot_share": "Sut payi", "sot_share": "Isabetli sut payi",
                "corner_share": "Korner payi", "fouls": "Faul/mac",
                "fouls_drawn": "Cekilen faul/mac", "yellows": "Sari kart/mac",
                "sot_ratio": "Isabet orani", "sot_ratio_against": "Rakibin isabet orani"}
TARGET_TR = {"res_goals": "gol", "res_goals_against": "yenilen gol", "res_corners": "korner",
             "res_fouls": "faul", "res_yellows": "sari kart"}


@st.cache_data(ttl=3600)
def load_json(name: str):
    return store.read_json(name.removesuffix(".json"))


@st.cache_data(ttl=3600)
def load_table(name: str) -> pd.DataFrame | None:
    return store.read_table(name)


def matchup() -> None:
    st.caption(
        "Her takimin profili son 10 macindan (sut, isabetli sut, korner, faul, kart). "
        "'Yuksek' ve 'dusuk', ayni ligin onceki sezonlarinin ust ve alt %25'i. Kurallar "
        "onceden yazildi ve modelin artigina karsi test edildi: 2022-24'te kesif, 2024-26'da "
        "dogrulama. Yalnizca iki donemde de tutan kurallar 'gecerli' sayilir."
    )
    now = load_table("style_now")
    if now is None:
        st.info("Profil verisi yok. CI'daki build_matchup adimi uretir.")
        return
    results = pd.DataFrame(load_json("matchup_rules.json") or [])
    th_all = load_json("style_thresholds.json") or {}

    leagues = {LEAGUES[k].name: k for k in sorted(now["league"].unique()) if k in LEAGUES}
    lg = leagues[ui.pick("Lig", list(leagues), "taktik_league", format_func=str)]
    teams = sorted(now.loc[now["league"] == lg, "team"])
    c1, c2 = st.columns(2)
    home = c1.selectbox("Ev sahibi", teams, index=0)
    away = c2.selectbox("Deplasman", teams, index=min(1, len(teams) - 1))
    if home == away:
        st.warning("Iki farkli takim sec.")
        return

    prof = now[now["league"] == lg]
    pair = ST.pair_rows(prof, home, away)
    th = pd.DataFrame(th_all.get(lg, [])).set_index("index") if th_all.get(lg) else None

    st.subheader("Profiller (son maclar)")
    show = prof.set_index("team").loc[[home, away], [f"pre_{c}" for c in ST.PROFILE_COLS]]
    show.columns = [MATCH_LABELS[c[4:]] for c in show.columns]
    show["Mac sayisi"] = prof.set_index("team").loc[[home, away], "n_prev"].to_numpy()
    st.dataframe(show.T.round(3), width="stretch")

    if th is None or th.isna().all().all():
        st.info("Bu lig icin esik yok (onceki sezon verisi gerekiyor).")
        return
    flags = ST.apply_thresholds(pair, th)

    st.subheader("Kurallar")
    rows = []
    for rule in R.RULES:
        fired = rule.when(flags).to_numpy()
        res = results[results["rule"] == rule.key].iloc[0] if not results.empty and (
            results["rule"] == rule.key).any() else None
        for side, f in zip((home, away), fired, strict=True):
            if not f:
                continue
            if res is not None and bool(res["kept"]):
                status = (f"Gecerli: kesifte {res['diff_disc']:+.2f}, dogrulamada "
                          f"{res['diff_conf']:+.2f} {TARGET_TR[rule.target]}/mac")
            elif res is not None and bool(res.get("reversed", False)):
                status = "Ters yonde tekrar etti: kural yanlis yazilmis, yalnizca bilgi"
            elif res is not None:
                status = "Geriye donuk testte tutmadi: yalnizca bilgi"
            else:
                status = "Test sonucu yok"
            n_min = int(prof.set_index("team").loc[[home, away], "n_prev"].min())
            weak = ["vekil metrikler (olay verisi yok)"]
            if n_min < ST.WINDOW:
                weak.append(f"profil yalnizca {n_min} mactan")
            rows.append({"Kural": rule.title, "Kimin lehine / kimde": side,
                         "Metrikler": ", ".join(rule.metrics), "Beklenen etki": (
                             f"{'+' if rule.sign > 0 else '-'} {TARGET_TR[rule.target]}"),
                         "Durum": status, "Zayiflatan faktorler": "; ".join(weak)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("Bu eslesmede hicbir kural ateslenmiyor.")

    if not results.empty:
        with st.expander("Tum kurallarin geriye donuk test sonucu"):
            t = results[["title", "target", "n_fired_disc", "diff_disc", "q", "n_fired_conf",
                         "diff_conf", "p_conf_dir", "kept"]
                        + (["reversed"] if "reversed" in results else [])].copy()
            t["target"] = t["target"].map(TARGET_TR)
            st.dataframe(t.round(3), hide_index=True, width="stretch")
            st.caption("fark = kural ateslendiginde ortalama artik - ateslenmediginde ortalama "
                       "artik (gerceklesen - model). q = Benjamini-Hochberg duzeltilmis p.")

    clusters = load_json("style_clusters.json")
    if clusters:
        with st.expander("Stil gruplari (K-Means)"):
            centers = np.asarray(clusters["centers"])
            z = prof.set_index("team")[[f"pre_{c}" for c in ST.CLUSTER_FEATURES]]
            z = (z - z.mean()) / z.std(ddof=0)
            lab = ((z.to_numpy()[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
            st.write({t: clusters["labels"][g] for t, g in zip(z.index, lab, strict=True)
                      if t in (home, away)})
            st.caption(f"k={clusters['k']}; siluet skoru dusukse gruplar ic ice demektir.")


tab_match, tab_now, tab_sb = st.tabs(["Eslesme", "Guncel stil profili", "StatsBomb 2015/16"])

with tab_match:
    matchup()

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
        lg = lg_names[c1.selectbox("Lig", list(lg_names), key="profile_league")]
        seasons = sorted(prof["season"].unique(), reverse=True)
        season = c2.selectbox("Sezon", seasons, format_func=season_label, key="profile_season")
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
        lg = lg_names[st.selectbox("Lig", list(lg_names), key="sb_league")]
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
