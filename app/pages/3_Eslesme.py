"""Taktiksel Eslesme: iki takimin stil profili ve ateslenen kurallar."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from footintel.analysis import rules as R
from footintel.config import CLEAN_DIR, LEAGUES
from footintel.features import style as ST

st.set_page_config(page_title="Taktiksel Eslesme", layout="wide")
st.title("Taktiksel Eslesme")
st.caption(
    "Her takimin profili son 10 macindan (football-data: sut, isabetli sut, korner, faul, "
    "kart). 'Yuksek' ve 'dusuk', ayni ligin onceki sezonlarinin ust ve alt %25'i. Kurallar "
    "onceden yazildi ve modelin artigina karsi test edildi: 2022-24'te kesif, 2024-26'da "
    "dogrulama. Yalnizca iki donemde de tutan kurallar 'gecerli' sayilir."
)

LABELS = {"shot_share": "Sut payi", "sot_share": "Isabetli sut payi",
          "corner_share": "Korner payi", "fouls": "Faul/mac", "fouls_drawn": "Cekilen faul/mac",
          "yellows": "Sari kart/mac", "sot_ratio": "Isabet orani",
          "sot_ratio_against": "Rakibin isabet orani"}
TARGET_TR = {"res_goals": "gol", "res_goals_against": "yenilen gol", "res_corners": "korner",
             "res_fouls": "faul", "res_yellows": "sari kart"}


def load_json(name: str):
    p = CLEAN_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


path = CLEAN_DIR / "style_now.parquet"
if not path.exists():
    st.info("Profil verisi yok. CI'daki build_matchup adimi uretir.")
    st.stop()
now = pd.read_parquet(path)
results = pd.DataFrame(load_json("matchup_rules.json") or [])
th_all = load_json("style_thresholds.json") or {}

leagues = {LEAGUES[k].name: k for k in sorted(now["league"].unique()) if k in LEAGUES}
lg = leagues[st.selectbox("Lig", list(leagues))]
teams = sorted(now.loc[now["league"] == lg, "team"])
c1, c2 = st.columns(2)
home = c1.selectbox("Ev sahibi", teams, index=0)
away = c2.selectbox("Deplasman", teams, index=min(1, len(teams) - 1))
if home == away:
    st.warning("Iki farkli takim sec.")
    st.stop()

prof = now[now["league"] == lg]
pair = ST.pair_rows(prof, home, away)
th = pd.DataFrame(th_all.get(lg, [])).set_index("index") if th_all.get(lg) else None

st.subheader("Profiller (son maclar)")
show = prof.set_index("team").loc[[home, away], [f"pre_{c}" for c in ST.PROFILE_COLS]]
show.columns = [LABELS[c[4:]] for c in show.columns]
show["Mac sayisi"] = prof.set_index("team").loc[[home, away], "n_prev"].to_numpy()
st.dataframe(show.T.round(3), width="stretch")

if th is None or th.isna().all().all():
    st.info("Bu lig icin esik yok (onceki sezon verisi gerekiyor).")
    st.stop()
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
                     "diff_conf", "p_conf_dir", "kept"]].copy()
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
