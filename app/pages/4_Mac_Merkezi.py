"""Mac Merkezi: gecmis maclarda mac oncesi tahmin ve gerceklesen sonuc yan yana (Faz 9)."""

from __future__ import annotations

# Streamlit Cloud paketi kurmadan calistirir; src/ klasorunu yola ekliyoruz.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st

from footintel import store
from footintel.analysis import history_view as V
from footintel.config import LEAGUES

st.set_page_config(page_title="Mac Merkezi", layout="wide")
st.title("Mac Merkezi")
st.caption("Her tahmin, mactan once yalnizca o haftaya kadarki verilerle uretildi ve "
           "degistirilmeden saklandi. Gercek sonuc ayri bir tablodan eslestirilir.")


@st.cache_data(ttl=3600)
def load():
    return (store.read_table("historical_predictions"),
            store.read_table("prediction_evaluation"),
            store.read_table("evaluation_summary"), store.read_json("halftime_eval"))


preds, ev, summary, ht_eval = load()
if preds is None or preds.empty:
    st.info("Gecmis tahmin tablosu henuz yok. CI'daki 'build_history' adimi uretir.")
    st.stop()

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
leagues = {LEAGUES[k].name if k in LEAGUES else k: k for k in sorted(preds["league"].unique())}
league = leagues[c1.selectbox("Lig", list(leagues))]
p = preds[preds["league"] == league]
season = c2.selectbox("Sezon", sorted(p["season"].unique(), reverse=True),
                      format_func=lambda s: f"20{s[:2]}/{s[2:]}")
p = p[p["season"] == season]
weeks = sorted(p["matchweek"].dropna().astype(int).unique())
week = c3.selectbox("Hafta", weeks, index=len(weeks) - 1)
versions = sorted(p["model_version"].unique(), reverse=True)
version = c4.selectbox("Model surumu", versions)
p = p[(p["matchweek"] == week) & (p["model_version"] == version)]

if ev is not None and not ev.empty:
    keep = ["match_id", "model_version", *[c for c in ev.columns if c not in p.columns]]
    p = p.merge(ev[keep], on=["match_id", "model_version"], how="left")
p = p.sort_values(["match_date", "home_team"])

# Haftanin ozeti: yuzde her zaman n ile; baseline yanina
if summary is not None and not summary.empty:
    s = summary[(summary["scope"] == "week") & (summary["league"] == league)
                & (summary["season"] == season) & (summary["matchweek"] == week)
                & (summary["model_version"] == version)].set_index("market")
    names = {"result": "1X2", "ou25": "2.5 Ust/Alt", "btts": "KG", "goals_total": "Toplam gol",
             "corners_total": "Toplam korner", "yellow_total": "Toplam sari"}
    cols = st.columns(len(names))
    for col, (m, label) in zip(cols, names.items(), strict=True):
        if m not in s.index or s.loc[m, "n"] == 0:
            col.metric(label, V.NA, help="Bu hafta degerlendirilebilen mac yok")
            continue
        r = s.loc[m]
        pm1 = f" · ±1: %{100 * r['pct_pm1']:.0f}" if pd.notna(r.get("pct_pm1")) else ""
        base = r.get("base_pct")
        delta = (f"{100 * (r['pct'] - base):+.0f} puan kiyasa gore" if pd.notna(base)
                 else None)
        col.metric(label, f"%{100 * r['pct']:.0f}", delta,
                   help=f"{int(r['hits'])}/{int(r['n'])} mac{pm1}")
    st.caption("Kiyas: 1X2 ve 2.5 icin kapanis oranlarindaki piyasa favorisi; digerleri icin "
               "ligin onceki sezon ortalamasi. Tek haftalik oranlar az macla hesaplanir, "
               "gurultuludur; sezon resmi icin Performans Paneli.")

st.markdown(V.CSS, unsafe_allow_html=True)
grid = st.columns(2)
for i, (_, m) in enumerate(p.iterrows()):
    grid[i % 2].markdown(V.card_html(m), unsafe_allow_html=True)

st.caption("✓ dogru · ✗ yanlis · – degerlendirilemedi (veri yok ya da mac oynanmadi). "
           "Sayilarda model beklentisi yarim yukari yuvarlanir; 'Tam' birebir, '±1' bir "
           "fark toleransli isabettir. Kirmizi kart olasiligi lig oranina yakin oldugu icin "
           "neredeyse hep 'Yok' der; anlamli olcusu Brier skorudur. Ilk golu atan takim icin "
           "veri kaynagimizda dakika bilgisi yok (N/A); yerine ilk yari sonucu tahmini "
           "gosteriliyor. Rozetler logo degil, isimden uretilen kisaltmadir.")
if ht_eval and not ht_eval.get("gate_passed"):
    st.caption("Ilk yari sonucu: dusuk guven. Test sezonlarinda model, ligin ilk yari "
               "frekansindan anlamli olarak daha iyi cikmadi (Brier).")
