"""Faz 10: sayfalarin ortak gorunumu (renk, tipografi, KPI, lig secimi).

Neden ortak modul? Her sayfa kendi rengini ve basligini secince arayuz daginik gorunuyordu.
Tek yerde tanimlamak tutarliligi garanti eder:
- Model her yerde mavi, kiyas (piyasa ya da lig ortalamasi) her yerde turuncu.
- Dogru yesil, yanlis kirmizi, bilinmeyen gri. Baska anlamli renk yok.
- Lig ve sezon secimi sayfalar arasinda hatirlanir (st.session_state), kullanici her
  sayfada ayni filtreyi yeniden secmek zorunda kalmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from footintel.config import LEAGUES  # noqa: E402

MODEL, BASE = "#3987e5", "#d95926"
GOOD, BAD, MUTED = "#22c55e", "#ef4444", "#9ca3af"
DISCLAIMER = "Portfoy projesidir, bahis tavsiyesi degildir."

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
h1 {font-size: 1.9rem !important; font-weight: 700; margin-bottom: 0 !important;}
h2, h3 {font-size: 1.15rem !important; font-weight: 600; margin-top: 1.4rem !important;}
[data-testid="stMetric"] {background: #1a1b1e; border: 1px solid #2a2b30;
  border-radius: 10px; padding: 12px 14px;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #a1a1aa;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
[data-testid="stCaptionContainer"] {color: #a1a1aa;}
.fi-lead {color: #c9c9c4; font-size: 0.95rem; margin: 0.2rem 0 1rem 0;}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem;}
  h1 {font-size: 1.5rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.2rem;}
  /* KPI kartlari telefonda alt alta dort uzun kutu yerine 2x2 izgara */
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    flex-direction: row !important; flex-wrap: wrap !important; gap: 0.5rem !important;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    width: calc(50% - 0.25rem) !important; min-width: calc(50% - 0.25rem) !important;
    flex: 1 1 calc(50% - 0.25rem) !important;}
}
</style>
"""


def page(title: str, lead: str) -> None:
    """Sayfa ayari + baslik + tek cumlelik aciklama + ortak stil."""
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="fi-lead">{lead}</p>', unsafe_allow_html=True)


def league_name(code: str) -> str:
    return LEAGUES[code].name if code in LEAGUES else code


def season_text(s: str) -> str:
    return f"20{s[:2]}/{s[2:]}"


def pick(label: str, options: list, key: str, container=st, all_label: str | None = None,
         format_func=str):
    """Sayfalar arasinda hatirlanan secim kutusu. all_label verilirse basa 'Tum...' eklenir.

    Not: deger widget anahtarinda degil, ayri bir anahtarda saklanir; Streamlit sayfa
    degisince widget anahtarlarini siler ama normal session_state degerlerini korur.
    """
    opts = ([None] if all_label else []) + list(options)
    if not opts:
        return None
    saved = st.session_state.get(f"keep_{key}")
    index = opts.index(saved) if saved in opts else 0
    fmt = (lambda v: all_label if v is None else format_func(v)) if all_label else format_func
    value = container.selectbox(label, opts, index=index, format_func=fmt, key=f"w_{key}")
    st.session_state[f"keep_{key}"] = value
    return value


def pct(x: float | None, digits: int = 1) -> str:
    return "N/A" if x is None or pd.isna(x) else f"%{100 * x:.{digits}f}"


def delta_points(model: float | None, base: float | None, digits: int = 1) -> str | None:
    """Kiyasa gore yuzde puan farki; kiyas yoksa None (Streamlit kutu gostermez)."""
    if model is None or base is None or pd.isna(model) or pd.isna(base):
        return None
    d = 100 * (model - base)
    return "±0 puan" if abs(d) < 0.5 * 10 ** -digits else f"{d:+.{digits}f} puan"
