"""Faz 9 asama 4: Mac Merkezi icin gosterim yardimcilari (Streamlit'ten bagimsiz, test edilir).

Sayfa yalnizca bu fonksiyonlarin urettigi metinleri cizer. Boylece "eksik veri N/A olur",
"sonucu olmayan mac isaretlenmez" gibi kurallar UI kodunda degil, burada test edilir.
"""

from __future__ import annotations

import colorsys
import hashlib
import html

import numpy as np
import pandas as pd

NA = "N/A"
RESULT_TR = {"H": "1", "D": "X", "A": "2"}
# isaret -> (sembol, css sinifi)
MARKS = {True: ("✓", "ok"), False: ("✗", "bad"), None: ("–", "na")}


def _isna(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def _pct(p) -> str:
    return NA if _isna(p) else f"%{100 * float(p):.0f}"


def _num(v, digits: int = 0) -> str:
    return NA if _isna(v) else f"{float(v):.{digits}f}"


def _mark(v):
    return None if _isna(v) else bool(v)


def badge(team: str) -> tuple[str, str]:
    """Takim rozeti: 3 harf kisaltma ve isimden turetilen sabit bir renk.

    Neden gercek logo degil? Kulup logolari tescilli; ucretsiz ve lisansli bir kaynak yok.
    Renk isimden hash ile uretilir: her calismada ayni, ama kulubun gercek rengi DEGIL.
    """
    words = [w for w in team.replace(".", " ").split() if w]
    abbr = (words[0][:3] if len(words) == 1 else "".join(w[0] for w in words[:3])).upper()
    h = int(hashlib.md5(team.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hls_to_rgb(h, 0.38, 0.55)
    return abbr, f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def card_rows(m: pd.Series) -> list[dict]:
    """Bir macin tahmin/gercek/isaret satirlari. m: tahmin + (varsa) degerlendirme kolonlari.

    Sonucu olmayan macta gercek "—" ve isaret None (gri) olur.
    """
    played = not _isna(m.get("home_goals"))
    g = m.get
    p_res = {"H": g("p_home"), "D": g("p_draw"), "A": g("p_away")}.get(g("pred_result"))

    def yn(p, yes, no):
        return NA if _isna(p) else f"{yes if p > 0.5 else no} ({_pct(p)})"

    def actual_yn(v, yes, no):
        return "—" if not played else (NA if _isna(v) else (yes if v else no))

    def actual_num(v):
        return "—" if not played else _num(v)

    rows = [
        {"label": "Mac sonucu (1X2)",
         "pred": f"{RESULT_TR.get(g('pred_result'), NA)} ({_pct(p_res)})",
         "actual": "—" if not played else RESULT_TR.get(g("result_actual"), NA),
         "mark": _mark(g("result_ok")), "mark_pm1": "skip"},
        {"label": "Skor",
         "pred": f"{g('pred_score') or NA} ({_pct(g('p_pred_score'))})",
         "actual": "—" if not played else (g("score_actual") or NA),
         "mark": _mark(g("score_ok")), "mark_pm1": "skip"},
        {"label": "Toplam gol",
         "pred": f"{_num(g('exp_total_goals'), 1)} → {_num(g('goals_total_pred'))}"
         if not _isna(g("exp_total_goals")) else NA,
         "actual": actual_num(g("goals_total_actual")),
         "mark": _mark(g("goals_total_ok")), "mark_pm1": _mark(g("goals_total_ok_pm1"))},
        {"label": "2.5 Ust/Alt", "pred": yn(g("p_over25"), "Ust", "Alt"),
         "actual": actual_yn(g("ou25_actual"), "Ust", "Alt"),
         "mark": _mark(g("ou25_ok")), "mark_pm1": "skip"},
        {"label": "Karsilikli gol", "pred": yn(g("p_btts"), "Var", "Yok"),
         "actual": actual_yn(g("btts_actual"), "Var", "Yok"),
         "mark": _mark(g("btts_ok")), "mark_pm1": "skip"},
    ]
    for key, label, exp in (("corners_home", "Korner (ev)", "exp_home_corners"),
                            ("corners_away", "Korner (dep)", "exp_away_corners"),
                            ("corners_total", "Korner (toplam)", "exp_corners"),
                            ("yellow_home", "Sari kart (ev)", "exp_home_yellow"),
                            ("yellow_away", "Sari kart (dep)", "exp_away_yellow"),
                            ("yellow_total", "Sari kart (toplam)", None)):
        e = g(exp) if exp else (g("exp_home_yellow") + g("exp_away_yellow")
                                if not (_isna(g("exp_home_yellow"))
                                        or _isna(g("exp_away_yellow"))) else np.nan)
        rows.append({"label": label,
                     "pred": NA if _isna(e) else f"{e:.1f} → {np.floor(e + 0.5):.0f}",
                     "actual": actual_num(g(f"{key}_actual")),
                     "mark": _mark(g(f"{key}_ok")), "mark_pm1": _mark(g(f"{key}_ok_pm1"))})
    rows.append({"label": "Kirmizi kart", "pred": yn(g("p_red_any"), "Var", "Yok"),
                 "actual": actual_yn(g("red_actual"), "Var", "Yok"),
                 "mark": _mark(g("red_ok")), "mark_pm1": "skip"})
    rows.append({"label": "Ilk golu atan", "pred": NA, "actual": NA, "mark": None,
                 "mark_pm1": "skip"})
    return rows


def mark_html(v) -> str:
    if v == "skip":
        return ""
    sym, cls = MARKS[v]
    return f'<span class="fi-{cls}">{sym}</span>'


def card_html(m: pd.Series) -> str:
    """Mac karti (HTML). Tum metinler kacislanir; takim adi HTML olarak yorumlanmaz."""
    e = html.escape
    played = not _isna(m.get("home_goals"))
    score = (f'{int(m["home_goals"])} - {int(m["away_goals"])}' if played
             else '<span class="fi-na">Sonuc bekleniyor</span>')
    side = []
    for team in (m["home_team"], m["away_team"]):
        abbr, color = badge(team)
        side.append(f'<span class="fi-badge" style="background:{color}">{e(abbr)}</span>'
                    f'<span class="fi-team">{e(team)}</span>')
    body = "".join(
        f"<tr><td>{e(r['label'])}</td><td>{e(r['pred'])}</td><td>{e(r['actual'])}</td>"
        f"<td>{mark_html(r['mark'])}</td><td>{mark_html(r['mark_pm1'])}</td></tr>"
        for r in card_rows(m))
    date = pd.Timestamp(m["match_date"]).strftime("%d.%m.%Y")
    return (f'<div class="fi-card"><div class="fi-head"><div class="fi-side">{side[0]}</div>'
            f'<div class="fi-score">{score}</div><div class="fi-side fi-right">{side[1]}'
            f'</div></div><div class="fi-date">{date}</div>'
            '<table class="fi-table"><thead><tr><th>Tahmin</th><th>Model</th><th>Gercek</th>'
            '<th>Tam</th><th>±1</th></tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


CSS = """
<style>
.fi-card{border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:12px 14px;
  margin-bottom:14px}
.fi-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.fi-side{display:flex;align-items:center;gap:8px;flex:1;min-width:0}
.fi-right{justify-content:flex-end;flex-direction:row-reverse}
.fi-badge{color:#fff;font-weight:700;font-size:.75rem;border-radius:50%;width:38px;
  height:38px;display:inline-flex;align-items:center;justify-content:center;flex:none}
.fi-team{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fi-score{font-size:1.5rem;font-weight:700;white-space:nowrap;text-align:center}
.fi-date{opacity:.65;font-size:.8rem;text-align:center;margin:2px 0 8px}
.fi-table{width:100%;border-collapse:collapse;font-size:.85rem}
.fi-table th,.fi-table td{padding:3px 6px;border-top:1px solid rgba(128,128,128,.2);
  text-align:left}
.fi-ok{color:#16a34a;font-weight:700}.fi-bad{color:#dc2626;font-weight:700}
.fi-na{color:#9ca3af}
</style>
"""
