"""Faz 7: taktiksel eslesme kurallari ve geriye donuk testleri.

Kurallar ONCEDEN yazilir (veriye bakmadan), her biri futbol bilgisine dayanan bir
hipotezdir: "takimin bir gucu + rakibin ilgili zayifligi -> beklenenden fazla X".

Test neye karsi? Modelin artigina (residual = gerceklesen - modelin beklentisi).
Neden? "Guclu takim cok gol atar" bir kural degil; Dixon-Coles bunu zaten biliyor.
Bir kuralin degeri, modelin BILMEDIGI bir seyi soylemesidir. Fouls ve kartlar icin
modelimiz yok; beklenti iki takimin mac oncesi ortalamalarindan gelir.

Istatistik: kuralin ateslendigi satirlarin ortalama artigi EKSI ateslenmedigi satirlarin
ortalama artigi (fark). Baseline'in genel yanliligi (or. kart beklentisi lig genelinde
biraz dusukse) her iki grubu ayni etkiler ve farkta duser. Guven araligi maclar
uzerinden bootstrap ile. Coklu test duzeltmesi: Benjamini-Hochberg.

Kapi: kural yalnizca (1) kesif doneminde BH duzeltmesinden sonra anlamli VE (2) dogrulama
doneminde ayni yonde, tek yonlu p < 0.05 ise tutulur.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class Rule:
    key: str
    title: str
    why: str
    target: str            # artik kolonu: res_goals, res_corners, res_fouls, res_yellows
    sign: int              # beklenen yon: +1 beklenenden fazla, -1 az
    when: Callable[[pd.DataFrame], pd.Series]
    metrics: tuple[str, ...]


def _b(d: pd.DataFrame, col: str) -> pd.Series:
    return d[col].astype("boolean").fillna(False).astype(bool)


RULES: list[Rule] = [
    Rule("corner_pressure", "Korner baskisi",
         "Cok korner kazanan takim, korner veren (geride kalan) bir rakiple karsilasinca "
         "modelin beklediginden fazla korner kazanir.",
         "res_corners", +1,
         lambda d: _b(d, "corner_share_high") & _b(d, "opp_corner_share_low"),
         ("corner_share", "opp_corner_share")),
    Rule("shot_volume_vs_leaky", "Sut hacmi, sizdiran savunmaya karsi",
         "Cok sut ceken takim, rakibe isabetli sut orani yuksek olan bir savunmaya karsi "
         "beklenenden fazla gol atar.",
         "res_goals", +1,
         lambda d: _b(d, "shot_share_high") & _b(d, "opp_sot_ratio_against_high"),
         ("shot_share", "opp_sot_ratio_against")),
    Rule("clinical_vs_leaky", "Isabetli bitiricilik, sizdiran savunmaya karsi",
         "Sutlarinin buyuk kismi isabetli olan takim, isabetli sut izni veren savunmaya "
         "karsi beklenenden fazla gol atar.",
         "res_goals", +1,
         lambda d: _b(d, "sot_ratio_high") & _b(d, "opp_sot_ratio_against_high"),
         ("sot_ratio", "opp_sot_ratio_against")),
    Rule("tight_defence_vs_wasteful", "Siki savunma, israfci hucuma karsi",
         "Isabetli sut izni vermeyen savunma, sutlari az isabet bulan bir hucumu "
         "beklenenden fazla susturur (rakip beklenenden az gol atar).",
         "res_goals_against", -1,
         lambda d: _b(d, "sot_ratio_against_low") & _b(d, "opp_sot_ratio_low"),
         ("sot_ratio_against", "opp_sot_ratio")),
    Rule("fouler_vs_drawer", "Faulcu takim, faul cekene karsi",
         "Cok faul yapan takim, cok faul ceken (topu tutan, dripling yapan) rakibe karsi "
         "iki takimin ortalamasindan fazla faul yapar.",
         "res_fouls", +1,
         lambda d: _b(d, "fouls_high") & _b(d, "opp_fouls_drawn_high"),
         ("fouls", "opp_fouls_drawn")),
    Rule("card_magnet", "Kart eslesmesi",
         "Cok kart goren takim, cok faul ceken rakibe karsi beklenenden fazla kart gorur.",
         "res_yellows", +1,
         lambda d: _b(d, "yellows_high") & _b(d, "opp_fouls_drawn_high"),
         ("yellows", "opp_fouls_drawn")),
    Rule("dominant_vs_dominant", "Iki baskin takim",
         "Iki takim da sut payinda ustteyse oyun ortada kilitlenir; takim beklenenden "
         "az gol atar.",
         "res_goals", -1,
         lambda d: _b(d, "shot_share_high") & _b(d, "opp_shot_share_high"),
         ("shot_share", "opp_shot_share")),
    Rule("low_block_corners", "Alcak blok korner verir",
         "Sut payi dusuk (geride bekleyen) rakibe karsi takim, korner modelinin "
         "beklediginden fazla korner kazanir.",
         "res_corners", +1,
         lambda d: _b(d, "opp_shot_share_low") & _b(d, "shot_share_high"),
         ("shot_share", "opp_shot_share")),
]


def residuals(rows: pd.DataFrame) -> pd.DataFrame:
    """Artiklar. Beklentiler: gol ve korner modelden; faul ve kart mac oncesi profilden.

    Faul beklentisi = (bizim mac oncesi faul ortalamamiz + rakibin cektigi faul ort.) / 2.
    Kart beklentisi = bizim mac oncesi kart ortalamamiz.
    """
    d = rows.copy()
    d["res_goals"] = d["goals"] - d["exp_goals"]
    d["res_goals_against"] = d["goals_against"] - d["exp_goals_against"]
    d["res_corners"] = d["corners"] - d["exp_corners"]
    d["res_fouls"] = d["fouls"] - (d["pre_fouls"] + d["opp_pre_fouls_drawn"]) / 2
    d["res_yellows"] = d["yellows"] - d["pre_yellows"]
    return d


def _diff(res: np.ndarray, fired: np.ndarray) -> float:
    return float(res[fired].mean() - res[~fired].mean())


def test_rule(d: pd.DataFrame, rule: Rule, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ateslenen - ateslenmeyen ortalama artik farki, mac bazinda bootstrap ile."""
    x = d.dropna(subset=[rule.target]).copy()
    flags = [f"{m}_high" for m in rule.metrics]
    x = x[x[flags].notna().all(axis=1)]              # esigi olmayan (ilk sezon) satirlar disarida
    fired = rule.when(x).to_numpy()
    out = {"rule": rule.key, "n_fired": int(fired.sum()), "n": int(len(x))}
    if fired.sum() < 20 or (~fired).sum() < 20:
        return {**out, "diff": np.nan, "lo": np.nan, "hi": np.nan, "p_two": np.nan,
                "p_one": np.nan}
    res = x[rule.target].to_numpy(float)
    est = _diff(res, fired)
    # Mac bazinda yeniden ornekleme: ayni macin iki satiri birlikte secilir. Hizli olmasi
    # icin her mac icin (ateslenen artik toplami, sayisi, ateslenmeyen toplam, sayi)
    # hesaplayip, bootstrap'i maclarin agirliklari (kac kez secildigi) uzerinden yapiyoruz.
    codes, uniq = pd.factorize(x["match_id"])
    m = len(uniq)
    sf = np.bincount(codes, weights=res * fired, minlength=m)
    cf = np.bincount(codes, weights=fired.astype(float), minlength=m)
    sn = np.bincount(codes, weights=res * ~fired, minlength=m)
    cn = np.bincount(codes, weights=(~fired).astype(float), minlength=m)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, m, m), minlength=m)
        boots[b] = (w @ sf) / max(w @ cf, 1e-9) - (w @ sn) / max(w @ cn, 1e-9)
    se = boots.std(ddof=1)
    z = est / se if se > 0 else 0.0
    return {**out, "diff": est, "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "p_two": float(2 * norm.sf(abs(z))),
            "p_one": float(norm.sf(z * rule.sign))}


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH ile duzeltilmis p-degerleri (q). Cok kural test edince bazilari sansla
    'anlamli' cikar; BH, anlamli bulunanlar icindeki yanlis kesif oranini sinirlar."""
    p = p.dropna()
    n = len(p)
    order = p.sort_values()
    q = order * n / np.arange(1, n + 1)
    q = q[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def evaluate_rules(d: pd.DataFrame, discovery: list[str], confirmation: list[str],
                   rules: list[Rule] = RULES, alpha: float = 0.05) -> pd.DataFrame:
    disc = pd.DataFrame([test_rule(d[d["season"].isin(discovery)], r) for r in rules])
    conf = pd.DataFrame([test_rule(d[d["season"].isin(confirmation)], r) for r in rules])
    disc["q"] = benjamini_hochberg(disc["p_two"])
    t = disc.merge(conf, on="rule", suffixes=("_disc", "_conf"))
    meta = pd.DataFrame([{"rule": r.key, "title": r.title, "target": r.target,
                          "expected_sign": r.sign} for r in rules])
    t = meta.merge(t, on="rule")
    sign_disc = np.sign(t["diff_disc"])
    same_sign = np.sign(t["diff_conf"]) == sign_disc
    p_conf_dir = np.where(sign_disc == t["expected_sign"], t["p_one_conf"],
                          1 - t["p_one_conf"])
    t["p_conf_dir"] = p_conf_dir
    t["kept"] = (t["q"] < alpha) & same_sign & (t["p_conf_dir"] < alpha)
    t["as_expected"] = sign_disc == t["expected_sign"]
    return t
