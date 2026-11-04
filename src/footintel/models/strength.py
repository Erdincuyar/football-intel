"""Takim gucu modeli: Dixon-Coles (goller) ve Negatif Binom / Poisson (kornerler).

Model (Maher 1982, Dixon & Coles 1997):

    log(lambda_ev)  = c + h + hucum[ev]  + savunma[dep]
    log(mu_dep)     = c     + hucum[dep] + savunma[ev]

- c: ligin ortalama seviyesi, h: ev sahibi avantaji (lig bazinda, cunku her lig ayri
  fit edilir). hucum yuksek = cok atar; savunma yuksek = cok yer.
- Goller: Poisson. Dixon-Coles duzeltmesi (rho) yalnizca 0-0, 1-0, 0-1, 1-1 skorlarinin
  olasiligini carpar (tau). EDA'da bu skorlarda bagimsizliktan sapma gorduk.
- Kornerler: Negatif Binom (varyans = m + m^2 / r). r -> sonsuz iken Poisson'a doner.

Uc pratik ekleme:
1. Zaman agirligi: w = exp(-xi * gun). Eski maclar daha az sayilir; xi ile ayarlanir.
2. Ceza (ridge, Bayesci onsel gibi dusunulebilir): hucum/savunma degerleri 0'a (ortalama
   takim) dogru cekilir. Az maci olan takimin parametresi asiri sicramaz.
3. Yukselen takimlar: Onlari 0'a degil, ayni fit icindeki kume dusen takimlarin ortalama
   parametresine dogru cekeriz (Elo'daki fikrin aynisi). Sezon basinda yukselen takimin
   hic maci yoken tahmin bu onselden gelir; maclar biriktikce veri onseli gecer.

Optimizasyon: Negatif log-olabilirligi L-BFGS-B ile minimize ediyoruz. Gradyani elle
(analitik) yaziyoruz; sayisal gradyana gore ~40 kat hizli. Bu, walk-forward backtest'te
yuzlerce kez yeniden fit etmeyi mumkun kiliyor. Dogrulugu testte sayisal gradyanla
karsilastirilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

RHO_BOUNDS = (-0.25, 0.25)


@dataclass(frozen=True)
class ModelConfig:
    family: str = "poisson"        # "poisson" | "negbin"
    dixon_coles: bool = True       # rho duzeltmesi (yalnizca goller icin anlamli)
    xi: float = 0.002              # zaman azalmasi, 1/gun. Yari omur = ln2 / xi
    penalty: float = 8.0           # hucum/savunma icin ridge cezasi
    max_days: int = 730            # bundan eski maclar kullanilmaz (hiz icin)
    shot_weight: float = 0.0       # 0 = yalnizca gol; >0 = isabetli sutla harmanlanmis hedef


@dataclass
class FittedModel:
    config: ModelConfig
    teams: list[str]
    c: float
    home: float
    attack: np.ndarray
    defence: np.ndarray
    rho: float = 0.0
    log_r: float = np.inf          # negbin dagilim parametresi (log)
    success: bool = True
    newcomer: tuple[float, float] = (0.0, 0.0)  # bilinmeyen takim icin (hucum, savunma)
    index: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.teams)}

    def _params(self, team: str) -> tuple[float, float]:
        i = self.index.get(team)
        return (self.attack[i], self.defence[i]) if i is not None else self.newcomer

    def rates(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self._params(home)
        aa, da = self._params(away)
        return (float(np.exp(self.c + self.home + ah + da)),
                float(np.exp(self.c + aa + dh)))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"team": self.teams, "attack": self.attack,
                             "defence": self.defence}).sort_values("attack", ascending=False)


def time_weights(dates: pd.Series, ref: pd.Timestamp, xi: float) -> np.ndarray:
    days = (ref - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * days)


def _tau_terms(x, y, lam, mu, rho):
    """log tau ve turevleri (lambda, mu, rho'ya gore). Yalnizca dusuk skorlar etkilenir."""
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)
    d_lam[m00] = -mu[m00] * rho
    d_mu[m00] = -lam[m00] * rho
    d_rho[m00] = -lam[m00] * mu[m00]
    d_lam[m01] = rho
    d_rho[m01] = lam[m01]
    d_mu[m10] = rho
    d_rho[m10] = mu[m10]
    d_rho[m11] = -1.0
    return np.log(tau), d_lam / tau, d_mu / tau, d_rho / tau


def _count_ll(y, m, log_r, family):
    """Log-olabilirlik (sabitler dahil) ve d/d(log m), d/d(log r)."""
    if family == "poisson":
        return y * np.log(m) - m - gammaln(y + 1), y - m, np.zeros_like(m)
    r = np.exp(log_r)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
          + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
    d_eta = r * (y - m) / (r + m)
    d_logr = r * (digamma(y + r) - digamma(r) + np.log(r / (r + m)) + 1 - (r + y) / (r + m))
    return ll, d_eta, d_logr


def _shot_blend(d: pd.DataFrame, x: np.ndarray, y: np.ndarray, beta: float):
    """Hedef = (1 - beta) * gol + beta * c * isabetli_sut.

    Neden? Gol nadir ve sansa bagli bir olaydir; isabetli sut cok daha sik gorulur ve
    takimin pozisyon uretme gucunu daha az gurultuyle olcer (bir tur "xG vekili").
    c = egitim penceresindeki gol / isabetli sut orani (lig ortalamasi ~0.3). Boylece
    hedef gol olceginde kalir. Sut verisi eksik maclarda yalnizca gol kullanilir.
    Hedef artik tam sayi olmadigi icin olabilirlik "quasi-Poisson" olur: gradyan ve
    tahmin ayni kalir, yalnizca sabit terim anlamini yitirir.
    """
    sh = d["home_shots_on_target"].to_numpy(dtype=float)
    sa = d["away_shots_on_target"].to_numpy(dtype=float)
    ok = ~(np.isnan(sh) | np.isnan(sa))
    if not ok.any():
        return x, y
    c = (x[ok].sum() + y[ok].sum()) / max(sh[ok].sum() + sa[ok].sum(), 1.0)
    x = np.where(ok, (1 - beta) * x + beta * c * np.nan_to_num(sh), x)
    y = np.where(ok, (1 - beta) * y + beta * c * np.nan_to_num(sa), y)
    return x, y


def fit(
    matches: pd.DataFrame,
    ref_date: pd.Timestamp,
    config: ModelConfig = ModelConfig(),  # noqa: B008  (frozen dataclass, guvenli)
    *,
    home_col: str = "home_goals",
    away_col: str = "away_goals",
    promoted: set[str] | None = None,
    relegated: set[str] | None = None,
    warm_start: FittedModel | None = None,
    _return_problem: bool = False,
) -> FittedModel:
    """Tek bir lig icin, ref_date'ten ONCEKI maclarla fit eder.

    promoted / relegated: bu sezon yukselen ve gecen sezon kume dusen takimlar. Yukselenlerin
    parametreleri kume dusenlerin ortalamasina dogru cezalandirilir.
    """
    ref_date = pd.Timestamp(ref_date)
    d = matches[pd.to_datetime(matches["match_date"]) < ref_date]
    d = d[(ref_date - pd.to_datetime(d["match_date"])).dt.days <= config.max_days]
    d = d.dropna(subset=[home_col, away_col])
    promoted = set(promoted or ())
    relegated = set(relegated or ())
    teams = sorted(set(d["home_team_id"]) | set(d["away_team_id"]) | promoted)
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    hi = d["home_team_id"].map(idx).to_numpy()
    ai = d["away_team_id"].map(idx).to_numpy()
    x = d[home_col].to_numpy(dtype=float)
    y = d[away_col].to_numpy(dtype=float)
    if config.shot_weight > 0:
        x, y = _shot_blend(d, x, y, config.shot_weight)
    w = time_weights(d["match_date"], ref_date, config.xi)

    prom_idx = np.array([idx[t] for t in promoted if t in idx], dtype=int)
    rel_idx = np.array([idx[t] for t in relegated if t in idx], dtype=int)
    use_prior = len(prom_idx) > 0 and len(rel_idx) > 0
    is_prom = np.zeros(n_t, bool)
    is_prom[prom_idx] = True
    # Harmanlanmis hedef tam sayi degil; tau yalnizca gercek skorlar icin tanimli.
    use_rho = config.dixon_coles and config.family == "poisson" and config.shot_weight == 0
    use_r = config.family == "negbin"
    lam_pen = config.penalty

    # Parametre vektoru: [c, h, attack(n), defence(n), rho?, log_r?]
    n_par = 2 + 2 * n_t + int(use_rho) + int(use_r)

    def unpack(p):
        c, h = p[0], p[1]
        att, dfn = p[2:2 + n_t], p[2 + n_t:2 + 2 * n_t]
        k = 2 + 2 * n_t
        rho = p[k] if use_rho else 0.0
        log_r = p[k + int(use_rho)] if use_r else 0.0
        return c, h, att, dfn, rho, log_r

    def objective(p):
        c, h, att, dfn, rho, log_r = unpack(p)
        eta_h = c + h + att[hi] + dfn[ai]
        eta_a = c + att[ai] + dfn[hi]
        lam, mu = np.exp(eta_h), np.exp(eta_a)
        ll_h, g_h, gr_h = _count_ll(x, lam, log_r, config.family)
        ll_a, g_a, gr_a = _count_ll(y, mu, log_r, config.family)
        ll = ll_h + ll_a
        g_rho = 0.0
        if use_rho:
            lt, dl, dm, dr = _tau_terms(x, y, lam, mu, rho)
            ll = ll + lt
            g_h = g_h + dl * lam
            g_a = g_a + dm * mu
            g_rho = np.sum(w * dr)
        nll = -np.sum(w * ll)
        wg_h, wg_a = w * g_h, w * g_a
        grad = np.zeros(n_par)
        grad[0] = -(wg_h.sum() + wg_a.sum())
        grad[1] = -wg_h.sum()
        g_att = -(np.bincount(hi, wg_h, n_t) + np.bincount(ai, wg_a, n_t))
        g_def = -(np.bincount(ai, wg_h, n_t) + np.bincount(hi, wg_a, n_t))

        # Ceza: normal takimlar 0'a, yukselenler kume dusenlerin ortalamasina
        for vec, g in ((att, g_att), (dfn, g_def)):
            target = np.zeros(n_t)
            if use_prior:
                target[is_prom] = vec[rel_idx].mean()
            diff = vec - target
            nll += lam_pen * np.sum(diff ** 2)
            g += 2 * lam_pen * diff
            if use_prior:
                g[rel_idx] -= 2 * lam_pen * diff[is_prom].sum() / len(rel_idx)
        grad[2:2 + n_t] = g_att
        grad[2 + n_t:2 + 2 * n_t] = g_def
        k = 2 + 2 * n_t
        if use_rho:
            grad[k] = -g_rho
        if use_r:
            grad[k + int(use_rho)] = -np.sum(w * (gr_h + gr_a))
        return nll, grad

    p0 = np.zeros(n_par)
    mean_goals = max(np.average(np.r_[x, y], weights=np.r_[w, w]), 1e-3) if len(x) else 1.0
    p0[0] = np.log(mean_goals)
    p0[1] = 0.2
    if use_r:
        p0[-1] = np.log(20.0)
    if warm_start is not None:
        p0[0], p0[1] = warm_start.c, warm_start.home
        for t, i in idx.items():
            j = warm_start.index.get(t)
            if j is not None:
                p0[2 + i] = warm_start.attack[j]
                p0[2 + n_t + i] = warm_start.defence[j]
        if use_rho:
            p0[2 + 2 * n_t] = warm_start.rho
    bounds = [(None, None)] * n_par
    if use_rho:
        bounds[2 + 2 * n_t] = RHO_BOUNDS
    if use_r:
        bounds[-1] = (np.log(0.5), np.log(1e4))
    if _return_problem:  # testler gradyani sayisal turevle karsilastirmak icin kullanir
        return objective, p0, bounds  # type: ignore[return-value]
    res = minimize(objective, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    c, h, att, dfn, rho, log_r = unpack(res.x)
    newcomer = (
        (float(att[rel_idx].mean()), float(dfn[rel_idx].mean())) if len(rel_idx) else (0.0, 0.0)
    )
    return FittedModel(config, teams, float(c), float(h), att.copy(), dfn.copy(),
                       rho=float(rho), log_r=float(log_r) if use_r else np.inf,
                       success=bool(res.success), newcomer=newcomer)


# --- Olasilik dagilimlari ve pazarlar -------------------------------------------------

def count_pmf(m: float, max_k: int, family: str = "poisson", log_r: float = np.inf) -> np.ndarray:
    """0..max_k icin olasiliklar; son eleman 'max_k ve ustu' kuyrugunu da icerir."""
    k = np.arange(max_k + 1)
    if family == "poisson" or not np.isfinite(log_r):
        logp = k * np.log(m) - m - gammaln(k + 1)
    else:
        r = np.exp(log_r)
        logp = (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                + r * np.log(r / (r + m)) + k * np.log(m / (r + m)))
    p = np.exp(logp)
    p[-1] += max(0.0, 1 - p.sum())
    return p


def score_matrix(model: FittedModel, home: str, away: str, max_goals: int = 10) -> np.ndarray:
    """M[i, j] = P(ev i, dep j). Dixon-Coles tau dusuk skorlara uygulanir, toplam 1'e normalize."""
    lam, mu = model.rates(home, away)
    fam, lr = model.config.family, model.log_r
    m = np.outer(count_pmf(lam, max_goals, fam, lr), count_pmf(mu, max_goals, fam, lr))
    rho = model.rho
    if rho:
        m[0, 0] *= max(1 - lam * mu * rho, 0)
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
    return m / m.sum()


def goal_markets(m: np.ndarray) -> dict[str, float]:
    i, j = np.indices(m.shape)
    total = i + j
    out = {
        "p_home": m[i > j].sum(), "p_draw": m[i == j].sum(), "p_away": m[i < j].sum(),
        "p_btts": m[(i > 0) & (j > 0)].sum(),
        "exp_home_goals": (i * m).sum(), "exp_away_goals": (j * m).sum(),
    }
    for line in (1.5, 2.5, 3.5):
        out[f"p_over{line:.1f}".replace(".", "")] = m[total > line].sum()
    for side, g in (("home", i), ("away", j)):
        for line in (0.5, 1.5):
            out[f"p_{side}_over{line:.1f}".replace(".", "")] = m[g > line].sum()
    return {k: float(v) for k, v in out.items()}


def correct_score_top(m: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    flat = np.argsort(m, axis=None)[::-1][:n]
    return [(f"{a}-{b}", float(m[a, b])) for a, b in zip(*np.unravel_index(flat, m.shape),
                                                          strict=True)]


def corner_markets(model: FittedModel, home: str, away: str, max_k: int = 30,
                   lines=(8.5, 9.5, 10.5, 11.5)) -> dict[str, float]:
    """Toplam korner dagilimi = iki takimin dagiliminin konvolusyonu (bagimsizlik varsayimi)."""
    lam, mu = model.rates(home, away)
    fam, lr = model.config.family, model.log_r
    total = np.convolve(count_pmf(lam, max_k, fam, lr), count_pmf(mu, max_k, fam, lr))
    k = np.arange(len(total))
    out = {"exp_corners": float(lam + mu), "exp_home_corners": lam, "exp_away_corners": mu}
    for line in lines:
        out[f"p_corners_over{line:.1f}".replace(".", "")] = float(total[k > line].sum())
    return out
