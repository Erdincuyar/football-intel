"""Faz 6: StatsBomb 2015/16 taktik metrikleri + guncel sezon stil profili.

Adimlar:
1. StatsBomb acik verisini indir (2015/16; PL, La Liga, Serie A, Ligue 1). Bir kez.
2. xT izgarasini fit et, takim-mac taktik metriklerini hesapla.
3. Analiz: guvenilirlik, ongoru gucu, basit sayilarla yaklasiklik (proxy).
4. Dis dogrulama: PPDA'mizin takim siralamasi Understat'in yayimladigi PPDA ile tutarli mi?
4b. Tanim kontrolu: StatsBomb'dan saydigimiz sut/korner/faul, ayni sezonun
   football-data.co.uk sayilariyla ayni mi? (Proxy'yi football-data'ya tasimanin on kosulu.)
5. Guncel sezon (ve bir onceki) icin football-data'dan stil profili, Super Lig dahil.

Kullanim:
    uv run python pipelines/build_tactics.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from footintel.analysis import tactics as A
from footintel.clean.football_data import clean_matches, read_raw_csv
from footintel.config import CLEAN_DIR, CURRENT_SEASON, SEASONS
from footintel.features import tactics as T
from footintel.ingest import football_data as fd_ingest
from footintel.ingest import statsbomb as SB
from footintel.ingest import understat as US

log = logging.getLogger("build_tactics")
PROFILE_SEASONS = [SEASONS[SEASONS.index(CURRENT_SEASON) - 1], CURRENT_SEASON]


def definition_check(tm: pd.DataFrame) -> pd.DataFrame:
    """Lig bazinda mac basina ortalama: StatsBomb sayimi vs football-data (2015/16)."""
    rows = []
    for lg in SB.COMPETITIONS:
        try:
            path = fd_ingest.download_season(lg, SB.SEASON)
        except Exception as exc:          # kaynak erisilemezse kontrolu atla, hatti durdurma
            log.warning("%s 2015/16 football-data indirilemedi: %s", lg, exc)
            continue
        if path is None:
            continue
        fd = clean_matches(read_raw_csv(path), lg, SB.SEASON)
        sb = tm[tm["league"] == lg]
        for stat, col in (("shots", "shots"), ("sot", "shots_on_target"),
                          ("corners", "corners"), ("fouls", "fouls")):
            fd_mean = (fd[f"home_{col}"].astype(float) + fd[f"away_{col}"].astype(float)).mean()
            rows.append({"league": lg, "stat": stat, "statsbomb": sb[stat].sum() / sb[
                "match_id"].nunique(), "football_data": fd_mean})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ratio"] = out["statsbomb"] / out["football_data"]
    return out


def ppda_check(tm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faz 6 kapisinin acik maddesi: PPDA'mizi yayimlanmis (Understat) degerlerle karsilastir."""
    us = {}
    for lg in SB.COMPETITIONS:
        try:
            data = json.loads(US.download_season(lg, SB.SEASON).read_text(encoding="utf-8"))
            us[lg] = A.understat_team_ppda(data)
        except Exception as exc:            # kaynak erisilemezse kontrolu atla
            log.warning("%s 2015/16 Understat indirilemedi: %s", lg, exc)
    return A.ppda_external_check(tm, us) if us else (pd.DataFrame(), pd.DataFrame())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for lg in SB.COMPETITIONS:
        SB.download_league(lg)
    matches, events = SB.load()
    grid = T.fit_xt(events)
    tm = T.team_match_metrics(events, matches, grid)
    del events
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    tm.to_parquet(CLEAN_DIR / "tactics_team_match_1516.parquet", index=False)
    (CLEAN_DIR / "xt_grid.json").write_text(json.dumps(np.round(grid, 5).tolist()))
    log.info("Taktik metrikler: %d takim-mac satiri", len(tm))

    metrics = [*T.STYLE_METRICS, *T.OUTCOME_METRICS, "fouls_against"]
    rel = A.reliability(tm, metrics)
    pred = A.predictive_power(tm, ["xg_share", "xt_share", "field_tilt", "pass_share",
                                   "shot_share", "sot_share", "corner_share"])
    ts = A.team_season(tm, metrics)
    proxies = [A.proxy_fit(ts, t, A.PROXY_FEATURES)
               for t in [*A.PROXY_TARGETS, "ppda", "def_height"]]
    usable = [p for p in proxies if p["r2_leave_league_out"] >= 0.6]
    check = definition_check(tm)
    ppda_sum, ppda_teams = ppda_check(tm)

    (CLEAN_DIR / "tactics_proxy.json").write_text(json.dumps(usable, indent=1))
    summary = {
        "reliability": rel.round(4).reset_index().to_dict("records"),
        "predictive": pred.round(4).reset_index().to_dict("records"),
        "proxies": [{k: v for k, v in p.items() if k != "beta"} for p in proxies],
        "definition_check": check.round(3).to_dict("records"),
        "ppda_vs_understat": ppda_sum.round(3).to_dict("records"),
    }
    if not ppda_teams.empty:
        ppda_teams.to_csv(CLEAN_DIR / "ppda_vs_understat_1516.csv", index=False)
    (CLEAN_DIR / "tactics_summary.json").write_text(json.dumps(summary, indent=1))

    fd_path = CLEAN_DIR / "matches_football_data.parquet"
    profile = pd.DataFrame()
    if fd_path.exists():
        fd = pd.read_parquet(fd_path)
        fd = fd[fd["season"].isin(PROFILE_SEASONS)].dropna(subset=["home_goals", "away_goals"])
        profile = A.style_profile(fd, usable)
        profile.to_parquet(CLEAN_DIR / "style_profile.parquet", index=False)
    else:
        log.warning("%s yok (once build_football_data); stil profili atlandi", fd_path.name)

    pd.set_option("display.width", 160)
    text = "\n\n".join([
        "Guvenilirlik (sezon boyu, Spearman-Brown):\n" + rel.round(3).to_string(),
        "Ilk yari metrik -> ikinci yari puan:\n" + pred.round(3).to_string(),
        "Proxy (leave-one-league-out R^2):\n" + pd.DataFrame(summary["proxies"])[
            ["target", "r2_leave_league_out", "rmse_leave_league_out", "target_sd"]
        ].round(3).to_string(index=False),
        "Tanim kontrolu (mac basina, 2015/16):\n" + (check.round(2).to_string(index=False)
                                                   if not check.empty else "atlandi"),
        "PPDA dis dogrulama (Understat 2015/16, Spearman siralama):\n"
        + (ppda_sum.round(3).to_string(index=False) if not ppda_sum.empty else "atlandi"),
        f"Stil profili: {len(profile)} takim-sezon ({', '.join(PROFILE_SEASONS)})",
    ])
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as f:
            f.write(f"## Faz 6: taktik metrikler\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
