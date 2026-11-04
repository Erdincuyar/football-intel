"""Kaynaklar arasi takim ismi eslestirme (entity resolution).

Problem: football-data "Man United", Understat "Manchester United", ClubElo
"Man United" yazar. Super Lig'de football-data "Buyuksehyr", ClubElo "Basaksehir".

Yaklasim, iki katmanli:
1. Elle yazilmis eslemeler (data/reference/team_aliases.csv). Her zaman kazanir.
2. Otomatik benzerlik eslemesi: AYNI lig-sezondaki takimlar arasinda yapilir.
   Bir lig-sezonda yalnizca 18-20 takim oldugu icin aday kumesi kucuktur ve
   yanlis esleme ihtimali dusuktur. Esleme bire-bir olmak zorundadir.

Esik altinda kalan isim otomatik eslenmez; hata listesinde gosterilir ve
alias dosyasina elle eklenmesi beklenir. Yanlis bir esleme, iki farkli takimin
verisini sessizce birlestirir; bu, hic eslememekten cok daha tehlikelidir.
"""

from __future__ import annotations

import csv
import difflib
from dataclasses import dataclass
from pathlib import Path

from footintel.clean.teams import slugify
from footintel.config import REFERENCE_DIR

ALIASES_PATH = REFERENCE_DIR / "team_aliases.csv"

# Isimlerde ayirt edici olmayan kelimeler. "AC Milan" ile "Milan" ayni takim.
STOPWORDS = {
    "fc", "cf", "ac", "as", "sc", "afc", "ssc", "us", "ss", "rc", "ud", "cd", "sd",
    "club", "de", "calcio", "sk", "fk", "jk", "spor", "kulubu", "olympique", "stade",
    "1907", "1905", "1903", "1909", "1913", "1919",
}
MIN_SCORE = 0.72


@dataclass(frozen=True)
class MatchResult:
    source_name: str
    team_id: str | None
    score: float
    method: str  # "alias", "fuzzy", "unmatched"


def normalize(name: str) -> str:
    tokens = [t for t in slugify(name).split("-") if t not in STOPWORDS]
    return " ".join(tokens) or slugify(name)


def similarity(a: str, b: str) -> float:
    """0-1 arasi benzerlik. Biri digerini tamamen iceriyorsa ('milan' / 'ac milan') yuksek."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return 1.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        ratio = max(ratio, 0.9)
    # Kisaltmalar: "man city" vs "manchester city" -> on ek eslesmesi
    if all(any(y.startswith(x) or x.startswith(y) for y in tb) for x in ta):
        ratio = max(ratio, 0.85)
    return ratio


def load_aliases(source: str, path: Path = ALIASES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["source"].startswith("#")]
    return {r["source_name"]: r["team_id"] for r in rows if r["source"] == source}


def match_teams(
    source_names: list[str],
    candidates: dict[str, str],
    aliases: dict[str, str] | None = None,
) -> list[MatchResult]:
    """Kaynak isimlerini football-data takim kimliklerine esler.

    candidates: {team_id: football-data ismi}, ayni lig-sezondaki takimlar.
    """
    aliases = aliases or {}
    results: dict[str, MatchResult] = {}
    free = dict(candidates)

    for name in source_names:
        if name in aliases:
            results[name] = MatchResult(name, aliases[name], 1.0, "alias")
            free.pop(aliases[name], None)

    # Butun (kaynak, aday) ciftlerini skora gore sirala, en iyiden baslayarak
    # bire-bir ata (greedy). Kucuk kumelerde Hungarian algoritmasiyla ayni sonucu verir.
    pending = [n for n in source_names if n not in results]
    pairs = sorted(
        ((similarity(n, fd_name), n, tid) for n in pending for tid, fd_name in free.items()),
        reverse=True,
    )
    for score, name, tid in pairs:
        if name in results or tid not in free:
            continue
        if score < MIN_SCORE:
            break
        results[name] = MatchResult(name, tid, round(score, 3), "fuzzy")
        free.pop(tid)

    for name in pending:
        if name not in results:
            best = max(free.items(), key=lambda kv: similarity(name, kv[1]), default=(None, ""))
            score = similarity(name, best[1]) if best[0] else 0.0
            results[name] = MatchResult(name, None, round(score, 3), "unmatched")
    return [results[n] for n in source_names]


def describe_unmatched(results: list[MatchResult], candidates: dict[str, str], ctx: str) -> str:
    left = sorted(set(candidates) - {r.team_id for r in results if r.team_id})
    names = [r.source_name for r in results if r.team_id is None]
    return f"{ctx}: eslenemeyen {names}; bos kalan football-data takimlari {left}"
