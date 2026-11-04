"""Understat'tan lig-sezon JSON verisini indirir.

Understat'in resmi bir API'si yok. Sitenin kendi sayfalari veriyi tarayicida
`/getLeagueData/{lig}/{yil}` adresinden JSON olarak cekiyor; biz de ayni adresi
kullaniyoruz. Bu adres "X-Requested-With: XMLHttpRequest" basligi ve ana sayfadan
alinan cerezler olmadan cevap vermiyor. Resmi olmayan bir kaynak oldugu icin
format haber verilmeden degisebilir: ham JSON'u sakliyoruz ve temizleme adiminda
beklenen alanlari dogruluyoruz.

Bir istek bir lig-sezonun TUM maclarini ve takim bazinda mac istatistiklerini
(xG, npxG, PPDA, deep completion) verir. Sut bazli veri (koordinatlar) mac basina
ayri istek gerektirir; onu sonraki adima birakiyoruz.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from footintel.config import (
    CURRENT_SEASON,
    RAW_DIR,
    SEASONS,
    UNDERSTAT_LEAGUES,
    season_start_year,
)
from footintel.ingest.http import PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://understat.com"
RAW_SUBDIR = RAW_DIR / "understat"


def raw_path(league: str, season: str) -> Path:
    return RAW_SUBDIR / f"{league}_{season}.json"


def download_all(*, force: bool = False) -> list[Path]:
    session = PoliteSession(delay=2.0, headers={"X-Requested-With": "XMLHttpRequest"})
    session.get(BASE_URL).raise_for_status()  # cerezleri al
    paths = []
    for league, slug in UNDERSTAT_LEAGUES.items():
        for season in SEASONS:
            path = raw_path(league, season)
            if path.exists() and not force and season != CURRENT_SEASON:
                paths.append(path)
                continue
            url = f"{BASE_URL}/getLeagueData/{slug}/{season_start_year(season)}"
            response = session.get(url, headers={"Referer": f"{BASE_URL}/league/{slug}"})
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                # Hata ayiklamak icin sunucunun ne dondurdugunu goster.
                raise ValueError(
                    f"{url}: JSON beklenirken baska bir cevap geldi "
                    f"(HTTP {response.status_code}, "
                    f"Content-Type={response.headers.get('Content-Type')!r}, "
                    f"{len(response.content)} bayt): {response.text[:300]!r}"
                ) from exc
            for key in ("dates", "teams"):
                if key not in data:
                    raise ValueError(f"{url}: beklenen '{key}' alani yok; format degismis olabilir")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            log.info("Indirildi: %s (%d mac)", path.name, len(data["dates"]))
            paths.append(path)
    return paths
