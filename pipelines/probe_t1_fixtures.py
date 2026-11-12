"""GECICI: Super Lig icin acik fikstur kaynaklarini CI'dan dener (konteynerden erisilemiyor).

Sonuca gore biri secilip openfootball'a ek kaynak olarak yazilacak; sonra bu dosya silinecek.
"""

from __future__ import annotations

import json

import requests

URLS = [
    "https://fixturedownload.com/feed/json/turkey-super-lig-2026",
    "https://fixturedownload.com/feed/json/super-lig-2026",
    "https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id=4339&s=2026-2027",
    "https://www.thesportsdb.com/api/v1/json/3/eventsround.php?id=4339&r=13&s=2026-2027",
    "https://site.api.espn.com/apis/site/v2/sports/soccer/tur.1/scoreboard?dates=20261119-20261125",
]

for url in URLS:
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "football-intel/1.0"})
        body = r.text[:600].replace("\n", " ")
        print(f"\n=== {r.status_code} {url}\n{body}")
        if r.status_code == 200:
            try:
                d = r.json()
                if isinstance(d, list):
                    print("liste uzunlugu:", len(d), "ornek:", json.dumps(d[:2])[:400])
                elif isinstance(d, dict):
                    for k, v in d.items():
                        n = len(v) if hasattr(v, "__len__") else ""
                        print("anahtar:", k, type(v).__name__, n)
            except ValueError:
                print("JSON degil")
    except Exception as exc:
        print(f"\n=== HATA {url}: {type(exc).__name__}: {exc}")
