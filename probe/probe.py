"""GECICI Faz 11: FotMob ham ornekleri (lig + mac detayi) indirir, out/ altina yazar."""
import gzip, json, pathlib, time
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36", "Accept-Language": "en"}
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
LEAGUES = {"E0": 47, "SP1": 87, "D1": 54, "I1": 55, "F1": 53, "T1": 71}

def get(url):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=25)
            print(r.status_code, len(r.content), url)
            if r.ok and r.content:
                return r.json()
        except Exception as e:
            print("HATA", type(e).__name__, url)
        time.sleep(2 * (i + 1))

def save(name, obj):
    with gzip.open(OUT / f"{name}.json.gz", "wt") as f:
        json.dump(obj, f)

for code, lid in LEAGUES.items():
    lg = get(f"https://www.fotmob.com/api/data/leagues?id={lid}")
    if not lg: continue
    save(f"league_{code}", lg)
    seasons = lg.get("allAvailableSeasons") or []
    print(code, "sezonlar:", seasons[:6])
    fx = (lg.get("fixtures") or {}).get("allMatches") or []
    done = [m for m in fx if m.get("status", {}).get("finished")]
    nxt = [m for m in fx if not m.get("status", {}).get("finished")]
    picks = done[-2:] + nxt[:2] if code in ("E0", "T1") else done[-1:] + nxt[:1]
    for m in picks:
        d = get(f"https://www.fotmob.com/api/data/matchDetails?matchId={m['id']}")
        if d: save(f"match_{code}_{m['id']}", d)
        time.sleep(1)
    if code == "E0" and len(seasons) > 1:
        old = get(f"https://www.fotmob.com/api/data/leagues?id={lid}&season={seasons[1]}")
        if old:
            save("league_E0_prev", old)
            ofx = (old.get("fixtures") or {}).get("allMatches") or []
            if ofx:
                d = get(f"https://www.fotmob.com/api/data/matchDetails?matchId={ofx[100]['id']}")
                if d: save(f"match_E0prev_{ofx[100]['id']}", d)
    time.sleep(1)
print(sorted(p.name for p in OUT.iterdir()))
