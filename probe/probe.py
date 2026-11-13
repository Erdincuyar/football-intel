"""GECICI Faz 11 kaynak denemesi: erisim, kapsam ve icerik ozeti (yalnizca okuma, az istek)."""
import json, re, sys, time
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36", "Accept-Language": "en"}

def get(url, **kw):
    try:
        r = requests.get(url, headers=UA | kw.pop("headers", {}), timeout=25, **kw)
        return r
    except Exception as e:
        print("  HATA", url, type(e).__name__, str(e)[:120]); return None

def show(name, r, n=300):
    if r is None: return
    print(f"\n== {name}: {r.status_code} {r.headers.get('content-type','')[:40]} {len(r.content)}B")
    print("  ", r.text[:n].replace("\n", " "))

# 1 Understat
r = get("https://understat.com/league/EPL/2026"); show("understat league", r, 200)
if r is not None and r.ok: print("  has teamsData:", "teamsData" in r.text, "maintenance" in r.text.lower())

# 2 FotMob
for u in ["https://www.fotmob.com/api/leagues?id=47&ccode3=TUR",
          "https://www.fotmob.com/api/data/leagues?id=71",
          "https://www.fotmob.com/api/matchDetails?matchId=4813374"]:
    r = get(u); show("fotmob " + u.split("/api/")[1][:30], r, 200)
r = get("https://www.fotmob.com/leagues/71/overview/super-lig"); show("fotmob html super lig", r, 150)
if r is not None and r.ok:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    print("  NEXT_DATA:", bool(m), len(m.group(1)) if m else 0)
    if m:
        d = json.loads(m.group(1)); s = json.dumps(d)
        for k in ("expected_goals", "xg", "injur", "unavailable", "coach", "lineup"):
            print("   ", k, s.lower().count(k))

# 3 Transfermarkt injuries (Super Lig = TR1)
r = get("https://www.transfermarkt.com/super-lig/verletztespieler/wettbewerb/TR1"); show("transfermarkt TR1 injuries", r, 150)
if r is not None and r.ok: print("  rows ~", r.text.count('class="odd"') + r.text.count('class="even"'))
r = get("https://www.transfermarkt.com/robots.txt"); show("transfermarkt robots", r, 600)

# 4 Wikipedia managerial changes (CC BY-SA)
for t in ["2025–26 Premier League", "2025–26 Süper Lig", "2026–27 Premier League"]:
    r = get("https://en.wikipedia.org/w/api.php", params={"action": "parse", "page": t,
            "prop": "sections", "format": "json"})
    if r is not None and r.ok:
        secs = [s["line"] for s in r.json().get("parse", {}).get("sections", [])]
        print("\n== wikipedia", t, "| managerial section:", [s for s in secs if "anager" in s or "oach" in s])

# 5 ClubElo, football-data.org (no key)
show("clubelo", get("http://api.clubelo.com/2026-11-13"), 120)
show("football-data.org (anahtarsiz)", get("https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"), 200)
# 6 Sofascore
show("sofascore", get("https://api.sofascore.com/api/v1/unique-tournament/52/seasons"), 200)
