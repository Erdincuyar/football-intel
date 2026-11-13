"""GECICI Faz 11: API-Football ucretsiz plan erisim testi (~15 istek, anahtar yazdirilmaz)."""
import json, os, sys
import requests

KEY = os.environ.get("APIFOOTBALL_KEY", "").strip()
if not KEY:
    print("APIFOOTBALL_KEY tanimli degil; test atlandi."); sys.exit(0)
BASE = "https://v3.football.api-sports.io"
LEAGUES = {"E0": 39, "SP1": 140, "D1": 78, "I1": 135, "F1": 61, "T1": 203}

def call(path, **params):
    r = requests.get(BASE + path, headers={"x-apisports-key": KEY}, params=params, timeout=30)
    d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    print(f"\n== {path} {params} -> {r.status_code}, results={d.get('results')}, errors={d.get('errors')}")
    return d

s = call("/status")
resp = s.get("response") or {}
print("  plan:", (resp.get("subscription") or {}).get("plan"), "| istek:", resp.get("requests"))

d = call("/leagues", id=203)
for x in (d.get("response") or [])[:1]:
    for se in x["seasons"][-3:]:
        print("  sezon", se["year"], "current=", se["current"], "kapsam=", json.dumps(se.get("coverage"))[:300])

for season in (2026, 2025, 2024):
    d = call("/fixtures", league=203, season=season, next=3) if season == 2026 else \
        call("/fixtures", league=203, season=season, round="Regular Season - 10")
    for f in (d.get("response") or [])[:2]:
        print("  ", f["fixture"]["id"], f["fixture"]["date"], f["teams"]["home"]["name"], "-", f["teams"]["away"]["name"])

d = call("/injuries", league=39, season=2026)
for x in (d.get("response") or [])[:3]:
    print("  ", x["player"]["name"], "|", x["player"].get("type"), x["player"].get("reason"), "|", x["fixture"].get("date"))
d = call("/injuries", league=203, season=2026)

d = call("/fixtures", league=39, season=2025, round="Regular Season - 10")
fid = (d.get("response") or [{}])[0].get("fixture", {}).get("id")
if fid:
    st = call("/fixtures/statistics", fixture=fid)
    for t in (st.get("response") or [])[:1]:
        print("  istatistikler:", [x["type"] for x in t["statistics"]])
    lu = call("/fixtures/lineups", fixture=fid)
    for t in (lu.get("response") or [])[:1]:
        print("  dizilis:", t.get("formation"), "| hoca:", (t.get("coach") or {}).get("name"), "| ilk11:", len(t.get("startXI", [])))
call("/coachs", team=645)
