"""GECICI Faz 11 kaynak denemesi, tur 2: Understat yeni bicim, FotMob mac detayi icerigi."""
import json, re
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36", "Accept-Language": "en"}

def get(url, **kw):
    try:
        return requests.get(url, headers=UA | kw.pop("headers", {}), timeout=25, **kw)
    except Exception as e:
        print("  HATA", url, type(e).__name__); return None

def keys(o, depth=0, maxd=2, pre=""):
    if depth > maxd: return
    if isinstance(o, dict):
        for k, v in list(o.items())[:40]:
            t = type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict)) else "")
            print("   " * depth + f"{pre}{k}: {t}")
            keys(v, depth + 1, maxd)
    elif isinstance(o, list) and o:
        keys(o[0], depth, maxd, pre="[0].")

# Understat
r = get("https://understat.com/league/EPL/2026")
print("understat scripts:", re.findall(r'<script[^>]*src="([^"]+)"', r.text)[:10] if r else None)
print("understat JSON.parse:", r.text.count("JSON.parse") if r else None)
for u in ["https://understat.com/getLeagueData/EPL/2026", "https://understat.com/getLeagueData/EPL/2025"]:
    rr = get(u, headers={"X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/league/EPL"})
    print("\n==", u, rr.status_code if rr is not None else None, rr.headers.get("content-type") if rr is not None else "")
    if rr is not None and rr.ok:
        try:
            d = rr.json(); keys(d, maxd=1)
            dates = d.get("dates") or []
            print("  maclar:", len(dates), dates[0] if dates else None)
        except Exception as e:
            print("  json degil", rr.text[:200])

# FotMob: Super Lig (71) lig verisi -> bitmis bir mac id -> mac detayi
lg = get("https://www.fotmob.com/api/data/leagues?id=71").json()
print("\n== fotmob league keys"); keys(lg, maxd=0)
fx = lg.get("fixtures", {}).get("allMatches") or lg.get("matches", {}).get("allMatches") or []
done = [m for m in fx if m.get("status", {}).get("finished")]
nxt = [m for m in fx if not m.get("status", {}).get("finished")]
print("  bitmis:", len(done), "gelecek:", len(nxt))
for label, m in (("bitmis", done[-1] if done else None), ("gelecek", nxt[0] if nxt else None)):
    if not m: continue
    mid = m["id"]
    for u in [f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}",
              f"https://www.fotmob.com/api/matchDetails?matchId={mid}"]:
        rr = get(u)
        print(f"\n== {label} {m.get('home',{}).get('name')}-{m.get('away',{}).get('name')} {u.split('/api/')[1][:25]}:",
              rr.status_code if rr is not None else None, len(rr.content) if rr is not None else 0)
        if rr is not None and rr.ok and rr.content:
            d = rr.json(); keys(d, maxd=1)
            s = json.dumps(d).lower()
            for k in ("expected_goals", "unavailable", "injur", "suspen", "coach", "lineup", "predicted"):
                print("    ", k, s.count(k))
            lu = d.get("content", {}).get("lineup")
            if lu: print("  lineup keys:"); keys(lu, maxd=2)
            break
