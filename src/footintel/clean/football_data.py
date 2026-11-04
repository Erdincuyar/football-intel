"""football-data.co.uk ham CSV dosyalarini analize hazir tabloya cevirir.

Bu modulde ogrenilecek temel fikirler:

1. Sema sozlesmesi: Ham dosyalar sezondan sezona ve ligden lige farkli kolonlar
   tasiyabilir. Biz cikista HER ZAMAN ayni kolonlari, ayni tiplerle uretiriz.
   Kaynakta olmayan kolon NA (eksik) olarak eklenir; asla 0 ile doldurulmaz.
   0 "korner olmadi" demektir, NA ise "bilmiyoruz". Bu fark modelde kritik.

2. Deterministik kimlik: match_id ayni maci her calistirmada ayni koda
   cevirir. Boylece yeniden yukleme yinelenen satir uretmez.

3. Dogrulama iki seviyelidir:
   - Yapisal hatalar (yinelenen mac, skorla uyusmayan sonuc) tum hatti durdurur,
     cunku bunlar dosyanin kendisinin bozuk oldugunu gosterir.
   - Tek bir istatistikteki imkansiz deger (isabetli sut > toplam sut) ise o
     macin ilgili degerlerini NA yapar ve satira `quality_flags` notu duser.
     Kaynak verilerde tek tuk yazim hatasi normaldir; bir hucre yuzunden 30
     dosyalik hatti durdurmak yerine hatayi izole ediyor ve raporluyoruz.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from footintel.clean.teams import team_id
from footintel.config import season_label

log = logging.getLogger(__name__)

# Kaynak kolon -> bizim kolon. Sol taraf football-data.co.uk notes.txt'deki adlar.
STAT_COLUMNS: dict[str, str] = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

# Piyasa ortalamasi (Avg) ve kapanis (AvgC) oranlari. Tek bir bahis sirketi yerine
# ortalama kullanmak, sirkete ozgu marji ve hatalari yumusatir.
ODDS_COLUMNS: dict[str, str] = {
    "AvgH": "odds_home_avg",
    "AvgD": "odds_draw_avg",
    "AvgA": "odds_away_avg",
    "Avg>2.5": "odds_over25_avg",
    "Avg<2.5": "odds_under25_avg",
    "AvgCH": "odds_home_close",
    "AvgCD": "odds_draw_close",
    "AvgCA": "odds_away_close",
    "AvgC>2.5": "odds_over25_close",
    "AvgC<2.5": "odds_under25_close",
}

OUTPUT_COLUMNS: list[str] = [
    "match_id",
    "league",
    "season",
    "season_label",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "result",
    "referee",
    "quality_flags",
    *STAT_COLUMNS.values(),
    *ODDS_COLUMNS.values(),
]


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Ham dosyayi okur. Kodlama sorunlarini ve bos satirlari burada cozeriz.

    football-data dosyalari bazen UTF-8 (BOM'lu), bazen Latin-1 kodlanir. Once
    UTF-8 deneriz; olmazsa Latin-1, cunku Latin-1 her bayti okuyabilir.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text))
    # Bazi dosyalarin sonunda tamamen bos satir ve 'Unnamed: 105' gibi kolonlar olur.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.dropna(subset=["HomeTeam", "AwayTeam"])


def clean_matches(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Ham lig-sezon tablosunu sema sozlesmesine uygun tabloya cevirir."""
    df = pd.DataFrame(index=raw.index)

    df["league"] = league
    df["season"] = season
    df["season_label"] = season_label(season)
    # Tarih gun/ay/yil formatinda. dayfirst=True vermezsek 03/11 Mart 11 sanilir.
    df["match_date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed").dt.date
    df["kickoff_time"] = raw["Time"] if "Time" in raw else pd.NA
    df["home_team"] = raw["HomeTeam"].str.strip()
    df["away_team"] = raw["AwayTeam"].str.strip()
    df["home_team_id"] = df["home_team"].map(team_id)
    df["away_team_id"] = df["away_team"].map(team_id)
    df["result"] = raw["FTR"] if "FTR" in raw else pd.NA
    df["referee"] = raw["Referee"] if "Referee" in raw else pd.NA

    missing = [c for c in [*STAT_COLUMNS, *ODDS_COLUMNS] if c not in raw.columns]
    if missing:
        log.info("%s %s: kaynakta olmayan kolonlar NA eklendi: %s", league, season, missing)

    for src, dst in STAT_COLUMNS.items():
        # Int64 (buyuk I): pandas'in NA destekleyen tamsayi tipi. Normal int NA tutamaz.
        values = raw[src] if src in raw else pd.Series(pd.NA, index=raw.index)
        df[dst] = pd.to_numeric(values, errors="coerce").astype("Int64")
    for src, dst in ODDS_COLUMNS.items():
        values = raw[src] if src in raw else pd.Series(pd.NA, index=raw.index)
        df[dst] = pd.to_numeric(values, errors="coerce").astype("Float64")

    date_key = pd.to_datetime(df["match_date"]).dt.strftime("%Y%m%d")
    df["match_id"] = (
        df["league"] + "-" + df["season"] + "-" + date_key + "-"
        + df["home_team_id"] + "-" + df["away_team_id"]
    )

    df = quarantine_impossible_stats(df)
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    validate(df, context=f"{league} {season}")
    return df


# (kontrol adi, buyuk olamayacak kolon, ust sinir kolonu)
STAT_RULES: list[tuple[str, str, str]] = [
    (f"{side}_{kind}", f"{side}_{small}", f"{side}_{big}")
    for side in ("home", "away")
    for kind, small, big in [
        ("sot_gt_shots", "shots_on_target", "shots"),
        ("ht_gt_ft", "goals_ht", "goals"),
    ]
]


def quarantine_impossible_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Imkansiz istatistik ciftlerini NA yapar ve satira not duser.

    Ornek: away_shots_on_target=6, away_shots=5 ise ikisinden hangisinin yanlis
    oldugunu bilemeyiz; bu yuzden ikisini de NA yapariz. Mac sonu golu (FTHG)
    ise sonuc ve skor icin kritik oldugundan asla silinmez; ilk yari golu
    hataliysa yalnizca ilk yari degeri NA olur.
    """
    df = df.copy()
    flags = pd.Series("", index=df.index, dtype="string")
    for name, small, big in STAT_RULES:
        bad = (df[small] > df[big]).fillna(False).astype(bool)
        if not bad.any():
            continue
        log.warning(
            "%s: %d macta %s > %s, degerler NA yapildi: %s",
            name, bad.sum(), small, big, df.loc[bad, "match_id"].tolist(),
        )
        df.loc[bad, small] = pd.NA
        if not big.endswith("_goals"):
            df.loc[bad, big] = pd.NA
        flags[bad] = flags[bad] + name + ";"
    df["quality_flags"] = flags.str.rstrip(";").replace("", pd.NA)
    return df


def validate(df: pd.DataFrame, context: str = "") -> None:
    """Yapisal kontroller. Biri bile basarisizsa hata verir, sessizce gecmez."""
    problems: list[str] = []

    if df["match_id"].duplicated().any():
        dupes = df.loc[df["match_id"].duplicated(), "match_id"].tolist()
        problems.append(f"yinelenen match_id: {dupes[:5]}")

    if (df["home_team_id"] == df["away_team_id"]).any():
        problems.append("ev sahibi ve deplasman ayni takim")

    for name, small, big in STAT_RULES:
        bad = (df[small] > df[big]).fillna(False).astype(bool)
        if bad.any():
            problems.append(f"{name}: {df.loc[bad, 'match_id'].tolist()[:5]}")

    played = df["home_goals"].notna()
    expected = pd.Series(pd.NA, index=df.index, dtype="string")
    expected[played & (df["home_goals"] > df["away_goals"])] = "H"
    expected[played & (df["home_goals"] < df["away_goals"])] = "A"
    expected[played & (df["home_goals"] == df["away_goals"])] = "D"
    mismatch = played & (expected != df["result"]).fillna(True)
    if mismatch.any():
        problems.append(f"FTR sonucu skorla uyusmuyor: {df.loc[mismatch, 'match_id'].tolist()[:5]}")

    if problems:
        raise ValueError(f"Veri dogrulama basarisiz ({context}): " + "; ".join(problems))


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Lig-sezon bazinda her istatistigin dolu olma yuzdesi.

    Bu rapor Faz 1'deki acik soruyu cevaplar: Super Lig dosyalarinda sut ve
    korner gercekten var mi?
    """
    cols = ["home_shots", "home_shots_on_target", "home_corners", "home_fouls",
            "referee", "odds_home_avg", "odds_home_close", "odds_over25_avg"]
    report = df.groupby(["league", "season"])[cols].agg(lambda s: s.notna().mean() * 100)
    report.insert(0, "matches", df.groupby(["league", "season"]).size())
    report["flagged"] = df.groupby(["league", "season"])["quality_flags"].count()
    return report.round(1)
