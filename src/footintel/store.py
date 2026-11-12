"""Faz 8: uygulamanin veri katmani (Neon PostgreSQL, yerel dosya yedegiyle).

Neden boyle? Streamlit Community Cloud uygulamayi GitHub reposundan calistirir; ama temiz
veri (data/clean) CI'in icinde uretilir ve repoya yazilmaz. Her calismada parquet
dosyalarini commit'lemek repoyu sisirir. Bu yuzden CI, uygulamanin ihtiyac duydugu
tablolari bir Postgres veritabanina (Neon, ucretsiz katman) yazar; uygulama oradan okur.

Iki yol, tek arayuz:
- FOOTINTEL_DATABASE_URL tanimliysa okuma/yazma Postgres'e gider.
- Tanimli degilse (yerel gelistirme, testler) data/clean altindaki dosyalar okunur.
Sayfalar yalnizca `read_table` / `read_json` cagirir; verinin nereden geldigini bilmez.

JSON ciktilari jsonb degil text olarak saklanir: Python'un json modulu NaN yazabiliyor ve
Postgres'in jsonb tipi NaN'i kabul etmiyor.

Yayinlama tek islemde (transaction) yapilir: tablolar silinip yeniden yazilirken uygulama
yarim veri gormez; commit'e kadar eski surumu okumaya devam eder.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import footintel.config as cfg

log = logging.getLogger(__name__)

ENV_VAR = "FOOTINTEL_DATABASE_URL"
SCHEMA = "app"
BOOL_OID = 16                     # Postgres boolean tip kimligi

# Uygulamanin okudugu tablolar (data/clean/<ad>.parquet) ve JSON ciktilari.
APP_TABLES = ["matches_football_data", "backtest_predictions", "style_profile",
              "tactics_team_match_1516", "style_now",
              # Faz 9 turetilmis tablolar (her calismada bastan uretilir; tahminler degil)
              "match_results", "prediction_evaluation", "evaluation_summary",
              "evaluation_metrics"]
APP_JSON = ["tactics_proxy", "tactics_summary", "xt_grid", "matchup_rules",
            "style_thresholds", "style_clusters"]


def clean_url(raw: str) -> str:
    """Kopyalama sirasinda sik gelen bozulmalari temizler.

    Neon'un baglanti kutusu bazen `psql 'postgresql://...'` biciminde kopyalanir; secrets
    kutusuna yapistirirken tirnak, bosluk veya satir sonu da eklenebilir. Bunlar adresin
    parcasi degildir.
    """
    url = raw.strip()
    if url.lower().startswith("psql "):
        url = url[5:].strip()
    while len(url) >= 2 and url[0] == url[-1] and url[0] in "'\"":
        url = url[1:-1].strip()
    return "".join(url.split())                 # adreste bosluk/satir sonu olamaz


def describe_url(url: str) -> str:
    """Adresi YAZDIRMADAN neyin yanlis olabilecegini soyler (sifre ve sunucu gizli kalir)."""
    from urllib.parse import urlsplit

    if not url.startswith(("postgresql://", "postgres://")):
        return "adres 'postgresql://' ile baslamiyor"
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return "adres ayrisitirilamadi"
    if not host:
        return "adreste sunucu adi yok"
    labels = host.split(".")
    if any(not lb or len(lb) > 63 for lb in labels):
        return f"sunucu adi bozuk ({len(labels)} parca, bos ya da 63 karakterden uzun parca var)"
    if not host.endswith("neon.tech"):
        return "sunucu adi neon.tech ile bitmiyor"
    return "adres bicimi dogru gorunuyor"


def database_url() -> str | None:
    """Baglanti adresi: once ortam degiskeni (CI), sonra Streamlit secrets (Cloud).
    Gizli bilgi hicbir zaman koda veya repoya yazilmaz."""
    url = os.environ.get(ENV_VAR, "").strip()
    if not url:
        try:                                    # Streamlit Cloud: Advanced settings > Secrets
            import streamlit as st

            url = str(st.secrets.get(ENV_VAR, "")).strip()
        except Exception:                       # streamlit yok veya secrets dosyasi yok
            url = ""
    return clean_url(url) or None


def _connect(url: str):
    import psycopg  # yalnizca veritabani kullanilirken gerekli

    return psycopg.connect(url, connect_timeout=10)


# --- pandas tipi -> Postgres tipi ------------------------------------------------------

def pg_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "bigint"
    if pd.api.types.is_float_dtype(s):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "timestamptz" if getattr(s.dt, "tz", None) is not None else "timestamp"
    sample = s.dropna()
    if len(sample):
        v = sample.iloc[0]
        if isinstance(v, (bool, np.bool_)):
            return "boolean"
        if isinstance(v, dt.datetime):
            return "timestamp"
        if isinstance(v, dt.date):
            return "date"
    return "text"


def _py(v):
    """Bir hucreyi Postgres'e yazilacak Python degerine cevirir (NaN -> NULL)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
        if isinstance(v, float) and np.isnan(v):
            return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _write_table(cur, name: str, df: pd.DataFrame) -> None:
    from psycopg import sql

    ident = sql.Identifier(SCHEMA, name)
    cols = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(str(c)), sql.SQL(pg_type(df[c])))
        for c in df.columns)
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(ident))
    cur.execute(sql.SQL("CREATE TABLE {} ({})").format(ident, cols))
    copy = sql.SQL("COPY {} ({}) FROM STDIN").format(
        ident, sql.SQL(", ").join(sql.Identifier(str(c)) for c in df.columns))
    with cur.copy(copy) as cp:
        for row in df.itertuples(index=False, name=None):
            cp.write_row([_py(v) for v in row])


def publish(url: str, clean_dir: Path | None = None, ledger_dir: Path | None = None) -> dict:
    """Uygulama tablolarini ve JSON ciktilarini veritabanina yazar (tek islem)."""
    clean_dir = clean_dir or cfg.CLEAN_DIR
    ledger_dir = ledger_dir or cfg.LEDGER_DIR
    frames: dict[str, pd.DataFrame] = {}
    for name in APP_TABLES:
        p = clean_dir / f"{name}.parquet"
        if p.exists():
            frames[name] = pd.read_parquet(p)
        else:
            log.warning("%s yok, atlaniyor", p.name)
    if any(ledger_dir.glob("*.parquet")):
        from footintel import ledger

        frames["predictions"] = ledger.load(ledger_dir)
    docs = {n: (clean_dir / f"{n}.json").read_text() for n in APP_JSON
            if (clean_dir / f"{n}.json").exists()}

    with _connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        for name, df in frames.items():
            _write_table(cur, name, df)
        cur.execute(f"CREATE TABLE IF NOT EXISTS {SCHEMA}.json_docs "
                    "(name text PRIMARY KEY, body text NOT NULL)")
        for name, body in docs.items():
            cur.execute(f"INSERT INTO {SCHEMA}.json_docs VALUES (%s, %s) "
                        "ON CONFLICT (name) DO UPDATE SET body = EXCLUDED.body", (name, body))
        cur.execute(f"CREATE TABLE IF NOT EXISTS {SCHEMA}.meta "
                    "(key text PRIMARY KEY, value text NOT NULL)")
        cur.execute(f"INSERT INTO {SCHEMA}.meta VALUES ('published_at', %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),))
        # commit, with blogundan cikarken (hata olursa hicbir sey yazilmaz)
    return {"tables": {k: len(v) for k, v in frames.items()}, "json": sorted(docs)}


def append_immutable(url: str, name: str, df: pd.DataFrame, key: list[str]) -> int:
    """Yalnizca-ekleme tablosu: anahtar zaten varsa satir YAZILMAZ (eski tahmin korunur).

    Gecmis tahminler bir kez yazilir; ayni model surumuyle yeniden calistirmak hicbir seyi
    degistirmez, yeni surum yeni satirlar ekler. Donen deger: eklenen satir sayisi.
    """
    from psycopg import sql

    ident = sql.Identifier(SCHEMA, name)
    stage = sql.Identifier(f"_stage_{name}")
    cols = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(str(c)), sql.SQL(pg_type(df[c])))
        for c in df.columns)
    names = sql.SQL(", ").join(sql.Identifier(str(c)) for c in df.columns)
    with _connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({}, PRIMARY KEY ({}))").format(
            ident, cols, sql.SQL(", ").join(sql.Identifier(k) for k in key)))
        cur.execute(sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT DROP").format(stage, cols))
        with cur.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(stage, names)) as cp:
            for row in df.itertuples(index=False, name=None):
                cp.write_row([_py(v) for v in row])
        cur.execute(sql.SQL("INSERT INTO {} ({}) SELECT {} FROM {} ON CONFLICT DO NOTHING")
                    .format(ident, names, names, stage))
        return cur.rowcount


# --- okuma ---------------------------------------------------------------------------

def read_table(name: str, url: str | None = None) -> pd.DataFrame | None:
    """Tabloyu okur: once veritabani, o yoksa data/clean/<ad>.parquet. Hicbiri yoksa None."""
    url = url if url is not None else database_url()
    if url:
        from psycopg import sql

        try:
            with _connect(url) as conn, conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(SCHEMA, name)))
                cols = [d.name for d in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
                # NULL iceren boolean kolonlar (or. ilk sezonun bayraklari) parquet'teki
                # gibi True/False/NaN olsun; pandas None'u farkli bir deger sayar.
                for d in cur.description:
                    if d.type_code == BOOL_OID and df[d.name].isna().any():
                        df[d.name] = df[d.name].astype(object).where(df[d.name].notna(),
                                                                      np.nan)
                return df
        except Exception as exc:  # baglanti veya tablo yoksa dosyaya dus
            log.warning("Veritabanindan %s okunamadi (%s; %s); dosyaya bakiliyor", name,
                        type(exc).__name__, describe_url(url))
    p = cfg.CLEAN_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def read_json(name: str, url: str | None = None):
    url = url if url is not None else database_url()
    if url:
        try:
            with _connect(url) as conn, conn.cursor() as cur:
                cur.execute(f"SELECT body FROM {SCHEMA}.json_docs WHERE name = %s", (name,))
                row = cur.fetchone()
                if row is not None:
                    return json.loads(row[0])
        except Exception as exc:
            log.warning("Veritabanindan %s okunamadi (%s; %s); dosyaya bakiliyor", name,
                        type(exc).__name__, describe_url(url))
    p = cfg.CLEAN_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def published_at(url: str | None = None) -> str | None:
    url = url if url is not None else database_url()
    if not url:
        return None
    try:
        with _connect(url) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT value FROM {SCHEMA}.meta WHERE key = 'published_at'")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None

