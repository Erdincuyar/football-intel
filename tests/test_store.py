"""Faz 8: veri katmani testleri (sentetik veri).

Postgres'e gidip gelen testler FOOTINTEL_TEST_DATABASE_URL tanimliysa calisir. CI bunu
bir Postgres servis konteyneriyle saglar; yerelde tanimli degilse atlanir.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

import footintel.config as cfg
from footintel import store

TEST_URL = os.environ.get("FOOTINTEL_TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_URL, reason="FOOTINTEL_TEST_DATABASE_URL yok")


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "team": pd.Series(["A", "B", None], dtype="string"),
        "n": np.array([1, 2, 3], dtype="int64"),
        "x": [0.5, np.nan, 2.0],
        "flag": [True, False, True],
        "flag_na": [True, np.nan, False],               # bayraklar: ilk sezonda NaN
        "match_date": [dt.date(2026, 11, 6), dt.date(2026, 11, 7), None],
        "created_at": pd.to_datetime(["2026-11-06 13:54", "2026-11-06 14:00", None]),
    })


def test_pg_types():
    d = sample()
    assert [store.pg_type(d[c]) for c in d] == [
        "text", "bigint", "double precision", "boolean", "boolean", "date", "timestamp"]


def test_py_converts_missing_to_null():
    assert store._py(np.nan) is None and store._py(pd.NA) is None and store._py(pd.NaT) is None
    assert store._py(np.int64(3)) == 3 and type(store._py(np.int64(3))) is int


def test_file_fallback_without_database(tmp_path, monkeypatch):
    monkeypatch.delenv(store.ENV_VAR, raising=False)
    monkeypatch.setattr(cfg, "CLEAN_DIR", tmp_path)
    sample().to_parquet(tmp_path / "style_now.parquet")
    (tmp_path / "xt_grid.json").write_text(json.dumps([[0.1, 0.2]]))
    assert len(store.read_table("style_now")) == 3
    assert store.read_json("xt_grid") == [[0.1, 0.2]]
    assert store.read_table("yok") is None and store.read_json("yok") is None


def test_unreachable_database_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CLEAN_DIR", tmp_path)
    sample().to_parquet(tmp_path / "style_now.parquet")
    bad = "postgresql://x@127.0.0.1:1/x"                 # kimse dinlemiyor
    assert len(store.read_table("style_now", url=bad)) == 3


@needs_db
def test_publish_and_read_roundtrip(tmp_path, monkeypatch):
    clean, led = tmp_path / "clean", tmp_path / "predictions"
    clean.mkdir()
    led.mkdir()
    d = sample()
    d.to_parquet(clean / "style_now.parquet")
    # NaN iceren JSON: jsonb kabul etmezdi, text olarak saklandigi icin sorun degil
    (clean / "style_thresholds.json").write_text(json.dumps({"E0": [{"a": float("nan")}]}))
    out = store.publish(TEST_URL, clean, led)
    assert out["tables"] == {"style_now": 3}

    monkeypatch.setattr(cfg, "CLEAN_DIR", tmp_path / "bos")   # dosyaya dusmesin
    back = store.read_table("style_now", url=TEST_URL)
    pd.testing.assert_frame_equal(back, d, check_dtype=False)
    th = store.read_json("style_thresholds", url=TEST_URL)
    assert np.isnan(th["E0"][0]["a"])
    assert store.published_at(TEST_URL) is not None

    # Ikinci yayin tabloyu bastan yazar (eski satirlar kalmaz)
    d.iloc[:1].to_parquet(clean / "style_now.parquet")
    store.publish(TEST_URL, clean, led)
    assert len(store.read_table("style_now", url=TEST_URL)) == 1


@pytest.mark.parametrize("raw", [
    "postgresql://u:p@ep-x.us-east-1.aws.neon.tech/db?sslmode=require",
    "  psql 'postgresql://u:p@ep-x.us-east-1.aws.neon.tech/db?sslmode=require'\n",
    "\"postgresql://u:p@ep-x.us-east-1.aws.neon.tech/db?sslmode=require\"",
    "postgresql://u:p@ep-x.us-east-1.aws.\nneon.tech/db?sslmode=require",
])
def test_clean_url_fixes_copy_paste(raw):
    url = store.clean_url(raw)
    assert url == "postgresql://u:p@ep-x.us-east-1.aws.neon.tech/db?sslmode=require"
    assert store.describe_url(url) == "adres bicimi dogru gorunuyor"


def test_describe_url_never_prints_secret():
    bad = "postgresql://u:gizli@ep-x..neon.tech/db"
    msg = store.describe_url(bad)
    assert "bozuk" in msg and "gizli" not in msg and "ep-x" not in msg
    assert "baslamiyor" in store.describe_url("neondb_owner:gizli@host")
