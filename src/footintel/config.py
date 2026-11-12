"""Projenin tek ayar noktasi: ligler, sezonlar ve dosya yollari.

Ayarlari koda dagitmak yerine burada toplamak, ileride bir lig veya sezon
eklemeyi tek satirlik bir degisiklige indirir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# FOOTINTEL_DATA_DIR: testlerde ve arayuz denemelerinde baska bir veri klasoru kullanmak icin.
DATA_DIR = Path(os.environ.get("FOOTINTEL_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
REFERENCE_DIR = DATA_DIR / "reference"
DB_PATH = DATA_DIR / "footintel.duckdb"
LEDGER_DIR = DATA_DIR / "predictions"
SQL_DIR = PROJECT_ROOT / "sql"


@dataclass(frozen=True)
class League:
    code: str  # football-data.co.uk dosya kodu
    name: str
    country: str


LEAGUES: dict[str, League] = {
    "E0": League("E0", "Premier League", "England"),
    "SP1": League("SP1", "La Liga", "Spain"),
    "T1": League("T1", "Super Lig", "Turkey"),
    "I1": League("I1", "Serie A", "Italy"),
    "F1": League("F1", "Ligue 1", "France"),
    "D1": League("D1", "Bundesliga", "Germany"),
}

# Sezon kodu football-data.co.uk formatinda: "2526" = 2025/26.
# 5 tamamlanmis sezon + guncel sezon. Modeller icin 5 sezon yeterli bir gecmis;
# daha eskisi oyun stillerinin degismesi nedeniyle bilgi degeri dusuk.
SEASONS: list[str] = ["2122", "2223", "2324", "2425", "2526", "2627"]
CURRENT_SEASON = "2627"


def season_label(code: str) -> str:
    """'2526' -> '2025/26'"""
    return f"20{code[:2]}/{code[2:]}"


def season_start_year(code: str) -> int:
    """'2526' -> 2025"""
    return 2000 + int(code[:2])


# Understat lig kodlari (URL'deki yazim). Super Lig Understat'ta yok.
UNDERSTAT_LEAGUES: dict[str, str] = {
    "E0": "EPL",
    "SP1": "La_liga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
    "D1": "Bundesliga",
}

# ClubElo ulke kodlari. ClubElo'da Level=1 en ust lig demektir.
CLUBELO_COUNTRIES: dict[str, str] = {
    "E0": "ENG",
    "SP1": "ESP",
    "T1": "TUR",
    "I1": "ITA",
    "F1": "FRA",
    "D1": "GER",
}
