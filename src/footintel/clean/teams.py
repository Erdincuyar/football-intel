"""Takim kimlikleri.

Ayni takim her kaynakta farkli yazilir: football-data "Man United", Understat
"Manchester United", FBref "Manchester Utd". Kaynaklari birlestirebilmek icin her
takima tek bir kimlik (team_id) veriyoruz.

Kanonik kaynak football-data.co.uk: 5 ligin hepsini kapsayan tek kaynak o.
Diger kaynaklarin isimleri data/reference/team_aliases.csv dosyasinda bu
kimliklere eslenir (Adim 2'de, Understat eklenince).
"""

from __future__ import annotations

import re
import unicodedata


def slugify(name: str) -> str:
    """'Atletico Madrid' -> 'atletico-madrid', 'Göztepe' -> 'goztepe'."""
    # Turkce 'ı' NFKD ile ayrismaz ve ascii'ye cevrilirken kaybolur; once elle cevirelim.
    name = name.replace("ı", "i").replace("İ", "I")
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    if not slug:
        raise ValueError(f"Takim adindan kimlik uretilemedi: {name!r}")
    return slug


def team_id(source_name: str) -> str:
    return slugify(source_name)
