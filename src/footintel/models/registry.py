"""Canlida kullanilan model surumu. Defterdeki her tahmin bu surum etiketini tasir.

Kural: Ayarlar degisirse surum adi da degisir. Boylece eski tahminler hangi
modelle yapildiklarini kaybetmez ve iki surum ayni maclar uzerinde kiyaslanabilir.
Ayarlar Faz 3 walk-forward ayarindan (2024/25) gelir: pipelines/build_models.py.
"""

from __future__ import annotations

from footintel.models.strength import ModelConfig

MODEL_VERSION = "dc-sot-v1"
GOAL_CONFIG = ModelConfig(family="poisson", dixon_coles=True, xi=0.004, penalty=2.0,
                          shot_weight=0.5)
CORNER_CONFIG = ModelConfig(family="negbin", dixon_coles=False, xi=0.004, penalty=32.0)
