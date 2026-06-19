"""
Configuration globale du système de risque.
Adaptée pour l'Euro Stoxx 50.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    # ── connexion IBKR ────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 4002 # Port standard pour IB Gateway en mode Paper/Demo
    client_id: int = 1

    # ── paramètres de marché par défaut ───────────────────────────────────
    r: float = 0.035   # Taux sans risque EUR (ex: ESTER ou Bund ~ 3.5%)
    q: float = 0.0

    # ── filtre delta ──────────────────────────────────────────────────────
    DELTA_ABS_MIN: float = 0.20
    DELTA_ABS_MAX: float = 0.80

    # ── Euro Stoxx 50 (EUREX, OESX, mult=10) ──────────────────────────────
    MULTIPLIER: int = 10  # ATTENTION: Le S&P500 était à 100, le SX5E est à 10 !

    # ── conventions de temps ──────────────────────────────────────────────
    T_BASIS: float = 365.25
    THETA_BASIS: float = 365.0

    # ── chemins ───────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    parquet_dir: Path = field(default_factory=lambda: _ROOT / "data" / "parquet")
    
    # On pointe vers le nouvel univers
    tickers_file: Path = field(
        default_factory=lambda: _ROOT / "data" / "reference" / "euro_stoxx_50_universe.xlsx"
    )

SETTINGS = Settings()