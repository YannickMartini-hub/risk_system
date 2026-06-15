"""
Configuration globale du système de risque.
Aucun chemin absolu : tous les chemins sont résolus relativement à la racine du projet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Racine du projet : deux niveaux au-dessus de src/risk_system/
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Paramètres globaux (connexion IBKR, marché, chemins, conventions)."""

    # ── connexion IBKR ────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1

    # ── paramètres de marché par défaut ───────────────────────────────────
    r: float = 0.025
    q: float = 0.0

    # ── filtre delta : borne basse = ATM − 30Δ, borne haute = ATM + 30Δ ──
    # En valeur absolue : 0.20 ≤ |Δ| ≤ 0.80 (calls et puts)
    DELTA_ABS_MIN: float = 0.20
    DELTA_ABS_MAX: float = 0.80

    # ── SX5E (Eurex) ──────────────────────────────────────────────────────
    MULTIPLIER: int = 10

    # ── conventions de temps ──────────────────────────────────────────────
    T_BASIS: float = 365.25   # ACT/365.25 pour annualiser T
    THETA_BASIS: float = 365.0  # theta par jour calendaire

    # ── chemins relatifs à la racine ──────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    parquet_dir: Path = field(default_factory=lambda: _ROOT / "data" / "parquet")
    reference_dir: Path = field(default_factory=lambda: _ROOT / "data" / "reference")
    tickers_file: Path = field(
        default_factory=lambda: _ROOT / "data" / "reference" / "euro_stoxx_50_tickers.xlsx"
    )

    def __post_init__(self) -> None:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
