#!/usr/bin/env python3
"""
Extrait les données d'options depuis Interactive Brokers et sauvegarde un snapshot Parquet.

Usage :
    python scripts/fetch_data.py                         # tous les symboles
    python scripts/fetch_data.py --symbols SX5E AIR ALV  # symboles spécifiques
    python scripts/fetch_data.py --expiries-max 4        # limite le nombre de maturités
    python scripts/fetch_data.py --dry-run               # sans connexion IBKR
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from risk_system.config import SETTINGS
from risk_system.market_data import fetch_symbol, save_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_data")


def _load_tickers() -> list[str]:
    """Charge les tickers depuis le fichier Excel de référence (header=0 → colonne 'Sigle')."""
    df = pd.read_excel(SETTINGS.tickers_file, header=0)
    tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    seen: dict[str, None] = {}
    for t in tickers:
        seen[t] = None
    return list(seen.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch option data from IBKR.")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        metavar="SYM",
        help="Symboles à traiter (défaut : tous les tickers du fichier Excel + SX5E).",
    )
    parser.add_argument(
        "--expiries-max", type=int, default=None,
        metavar="N",
        help="Nombre maximum de maturités par symbole.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche ce qui serait extrait sans se connecter à IBKR.",
    )
    args = parser.parse_args()

    tickers = ["SX5E"] + [t for t in _load_tickers() if t != "SX5E"]
    symbols: list[str] = args.symbols if args.symbols else tickers

    if args.dry_run:
        logger.info("Dry-run — aucune connexion IBKR.")
        for s in symbols:
            logger.info("  À extraire : %s", s)
        return

    try:
        from ib_insync import IB
    except ImportError:
        logger.error("ib_insync n'est pas installé. Lancez : pip install ib_insync")
        sys.exit(1)

    ib = IB()
    try:
        ib.connect(SETTINGS.host, SETTINGS.port, clientId=SETTINGS.client_id)
        logger.info("Connecté à IBKR (%s:%d).", SETTINGS.host, SETTINGS.port)
    except Exception as exc:
        logger.error("Impossible de se connecter à IBKR : %s", exc)
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    try:
        for symbol in symbols:
            df = fetch_symbol(ib, symbol, max_expiries=args.expiries_max)
            if not df.empty:
                frames.append(df)
    except KeyboardInterrupt:
        logger.warning("Interruption utilisateur — arrêt propre.")
    finally:
        ib.disconnect()
        logger.info("Déconnecté de IBKR.")

    if not frames:
        logger.warning("Aucune donnée collectée. Snapshot non créé.")
        return

    combined = pd.concat(frames, ignore_index=True)
    path = save_snapshot(combined)
    logger.info("Snapshot créé : %s (%d lignes).", path, len(combined))


if __name__ == "__main__":
    main()
