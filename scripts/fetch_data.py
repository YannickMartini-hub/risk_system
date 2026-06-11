#!/usr/bin/env python3
"""
CLI pour l'extraction des données de marché IBKR.

Usage :
    python scripts/fetch_data.py
    python scripts/fetch_data.py --symbols SX5E ASML.AS LVMH.PA
    python scripts/fetch_data.py --expiries-max 4
    python scripts/fetch_data.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permet de lancer le script directement sans pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_data")


def _load_tickers() -> list[str]:
    """Charge la liste des 50 tickers depuis data/reference/."""
    from risk_system.config import SETTINGS
    df = pd.read_excel(SETTINGS.tickers_file, header=None)
    return df.iloc[:, 0].dropna().tolist()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extrait les options IBKR (SX5E + Euro Stoxx 50) et sauvegarde en Parquet.",
    )
    p.add_argument(
        "--symbols", nargs="*", default=None,
        help="Symboles à traiter (défaut : SX5E + 50 tickers)",
    )
    p.add_argument(
        "--expiries-max", type=int, default=6,
        help="Nombre maximum de maturités mensuelles (défaut : 6)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simule le run sans connexion IBKR ni sauvegarde",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = ["SX5E"] + _load_tickers()

    logger.info("Symboles à traiter : %d  (premiers : %s)", len(symbols), symbols[:5])

    if args.dry_run:
        logger.info("[dry-run] Connexion IBKR ignorée. Symboles : %s", symbols)
        return

    from ib_insync import IB
    from risk_system.config import SETTINGS
    from risk_system.market_data import fetch_symbol, save_snapshot

    ib = IB()
    try:
        ib.connect(SETTINGS.host, SETTINGS.port, clientId=SETTINGS.client_id, timeout=15)
        logger.info("Connecté à IBKR Gateway %s:%d", SETTINGS.host, SETTINGS.port)

        all_frames: list[pd.DataFrame] = []

        for symbol in symbols:
            logger.info("=== %s ===", symbol)
            df = fetch_symbol(ib, symbol)
            if not df.empty:
                all_frames.append(df)
            else:
                logger.warning("%s : aucune donnée retenue, ignoré.", symbol)

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            path = save_snapshot(combined)
            logger.info("Snapshot : %s  (%d lignes)", path, len(combined))
        else:
            logger.warning("Aucune donnée collectée pour l'ensemble des symboles.")

    except Exception:
        logger.exception("Erreur fatale lors de l'extraction.")
        sys.exit(1)
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Déconnexion IBKR.")


if __name__ == "__main__":
    main()
