"""
Extraction des données de marché via Interactive Brokers (ib_insync).
Une seule connexion réutilisée pour tous les symboles.
Aucun print : logging uniquement.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import SETTINGS
from .greeks import all_greeks
from .implied_vol import implied_vol

logger = logging.getLogger(__name__)

# ── constantes de configuration ───────────────────────────────────────────────
_REF_VOL    = 0.25    # vol de référence pour le pré-filtrage des strikes
_N_SIGMAS   = 2.0     # demi-largeur de la fenêtre en nombre de σ√T
_N_EXPIRIES = 6       # maturités mensuelles conservées
_BATCH_SIZE = 50      # contrats par lot de reqMktData
_SLEEP_SECS = 2.0     # pause après chaque lot (respect des limites IB)


# ── helpers de conversion ─────────────────────────────────────────────────────

def _expiry_to_T(expiry_str: str) -> float:
    """Convertit YYYYMMDD en T en années (ACT/365.25). Retourne 0 si passé."""
    mat_dt = datetime.strptime(expiry_str, "%Y%m%d")
    days = (mat_dt - datetime.now()).days
    return max(days / SETTINGS.T_BASIS, 0.0)


def _expiry_to_iso(expiry_str: str) -> str:
    """YYYYMMDD → YYYY-MM-DD."""
    return f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"


def _strike_window(spot: float, T: float) -> tuple[float, float]:
    """
    Fenêtre de strikes : intersection de ±_N_SIGMAS*σ_ref*√T et [0.80S, 1.20S].
    Évite d'interroger des strikes dont le delta sera systématiquement hors filtre.
    """
    width = _REF_VOL * math.sqrt(max(T, 1e-4)) * _N_SIGMAS
    lo = max(spot * math.exp(-width), 0.80 * spot)
    hi = min(spot * math.exp(+width), 1.20 * spot)
    return lo, hi


def _select_expiries(expirations: list[str], max_n: int = _N_EXPIRIES) -> list[str]:
    """Retourne au plus max_n maturités futures triées par date croissante."""
    today = date.today().strftime("%Y%m%d")
    return sorted(e for e in expirations if e >= today)[:max_n]


# ── construction des contrats IBKR ────────────────────────────────────────────

def _make_underlying(symbol: str):
    """Index pour SX5E (Eurex), Stock pour les 50 composants."""
    from ib_insync import Index, Stock
    if symbol == "SX5E":
        return Index("ESTX50", "EUREX", "EUR")
    return Stock(symbol, "SMART", "EUR")


def _make_option(symbol: str, expiry: str, strike: float, right: str):
    """Crée le contrat Option adapté au sous-jacent."""
    from ib_insync import Option
    if symbol == "SX5E":
        return Option(
            "ESTX50", expiry, strike, right,
            exchange="EUREX", multiplier="10", currency="EUR",
        )
    return Option(symbol, expiry, strike, right, exchange="SMART", currency="EUR")


# ── extraction pour un symbole ────────────────────────────────────────────────

def fetch_symbol(
    ib,
    symbol: str,
    r: Optional[float] = None,
    q: Optional[float] = None,
) -> pd.DataFrame:
    """
    Extrait les options pour un symbole, calcule IV + grecs, filtre delta.

    Paramètres
    ----------
    ib     : instance IB() déjà connectée
    symbol : 'SX5E' ou un ticker Euro Stoxx 50
    r, q   : taux/dividende (défaut : SETTINGS.r / SETTINGS.q)

    Retourne un DataFrame avec les colonnes standardisées, ou vide si erreur.
    """
    r = r if r is not None else SETTINGS.r
    q = q if q is not None else SETTINGS.q
    multiplier = SETTINGS.MULTIPLIER if symbol == "SX5E" else 1

    # ── 1. spot ───────────────────────────────────────────────────────────
    try:
        underlying = _make_underlying(symbol)
        qualified = ib.qualifyContracts(underlying)
        if not qualified:
            logger.warning("[%s] Contrat introuvable.", symbol)
            return pd.DataFrame()
        underlying = qualified[0]

        bars = ib.reqHistoricalData(
            underlying,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            logger.warning("[%s] Pas d'historique.", symbol)
            return pd.DataFrame()
        spot = bars[-1].close
        logger.info("[%s] Spot = %.2f", symbol, spot)
    except Exception as exc:
        logger.error("[%s] Erreur lors de la récupération du spot : %s", symbol, exc)
        return pd.DataFrame()

    # ── 2. paramètres options ──────────────────────────────────────────────
    try:
        params = ib.reqSecDefOptParams(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
        ib.sleep(1)
        if not params:
            logger.warning("[%s] Aucun paramètre d'options.", symbol)
            return pd.DataFrame()
    except Exception as exc:
        logger.error("[%s] Erreur reqSecDefOptParams : %s", symbol, exc)
        return pd.DataFrame()

    all_strikes: set[float] = set()
    all_expiries: set[str]  = set()
    for p in params:
        all_strikes.update(p.strikes)
        all_expiries.update(p.expirations)

    expiries = _select_expiries(list(all_expiries))
    if not expiries:
        logger.warning("[%s] Aucune maturité future.", symbol)
        return pd.DataFrame()

    # ── 3. boucle sur les maturités ────────────────────────────────────────
    raw_rows: list[dict] = []

    for expiry in expiries:
        T = _expiry_to_T(expiry)
        if T <= 0:
            continue

        lo, hi = _strike_window(spot, T)
        strikes_ok = sorted(k for k in all_strikes if lo <= k <= hi)
        if not strikes_ok:
            logger.debug("[%s %s] Aucun strike dans la fenêtre (%.0f–%.0f).", symbol, expiry, lo, hi)
            continue

        contracts = [
            _make_option(symbol, expiry, strike, right)
            for strike in strikes_ok
            for right in ("C", "P")
        ]

        # Lots de _BATCH_SIZE contrats
        for i in range(0, len(contracts), _BATCH_SIZE):
            batch = contracts[i : i + _BATCH_SIZE]
            if not batch:
                continue

            try:
                qualified_batch = ib.qualifyContracts(*batch)
            except Exception as exc:
                logger.warning("[%s %s] qualifyContracts batch échoué : %s", symbol, expiry, exc)
                continue

            tickers = [ib.reqMktData(c, "", False, False) for c in qualified_batch]
            ib.sleep(_SLEEP_SECS)

            for c, ticker in zip(qualified_batch, tickers):
                try:
                    bid  = float(ticker.bid)  if ticker.bid  not in (None, -1) else math.nan
                    ask  = float(ticker.ask)  if ticker.ask  not in (None, -1) else math.nan
                    last = float(ticker.last) if ticker.last not in (None, -1) else math.nan

                    mid = (bid + ask) / 2.0 if (math.isfinite(bid) and math.isfinite(ask)) else last

                    ib.cancelMktData(c)

                    if not math.isfinite(mid) or mid <= 0:
                        continue

                    raw_rows.append({
                        "Symbol":   symbol,
                        "Spot":     spot,
                        "Strike":   float(c.strike),
                        "Maturity": _expiry_to_iso(expiry),
                        "T":        T,
                        "Type":     c.right,
                        "Bid":      bid,
                        "Ask":      ask,
                        "Mid":      mid,
                    })
                except Exception as exc:
                    logger.debug("[%s] Erreur lecture ticker %s : %s", symbol, c, exc)

    if not raw_rows:
        logger.warning("[%s] Aucun prix collecté.", symbol)
        return pd.DataFrame()

    # ── 4. IV + grecs + filtre delta ──────────────────────────────────────
    records: list[dict] = []
    for row in raw_rows:
        iv = implied_vol(row["Spot"], row["Strike"], row["T"], r, q, row["Mid"], row["Type"])
        if iv is None:
            continue

        g = all_greeks(
            row["Spot"], row["Strike"], row["T"], r, q, iv,
            right=row["Type"], multiplier=multiplier,
        )

        if not (SETTINGS.DELTA_MIN <= g["Delta"] <= SETTINGS.DELTA_MAX):
            continue

        records.append({
            "Symbol":     row["Symbol"],
            "Spot":       row["Spot"],
            "Strike":     row["Strike"],
            "Maturity":   row["Maturity"],
            "T":          row["T"],
            "Type":       row["Type"],
            "Bid":        row["Bid"],
            "Ask":        row["Ask"],
            "Mid":        row["Mid"],
            "ImpliedVol": iv,
            "Delta":      g["Delta"],
            "Gamma":      g["Gamma"],
            "Vega":       g["Vega"],
            "Theta":      g["Theta"],
            "Rho":        g["Rho"],
        })

    result = pd.DataFrame(records)
    logger.info("[%s] %d lignes après filtre delta ±30%%.", symbol, len(result))
    return result


# ── persistence ───────────────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Path:
    """Sauvegarde un snapshot horodaté en Parquet dans data/parquet/."""
    out_dir = out_dir or SETTINGS.parquet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"snapshot_{ts}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Snapshot sauvegardé : %s (%d lignes)", path, len(df))
    return path


def load_latest_snapshot(parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    """Charge le snapshot Parquet le plus récent disponible."""
    parquet_dir = parquet_dir or SETTINGS.parquet_dir
    files = sorted(parquet_dir.glob("snapshot_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def list_snapshots(parquet_dir: Optional[Path] = None) -> list[Path]:
    """Liste tous les snapshots, du plus récent au plus ancien."""
    parquet_dir = parquet_dir or SETTINGS.parquet_dir
    return sorted(parquet_dir.glob("snapshot_*.parquet"), reverse=True)
