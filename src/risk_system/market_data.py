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
# Maturités cibles (jours calendaires) : 2 semaines → 36 mois
_TARGET_MATURITIES_DAYS = (14, 30, 91, 182, 273, 365, 548, 730, 1095)
# Écart max toléré entre une cible et l'échéance cotée la plus proche.
_MATURITY_TOL_DAYS = 45
_BATCH_SIZE = 50      # contrats par lot de reqMktData
_SLEEP_SECS = 2.0     # pause après chaque lot (respect des limites IB)

# Type de market data :
# 1 = temps réel (abonnement requis), 3 = différé, 4 = différé-figé.
# 3 retombe automatiquement sur le temps réel si l'abonnement existe.
_MARKET_DATA_TYPE = 3


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


def _expiry_days(expiry_str: str) -> int:
    """Jours calendaires entre aujourd'hui et l'échéance."""
    return (datetime.strptime(expiry_str, "%Y%m%d") - datetime.now()).days


def _select_expiries(expirations: list[str]) -> list[str]:
    """
    Pour chaque maturité cible, retient l'échéance cotée future dont le
    nombre de jours est le plus proche, si l'écart reste sous la tolérance.
    Déduplique et trie par date croissante.
    """
    futures = [(e, _expiry_days(e)) for e in expirations if _expiry_days(e) > 0]
    if not futures:
        return []
    chosen = set()
    for target in _TARGET_MATURITIES_DAYS:
        best_e, best_gap = None, None
        for e, days in futures:
            gap = abs(days - target)
            if best_gap is None or gap < best_gap:
                best_e, best_gap = e, gap
        if best_e is not None and best_gap <= _MATURITY_TOL_DAYS:
            chosen.add(best_e)
    return sorted(chosen)


def _in_delta_band(delta_val: float, right: str, q: float, T: float) -> bool:
    """
    Bande "du put 30Δ au call 30Δ" autour de l'ATM.

    On convertit en delta call-équivalent via la parité des deltas
    (Δ_put = Δ_call − e^(-qT)) :
        Δc = delta            si call
        Δc = delta + e^(-qT)  si put
    et on conserve si  |DELTA_MIN| ≤ Δc ≤ e^(-qT) − DELTA_MAX,
    soit [0.30, 0.70] pour q = 0. ATM (Δc ≈ 0.50) inclus ;
    deep ITM (Δc > 0.70) et deep OTM (Δc < 0.30) exclus.
    """
    disc_q = math.exp(-q * T)
    d_call = delta_val if right == "C" else delta_val + disc_q
    lo = abs(SETTINGS.DELTA_MIN)
    hi = disc_q - SETTINGS.DELTA_MAX
    return lo <= d_call <= hi


# ── construction des contrats IBKR ────────────────────────────────────────────

_CURRENCIES = ("EUR", "CHF", "GBP", "USD")

# Bourses dérivés européennes préférées pour la sonde d'options
_PREFERRED_OPT_EXCHANGES = (
    "EUREX", "MONEP", "FTA", "SOFFEX", "IDEM", "MEFFRV", "OMS", "DTB",
)


def _resolve_underlying(ib, symbol: str):
    """
    Résout le contrat sous-jacent.
    Index Eurex pour SX5E ; sinon Stock en essayant plusieurs devises
    (certains tickers de la liste ne cotent pas en EUR, ex: UBSG en CHF).
    Retourne le contrat qualifié, ou None.
    """
    from ib_insync import Index, Stock

    if symbol == "SX5E":
        qualified = ib.qualifyContracts(Index("ESTX50", "EUREX", "EUR"))
        return qualified[0] if qualified else None

    for ccy in _CURRENCIES:
        try:
            qualified = ib.qualifyContracts(Stock(symbol, "SMART", ccy))
        except Exception:
            qualified = []
        if qualified:
            if ccy != "EUR":
                logger.info("[%s] Résolu en %s (pas EUR).", symbol, ccy)
            return qualified[0]
    return None


def _select_option_param(params: list, symbol: str):
    """
    Choisit l'entrée de reqSecDefOptParams à utiliser pour la sonde.

    Les options européennes ne sont pas SMART-routables : il faut
    interroger la bourse dérivés native (MONEP, EUREX...). On préfère
    donc une bourse connue, sinon la première entrée non-SMART,
    sinon la première tout court.
    """
    if symbol == "SX5E":
        oesx = [p for p in params if p.tradingClass == "OESX"]
        if oesx:
            return oesx[0]

    for exch in _PREFERRED_OPT_EXCHANGES:
        for p in params:
            if p.exchange == exch:
                return p
    non_smart = [p for p in params if p.exchange != "SMART"]
    if non_smart:
        return non_smart[0]
    return params[0]


def _valid_contracts_for_expiry(
    ib, symbol: str, expiry: str, exchange: str, trading_class: str, currency: str,
) -> list:
    """
    Récupère, en UNE requête, tous les contrats d'options réellement
    existants pour (symbole, échéance) via reqContractDetails, sur la
    bourse dérivés native du sous-jacent.

    Les contrats retournés sont déjà entièrement qualifiés (conId inclus) :
    plus besoin de qualifyContracts.
    """
    from ib_insync import Option

    probe = Option(
        "ESTX50" if symbol == "SX5E" else symbol,
        lastTradeDateOrContractMonth=expiry,
        exchange=exchange,
        currency=currency,
    )
    if trading_class:
        probe.tradingClass = trading_class
    if symbol == "SX5E":
        probe.multiplier = "10"

    try:
        details = ib.reqContractDetails(probe)
    except Exception as exc:
        logger.warning("[%s %s] reqContractDetails échoué : %s", symbol, expiry, exc)
        return []
    return [d.contract for d in details]


# ── extraction pour un symbole ────────────────────────────────────────────────

def fetch_symbol(
    ib,
    symbol: str,
    r: Optional[float] = None,
    q: Optional[float] = None,
    max_expiries: Optional[int] = None,
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

    # Données différées si pas d'abonnement temps réel (idempotent).
    ib.reqMarketDataType(_MARKET_DATA_TYPE)

    # ── 1. spot ───────────────────────────────────────────────────────────
    try:
        underlying = _resolve_underlying(ib, symbol)
        if underlying is None:
            logger.warning("[%s] Contrat introuvable (devises testées : %s).",
                           symbol, ", ".join(_CURRENCIES))
            return pd.DataFrame()

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

    # ── 2. liste des maturités disponibles ────────────────────────────────
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

    # Sélection de la bourse dérivés native + sa trading class :
    # on n'utilise QUE les échéances de cette entrée (cohérence garantie
    # entre échéances, strikes et bourse — fin des erreurs 200).
    param = _select_option_param(params, symbol)
    logger.info("[%s] Options via %s (tradingClass=%s).",
                symbol, param.exchange, param.tradingClass)

    expiries = _select_expiries(list(param.expirations))
    if max_expiries is not None:
        expiries = expiries[:max_expiries]
    if not expiries:
        logger.warning("[%s] Aucune maturité future.", symbol)
        return pd.DataFrame()

    # ── 3. boucle sur les maturités ────────────────────────────────────────
    raw_rows: list[dict] = []

    for expiry in expiries:
        T = _expiry_to_T(expiry)
        if T <= 0:
            continue

        # Contrats réellement existants pour cette échéance (déjà qualifiés)
        contracts = _valid_contracts_for_expiry(
            ib, symbol, expiry,
            exchange=param.exchange,
            trading_class=param.tradingClass,
            currency=underlying.currency,
        )
        if not contracts:
            logger.debug("[%s %s] Aucun contrat coté.", symbol, expiry)
            continue

        # Pré-filtrage par fenêtre de strikes (≈ |delta| ≤ 0.30)
        lo, hi = _strike_window(spot, T)
        contracts = [c for c in contracts if lo <= c.strike <= hi]
        if not contracts:
            logger.debug(
                "[%s %s] Aucun strike dans la fenêtre (%.0f–%.0f).",
                symbol, expiry, lo, hi,
            )
            continue

        logger.info("[%s %s] %d contrats dans la fenêtre.", symbol, expiry, len(contracts))

        # Lots de _BATCH_SIZE contrats
        for i in range(0, len(contracts), _BATCH_SIZE):
            batch = contracts[i : i + _BATCH_SIZE]

            tickers = [ib.reqMktData(c, "", False, False) for c in batch]
            ib.sleep(_SLEEP_SECS)

            for c, ticker in zip(batch, tickers):
                try:
                    def _f(x) -> float:
                        return float(x) if x is not None and x == x and x > 0 else math.nan

                    bid  = _f(ticker.bid)
                    ask  = _f(ticker.ask)
                    last = _f(ticker.last)
                    close = _f(ticker.close)

                    if math.isfinite(bid) and math.isfinite(ask):
                        mid = (bid + ask) / 2.0
                    elif math.isfinite(last):
                        mid = last
                    else:
                        mid = close   # dernier recours en données différées-figées

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

        if not _in_delta_band(g["Delta"], row["Type"], q, row["T"]):
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
    logger.info("[%s] %d lignes dans la bande [put 30Δ, call 30Δ].", symbol, len(result))
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