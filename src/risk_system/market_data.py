"""
Extraction des données de marché via Interactive Brokers (ib_insync).
Univers : SPX (Index/CBOE) + top 50 S&P 500 (Stock/SMART/USD).
Une seule connexion réutilisée pour tous les symboles.
Aucun print : logging uniquement.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .config import SETTINGS
from .greeks import all_greeks
from .implied_vol import implied_vol

logger = logging.getLogger(__name__)


class _SuppressDelayedDataFallback(logging.Filter):
    """
    Supprime les codes IBKR 10090 et 10167 qui signalent simplement que
    les données différées sont utilisées à la place du temps réel.
    Ces messages sont attendus dès que _MARKET_DATA_TYPE = 3 et qu'il
    n'y a pas d'abonnement temps réel — les logger comme ERROR est trompeur.
    """
    _CODES = {"10090", "10167"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(
            f"Error {c}," in msg or f"Warning {c}," in msg
            for c in self._CODES
        )


def _install_ibkr_log_filter() -> None:
    """Installe le filtre sur les loggers ib_insync (idempotent)."""
    _filter = _SuppressDelayedDataFallback()
    for name in ("ib_insync.wrapper", "ib_insync.ib", "ib_insync"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _SuppressDelayedDataFallback) for f in lg.filters):
            lg.addFilter(_filter)


# ── constantes de configuration ───────────────────────────────────────────────
# Maturités cibles (jours calendaires) : 2 semaines → 36 mois
_TARGET_MATURITIES_DAYS = (14, 30, 91, 182, 273, 365, 548, 730, 1095)
# Écart max toléré entre une cible et l'échéance cotée la plus proche.
_MATURITY_TOL_DAYS = 45
_BATCH_SIZE = 50      # contrats par lot de reqMktData
_SLEEP_SECS = 2.0     # pause après chaque lot (respect des limites IB)

# Type de market data :
# 1 = temps réel (abonnement requis), 3 = différé, 4 = différé-figé.
_MARKET_DATA_TYPE = 3

_ET = ZoneInfo("America/New_York")

# Mapping tickers → symbole IBKR (cas spéciaux)
_IBKR_SYMBOL_MAP = {"BRK.B": "BRK B"}


def _in_us_hours() -> bool:
    """True si on est dans les heures de trading US (lun-ven, 09h30-16h00 ET)."""
    now = datetime.now(tz=_ET)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= t < 16 * 60


# ── helpers de conversion ─────────────────────────────────────────────────────

def _expiry_to_T(expiry_str: str) -> float:
    """Convertit YYYYMMDD en T en années (ACT/365.25). Retourne 0 si passé."""
    mat_dt = datetime.strptime(expiry_str, "%Y%m%d")
    days = (mat_dt - datetime.now()).days
    return max(days / SETTINGS.T_BASIS, 0.0)


def _expiry_to_iso(expiry_str: str) -> str:
    """YYYYMMDD → YYYY-MM-DD."""
    return f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"


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


# ── construction des contrats IBKR ────────────────────────────────────────────

def _resolve_underlying_us(ib, symbol: str):
    """
    Résout le contrat sous-jacent pour l'univers S&P 500.
    SPX → Index("SPX", "CBOE", "USD")
    Stocks → Stock(ibkr_symbol, "SMART", "USD")
    Retourne le contrat qualifié, ou None.
    """
    from ib_insync import Index, Stock

    ibkr_sym = _IBKR_SYMBOL_MAP.get(symbol, symbol)

    if symbol == "SPX":
        qualified = ib.qualifyContracts(Index("SPX", "CBOE", "USD"))
        return qualified[0] if qualified else None

    try:
        qualified = ib.qualifyContracts(Stock(ibkr_sym, "SMART", "USD"))
    except Exception:
        qualified = []
    if qualified:
        return qualified[0]
    logger.warning("[%s] Contrat introuvable (SMART/USD).", symbol)
    return None


def _select_option_param_us(params: list, symbol: str):
    """
    Choisit l'entrée de reqSecDefOptParams pour la sonde d'options.

    SPX : préférer CBOE + tradingClass="SPX" (AM-settled mensuel standard).
    Stocks : n'importe quelle exchange non-SMART (OCC), sinon la première.
    """
    if symbol == "SPX":
        spx_params = [p for p in params if p.exchange == "CBOE" and p.tradingClass == "SPX"]
        if spx_params:
            return spx_params[0]
        cboe = [p for p in params if p.exchange == "CBOE"]
        if cboe:
            return cboe[0]

    non_smart = [p for p in params if p.exchange != "SMART"]
    if non_smart:
        return non_smart[0]
    return params[0]


def _valid_contracts_for_expiry_us(
    ib, symbol: str, expiry: str, exchange: str, trading_class: str, currency: str,
) -> list:
    """
    Récupère tous les contrats d'options réellement existants pour
    (symbole, échéance) via reqContractDetails.
    Les contrats retournés sont déjà qualifiés (conId inclus).
    """
    from ib_insync import Option

    ibkr_sym = "SPX" if symbol == "SPX" else _IBKR_SYMBOL_MAP.get(symbol, symbol)

    probe = Option(
        ibkr_sym,
        lastTradeDateOrContractMonth=expiry,
        exchange=exchange,
        currency=currency,
    )
    if symbol == "SPX":
        probe.multiplier = "100"
    if trading_class:
        probe.tradingClass = trading_class

    try:
        details = ib.reqContractDetails(probe)
    except Exception as exc:
        logger.warning("[%s %s] reqContractDetails échoué : %s", symbol, expiry, exc)
        return []
    return [d.contract for d in details]


# ── calcul du forward par parité put-call ────────────────────────────────────

def _compute_forwards(records: list[dict], r: float) -> dict[str, float]:
    """
    Calcule le forward implicite F(T) par maturité via la parité put-call :
        F ≈ K + e^(rT) * (C_mid - P_mid)
    Utilise les 5 strikes les plus proches de l'ATM avec call ET put disponibles.
    Retourne dict[maturity_iso] → forward.
    Fallback : Forward = Spot si pas assez de paires.
    """
    if not records:
        return {}
    df = pd.DataFrame(records)
    forwards: dict[str, float] = {}

    for mat, grp in df.groupby("Maturity"):
        spot = float(grp["Spot"].iloc[0])
        T    = float(grp["T"].iloc[0])

        calls = grp[grp["Type"] == "C"].set_index("Strike")["Mid"]
        puts  = grp[grp["Type"] == "P"].set_index("Strike")["Mid"]
        common = sorted(set(calls.index) & set(puts.index))

        if not common:
            forwards[mat] = spot
            continue

        atm = min(common, key=lambda k: abs(k - spot))
        idx = common.index(atm)
        near = common[max(0, idx - 2) : idx + 3]

        f_list = [k + math.exp(r * T) * (calls[k] - puts[k]) for k in near]
        forwards[mat] = sum(f_list) / len(f_list)

    return forwards


# ── extraction pour un symbole ────────────────────────────────────────────────

def fetch_symbol(
    ib,
    symbol: str,
    r: Optional[float] = None,
    q: Optional[float] = None,
    max_expiries: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extrait les options pour un symbole SPX ou action S&P 500,
    calcule IV + grecs + forward par parité put-call.

    Paramètres
    ----------
    ib     : instance IB() déjà connectée
    symbol : 'SPX' ou un ticker S&P 500 (ex: 'AAPL', 'BRK.B')
    r, q   : taux/dividende (défaut : SETTINGS.r / SETTINGS.q)

    Note : Black-Scholes européen utilisé pour tout.
    Pour les options sur actions US (style américain), c'est une approximation
    acceptable sur les maturités courtes et pour les grecs de premier ordre.

    Retourne un DataFrame avec les colonnes standardisées, ou vide si erreur.
    """
    r = r if r is not None else SETTINGS.r
    q = q if q is not None else SETTINGS.q
    multiplier = SETTINGS.MULTIPLIER  # 100 pour SPX et actions US

    _install_ibkr_log_filter()
    ib.reqMarketDataType(_MARKET_DATA_TYPE)

    # ── 1. spot ───────────────────────────────────────────────────────────
    try:
        underlying = _resolve_underlying_us(ib, symbol)
        if underlying is None:
            logger.warning("[%s] Contrat introuvable.", symbol)
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
        logger.error("[%s] Erreur récupération spot : %s", symbol, exc)
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

    param = _select_option_param_us(params, symbol)
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

    if _in_us_hours():
        logger.info("[%s] Marché US ouvert — bid/ask live.", symbol)
    else:
        logger.info("[%s] Hors heures US — fallback sur close du jour.", symbol)

    def _f(x) -> float:
        return float(x) if x is not None and x == x and x > 0 else math.nan

    def _fvol(x) -> float:
        if x is None or x != x:
            return math.nan
        v = float(x)
        return v if v >= 0 else math.nan

    for expiry in expiries:
        T = _expiry_to_T(expiry)
        if T <= 0:
            continue

        contracts = _valid_contracts_for_expiry_us(
            ib, symbol, expiry,
            exchange=param.exchange,
            trading_class=param.tradingClass,
            currency=underlying.currency,
        )
        if not contracts:
            logger.debug("[%s %s] Aucun contrat coté.", symbol, expiry)
            continue

        logger.info("[%s %s] %d contrats.", symbol, expiry, len(contracts))

        for i in range(0, len(contracts), _BATCH_SIZE):
            batch = contracts[i : i + _BATCH_SIZE]
            tickers = [ib.reqMktData(c, "", False, False) for c in batch]
            ib.sleep(_SLEEP_SECS)

            for c, ticker in zip(batch, tickers):
                try:
                    bid    = _f(ticker.bid)
                    ask    = _f(ticker.ask)
                    last   = _f(ticker.last)
                    close  = _f(ticker.close)
                    volume = _fvol(ticker.volume)

                    ib.cancelMktData(c)

                    if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > bid:
                        mid = (bid + ask) / 2.0
                        if (ask - bid) / mid > 1.0:   # spread > 100% → illiquide
                            continue
                        ref_type = "bid_ask"
                    elif math.isfinite(last) and last > 0:
                        mid = last
                        ref_type = "last"
                    elif math.isfinite(close) and close > 0:
                        mid = close
                        ref_type = "close"
                    else:
                        continue

                    raw_rows.append({
                        "Symbol":   symbol, "Spot":  spot,
                        "Strike":   float(c.strike),
                        "Maturity": _expiry_to_iso(expiry), "T": T,
                        "Type":     c.right,
                        "Bid":      bid,    "Ask":   ask,
                        "Mid":      mid,    "Volume": volume,
                        "RefType":  ref_type,
                    })
                except Exception as exc:
                    logger.debug("[%s] Ticker %s : %s", symbol, c, exc)

    if not raw_rows:
        logger.warning("[%s] Aucun prix collecté.", symbol)
        return pd.DataFrame()

    # ── 4. IV + grecs (tous les strikes avec prix valide) ────────────────
    records: list[dict] = []
    for row in raw_rows:
        iv = implied_vol(row["Spot"], row["Strike"], row["T"], r, q, row["Mid"], row["Type"])
        if iv is None:
            continue

        g = all_greeks(
            row["Spot"], row["Strike"], row["T"], r, q, iv,
            right=row["Type"], multiplier=multiplier,
        )

        records.append({
            "Symbol":      row["Symbol"],
            "Spot":        row["Spot"],
            "Strike":      row["Strike"],
            "Maturity":    row["Maturity"],
            "T":           row["T"],
            "Type":        row["Type"],
            "Bid":         row["Bid"],
            "Ask":         row["Ask"],
            "Mid":         row["Mid"],
            "Volume":      row["Volume"],
            "RefType":     row["RefType"],
            "ImpliedVol":  iv,
            "Delta":       g["Delta"],
            "Gamma":       g["Gamma"],
            "Vega":        g["Vega"],
            "Theta":       g["Theta"],
            "Rho":         g["Rho"],
            "DollarDelta": g["Delta"] * row["Spot"] * multiplier,
            "DollarGamma": g["DollarGamma"],
            "DollarVega":  g["DollarVega"],
            "DollarTheta": g["Theta"] * multiplier,
        })

    if not records:
        logger.warning("[%s] Aucune IV calculable.", symbol)
        return pd.DataFrame()

    # ── 5. Forward par parité put-call ────────────────────────────────────
    forwards = _compute_forwards(records, r)
    for rec in records:
        rec["Forward"] = forwards.get(rec["Maturity"], rec["Spot"])

    result = pd.DataFrame(records)
    logger.info("[%s] %d lignes collectées (IV valide).", symbol, len(result))
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
