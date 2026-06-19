"""
Extraction des données de marché via Interactive Brokers (ib_insync).
Univers : Euro Stoxx 50 (Index/EUREX + Actions/SMART/EUR).
Une seule connexion réutilisée pour tous les symboles.
Aucun print : logging uniquement.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
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
    Supprime les codes IBKR 10090 et 10167 signalant les données différées.
    """
    _CODES = {"10090", "10167"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(
            f"Error {c}," in msg or f"Warning {c}," in msg
            for c in self._CODES
        )

def _install_ibkr_log_filter() -> None:
    _filter = _SuppressDelayedDataFallback()
    for name in ("ib_insync.wrapper", "ib_insync.ib", "ib_insync"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _SuppressDelayedDataFallback) for f in lg.filters):
            lg.addFilter(_filter)

# ── constantes de configuration ───────────────────────────────────────────────
_TARGET_MATURITIES_DAYS = (14, 30, 91, 182, 273, 365, 548, 730, 1095)
_MATURITY_TOL_DAYS = 45
_BATCH_SIZE = 50      
_SLEEP_SECS = 2.0     

_MARKET_DATA_TYPE = 3
_CET = ZoneInfo("Europe/Paris")

def _in_eu_hours() -> bool:
    """True si on est dans les heures de trading EUREX (lun-ven, 09h00-17h30 CET)."""
    now = datetime.now(tz=_CET)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 <= t <= 17 * 60 + 30

# ── helpers de conversion ─────────────────────────────────────────────────────

def _expiry_to_T(expiry_str: str) -> float:
    mat_dt = datetime.strptime(expiry_str, "%Y%m%d")
    days = (mat_dt - datetime.now()).days
    return max(days / SETTINGS.T_BASIS, 0.0)

def _expiry_to_iso(expiry_str: str) -> str:
    return f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"

def _expiry_days(expiry_str: str) -> int:
    return (datetime.strptime(expiry_str, "%Y%m%d") - datetime.now()).days

def _select_expiries(expirations: list[str]) -> list[str]:
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

# ── construction dynamique des contrats Euro Stoxx 50 ─────────────────────────

def _get_universe_info(symbol: str) -> dict | None:
    """Récupère les infos du symbole depuis le fichier de référence."""
    if not SETTINGS.tickers_file.exists():
        logger.error("Fichier univers introuvable : %s", SETTINGS.tickers_file)
        return None
    df = pd.read_excel(SETTINGS.tickers_file)
    row = df[df["Ticker"] == symbol]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def _resolve_underlying(ib, symbol: str):
    """Résout le contrat en lisant l'Excel de référence."""
    from ib_insync import Index, Stock

    info = _get_universe_info(symbol)
    if not info:
        logger.warning("[%s] Symbole non trouvé dans l'univers.", symbol)
        return None

    ibkr_sym = info["IBKR_Symbol"]
    exch = info["PrimaryExchange"]
    curr = info["Currency"]

    # Traitement spécifique pour l'indice Euro Stoxx 50
    if symbol == "SX5E":
        # On tente la configuration standard ESTX50 sur DTB (nom natif IBKR pour l'indice)
        for sym_try, exch_try in [("ESTX50", "DTB"), ("SX5E", "EUREX"), ("ESTX50", "EUREX")]:
            try:
                contract = Index(sym_try, exch_try, curr)
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    logger.info("[%s] Contrat d'indice résolu sous le symbole IBKR: %s sur %s", symbol, sym_try, exch_try)
                    return qualified[0]
            except Exception:
                continue
    else:
        # Pour les actions EU
        contract = Stock(ibkr_sym, "SMART", curr, primaryExchange=exch)
        try:
            qualified = ib.qualifyContracts(contract)
            if qualified:
                return qualified[0]
        except Exception:
            pass

        # Fallback direct sur l'exchange pour l'action
        try:
            contract = Stock(ibkr_sym, exch, curr)
            qualified = ib.qualifyContracts(contract)
            if qualified:
                return qualified[0]
        except Exception:
            pass

    logger.warning("[%s] Contrat introuvable après essais multiples.", symbol)
    return None

def _select_option_param(params: list, symbol: str, info: dict):
    """Choisit la bonne chaîne d'options."""
    if symbol == "SX5E":
        oesx = [p for p in params if p.exchange == "EUREX" and p.tradingClass == "OESX"]
        if oesx: return oesx[0]
        eurex = [p for p in params if p.exchange == "EUREX"]
        if eurex: return eurex[0]

    # Pour les actions européennes
    target_exch = info["PrimaryExchange"]
    for exch in ["EUREX", target_exch, "SMART"]:
        match = [p for p in params if p.exchange == exch]
        if match:
            return match[0]

    return params[0]

def _valid_contracts_for_expiry(
    ib, symbol: str, expiry: str, exchange: str, trading_class: str, currency: str, info: dict
) -> list:
    """Récupère les contrats d'options qualifiés."""
    from ib_insync import Option
    ibkr_sym = info["IBKR_Symbol"]

    probe = Option(
        ibkr_sym,
        lastTradeDateOrContractMonth=expiry,
        exchange=exchange,
        currency=currency,
    )
    if symbol == "SX5E":
        probe.multiplier = str(SETTINGS.MULTIPLIER) # 10 pour OESX
        
    if trading_class:
        probe.tradingClass = trading_class

    try:
        details = ib.reqContractDetails(probe)
        return [d.contract for d in details]
    except Exception as exc:
        logger.warning("[%s %s] reqContractDetails échoué : %s", symbol, expiry, exc)
        return []

# ── calcul du forward par parité put-call ────────────────────────────────────

def _compute_forwards(records: list[dict], r: float) -> dict[str, float]:
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
    r = r if r is not None else SETTINGS.r
    q = q if q is not None else SETTINGS.q

    _install_ibkr_log_filter()
    ib.reqMarketDataType(_MARKET_DATA_TYPE)

    info = _get_universe_info(symbol)
    if not info:
        return pd.DataFrame()

    # ── 1. spot ───────────────────────────────────────────────────────────
    try:
        underlying = _resolve_underlying(ib, symbol)
        if underlying is None:
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
            logger.warning("[%s] Pas d'historique de spot.", symbol)
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

    param = _select_option_param(params, symbol, info)
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

    if _in_eu_hours():
        logger.info("[%s] Marché EU ouvert — bid/ask live.", symbol)
    else:
        logger.info("[%s] Hors heures EU — fallback sur close du jour.", symbol)

    def _f(x) -> float:
        return float(x) if x is not None and x == x and x > 0 else math.nan
    def _fvol(x) -> float:
        if x is None or x != x: return math.nan
        v = float(x)
        return v if v >= 0 else math.nan

    for expiry in expiries:
        T = _expiry_to_T(expiry)
        if T <= 0:
            continue

        contracts = _valid_contracts_for_expiry(
            ib, symbol, expiry,
            exchange=param.exchange,
            trading_class=param.tradingClass,
            currency=underlying.currency,
            info=info
        )
        if not contracts:
            continue

        logger.info("[%s %s] %d contrats.", symbol, expiry, len(contracts))

        for i in range(0, len(contracts), _BATCH_SIZE):
            batch = contracts[i : i + _BATCH_SIZE]
            tickers = [ib.reqMktData(c, "", False, False) for c in batch]
            ib.sleep(_SLEEP_SECS)

            for c, ticker in zip(batch, tickers):
                try:
                    bid, ask, last, close = _f(ticker.bid), _f(ticker.ask), _f(ticker.last), _f(ticker.close)
                    volume = _fvol(ticker.volume)
                    ib.cancelMktData(c)

                    if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > bid:
                        mid = (bid + ask) / 2.0
                        if (ask - bid) / mid > 1.0:
                            continue
                        ref_type = "bid_ask"
                    elif math.isfinite(last) and last > 0:
                        mid = last; ref_type = "last"
                    elif math.isfinite(close) and close > 0:
                        mid = close; ref_type = "close"
                    else:
                        continue
                    
                    # Récupération dynamique du multiplicateur pour les actions EU
                    c_mult = float(c.multiplier) if c.multiplier else float(SETTINGS.MULTIPLIER)

                    raw_rows.append({
                        "Symbol": symbol, "Spot": spot, "Strike": float(c.strike),
                        "Maturity": _expiry_to_iso(expiry), "T": T, "Type": c.right,
                        "Bid": bid, "Ask": ask, "Mid": mid, "Volume": volume,
                        "RefType": ref_type, "ContractMultiplier": c_mult
                    })
                except Exception as exc:
                    logger.debug("[%s] Ticker erreur : %s", symbol, exc)

    if not raw_rows:
        logger.warning("[%s] Aucun prix collecté.", symbol)
        return pd.DataFrame()

    # ── 4. IV + grecs ─────────────────────────────────────────────────────
    records: list[dict] = []
    for row in raw_rows:
        iv = implied_vol(row["Spot"], row["Strike"], row["T"], r, q, row["Mid"], row["Type"])
        if iv is None:
            continue

        c_mult = row["ContractMultiplier"]
        g = all_greeks(
            row["Spot"], row["Strike"], row["T"], r, q, iv,
            right=row["Type"], multiplier=c_mult,
        )

        records.append({
            "Symbol": row["Symbol"], "Spot": row["Spot"], "Strike": row["Strike"],
            "Maturity": row["Maturity"], "T": row["T"], "Type": row["Type"],
            "Bid": row["Bid"], "Ask": row["Ask"], "Mid": row["Mid"],
            "Volume": row["Volume"], "RefType": row["RefType"],
            "ImpliedVol": iv,
            "Delta": g["Delta"], "Gamma": g["Gamma"],
            "Vega": g["Vega"], "Theta": g["Theta"], "Rho": g["Rho"],
            "DollarDelta": g["Delta"] * row["Spot"] * c_mult,
            "DollarGamma": g["DollarGamma"],
            "DollarVega": g["DollarVega"],
            "DollarTheta": g["Theta"] * c_mult,
        })

    if not records:
        logger.warning("[%s] Aucune IV calculable.", symbol)
        return pd.DataFrame()

    forwards = _compute_forwards(records, r)
    for rec in records:
        rec["Forward"] = forwards.get(rec["Maturity"], rec["Spot"])

    result = pd.DataFrame(records)
    logger.info("[%s] %d lignes collectées (IV valide).", symbol, len(result))
    return result

# ── persistence ───────────────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or SETTINGS.parquet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"snapshot_{ts}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Snapshot sauvegardé : %s (%d lignes)", path, len(df))
    return path

def load_latest_snapshot(parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    parquet_dir = parquet_dir or SETTINGS.parquet_dir
    files = sorted(parquet_dir.glob("snapshot_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])

def list_snapshots(parquet_dir: Optional[Path] = None) -> list[Path]:
    parquet_dir = parquet_dir or SETTINGS.parquet_dir
    return sorted(parquet_dir.glob("snapshot_*.parquet"), reverse=True)