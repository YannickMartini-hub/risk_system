"""
Contrôle qualité des snapshots d'options.
Chaque check retourne un DataFrame des lignes / groupes problématiques.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def check_spread_pct(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """
    Signale les options où le spread bid-ask dépasse threshold * mid.
    Threshold par défaut : 50 % (options très illiquides).
    """
    mask = (
        df["Bid"].notna() & df["Ask"].notna() &
        df["Mid"].notna() & (df["Mid"] > 0) &
        (df["Ask"] > df["Bid"])
    )
    sub = df[mask].copy()
    sub["SpreadPct"] = (sub["Ask"] - sub["Bid"]) / sub["Mid"]
    flagged = sub[sub["SpreadPct"] > threshold].copy()
    logger.info("check_spread_pct: %d lignes flaggées (> %.0f%%)", len(flagged), threshold * 100)
    return flagged.reset_index(drop=True)


def check_staleness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signale les options dont le prix provient d'un close IBKR (RefType="close").
    Ces prix peuvent être décalés de plusieurs heures.
    """
    if "RefType" not in df.columns:
        return pd.DataFrame()
    flagged = df[df["RefType"] == "close"].copy()
    logger.info("check_staleness: %d lignes sur close (potentiellement obsolètes)", len(flagged))
    return flagged.reset_index(drop=True)


def check_chain_coverage(df: pd.DataFrame, min_strikes: int = 5) -> pd.DataFrame:
    """
    Signale les (Symbol, Maturity, Type) ayant moins de min_strikes strikes.
    Une chaîne trop courte ne suffit pas pour construire une surface fiable.
    """
    counts = (
        df.groupby(["Symbol", "Maturity", "Type"])
        .size()
        .reset_index(name="NStrikes")
    )
    flagged = counts[counts["NStrikes"] < min_strikes].copy()
    logger.info("check_chain_coverage: %d tranches avec < %d strikes", len(flagged), min_strikes)
    return flagged.reset_index(drop=True)


def check_put_call_parity(df: pd.DataFrame, r: float = 0.045, threshold: float = 0.05) -> pd.DataFrame:
    """
    Signale les maturités où le forward implicite (colonne Forward) s'écarte
    de plus de threshold * Spot par rapport au spot.
    Indique une violation de la parité put-call ou un problème de liquidité.
    """
    if "Forward" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby(["Symbol", "Maturity"])
        .agg(Spot=("Spot", "first"), Forward=("Forward", "first"))
        .reset_index()
    )
    summary["FwdSpotRatio"] = (summary["Forward"] - summary["Spot"]).abs() / summary["Spot"]
    flagged = summary[summary["FwdSpotRatio"] > threshold].copy()
    logger.info(
        "check_put_call_parity: %d maturités avec |F/S-1| > %.0f%%",
        len(flagged), threshold * 100,
    )
    return flagged.reset_index(drop=True)


def check_calendar_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signale les (Symbol, Strike, Type) où la vol implicite décroît avec la
    maturité (arbitrage de spread calendaire).
    Seules les paires de maturités consécutives sont vérifiées.
    """
    if df.empty or "ImpliedVol" not in df.columns:
        return pd.DataFrame()

    violations: list[dict] = []

    for (sym, strike, right), grp in df.groupby(["Symbol", "Strike", "Type"]):
        grp_sorted = grp.dropna(subset=["ImpliedVol", "T"]).sort_values("T")
        if len(grp_sorted) < 2:
            continue
        t_vals  = grp_sorted["T"].values
        iv_vals = grp_sorted["ImpliedVol"].values
        for i in range(len(t_vals) - 1):
            if iv_vals[i + 1] < iv_vals[i] - 1e-4:   # tolérance numérique 0.01%
                violations.append({
                    "Symbol":     sym,
                    "Strike":     strike,
                    "Type":       right,
                    "T_short":    t_vals[i],
                    "IV_short":   iv_vals[i],
                    "T_long":     t_vals[i + 1],
                    "IV_long":    iv_vals[i + 1],
                })

    flagged = pd.DataFrame(violations)
    logger.info("check_calendar_spread: %d violations calendaire détectées", len(flagged))
    return flagged.reset_index(drop=True)


def run_all_checks(df: pd.DataFrame, r: float = 0.045) -> dict[str, pd.DataFrame]:
    """
    Lance tous les checks QC sur un snapshot et retourne un dictionnaire
    check_name → DataFrame de lignes flaggées.
    """
    if df.empty:
        logger.warning("run_all_checks: snapshot vide.")
        return {}

    results = {
        "spread_pct":       check_spread_pct(df),
        "staleness":        check_staleness(df),
        "chain_coverage":   check_chain_coverage(df),
        "put_call_parity":  check_put_call_parity(df, r=r),
        "calendar_spread":  check_calendar_spread(df),
    }

    total_flags = sum(len(v) for v in results.values())
    logger.info("QC terminé : %d flags au total sur %d lignes.", total_flags, len(df))
    return results
