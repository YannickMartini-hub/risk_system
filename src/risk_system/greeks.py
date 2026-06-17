"""
Grecs Black-Scholes pour options européennes.
Formules de référence : base.ipynb / roadmap Eq. 13–19.

Conventions :
  - vega  : par 1% de vol (Δσ = 0.01)
  - theta : par jour calendaire (/365)
  - rho   : par 1% de taux (Δr = 0.01)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from .pricing import norm_cdf, norm_pdf, compute_d1, compute_d2

logger = logging.getLogger(__name__)


# ── grecs scalaires ──────────────────────────────────────────────────────────

def delta(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """
    Delta (Eq. 13) : première dérivée du prix par rapport au spot.
    Call :  e^(-qT) * N(d1)
    Put  : -e^(-qT) * N(-d1)
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    if right == "C":
        return math.exp(-q * T) * norm_cdf(d1)
    else:
        return -math.exp(-q * T) * norm_cdf(-d1)


def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Gamma (Eq. 14) : deuxième dérivée par rapport au spot.
    Γ = e^(-qT) * n(d1) / (S * σ * √T)
    Identique pour call et put.
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Vega (Eq. 15) : sensibilité à la volatilité par 1% de vol.
    V = S * e^(-qT) * n(d1) * √T / 100

    Identique pour call et put.
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    vega_raw = S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)
    return vega_raw / 100.0


def theta(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """
    Theta (Eq. 16) : décroissance temporelle par jour calendaire (/365).

    Call : -(S*e^(-qT)*n(d1)*σ)/(2√T) - r*K*e^(-rT)*N(d2)  + q*S*e^(-qT)*N(d1)
    Put  : -(S*e^(-qT)*n(d1)*σ)/(2√T) + r*K*e^(-rT)*N(-d2) - q*S*e^(-qT)*N(-d1)
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    d2 = compute_d2(d1, sigma, T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    sqrt_T = math.sqrt(T)

    common = -S * disc_q * norm_pdf(d1) * sigma / (2.0 * sqrt_T)

    if right == "C":
        theta_annual = common - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
    else:
        theta_annual = common + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)

    return theta_annual / 365.0


def rho(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """
    Rho : sensibilité au taux sans risque par 1% de taux (Δr = 0.01).
    Call :  K * T * e^(-rT) * N(d2)  * 0.01
    Put  : -K * T * e^(-rT) * N(-d2) * 0.01
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    d2 = compute_d2(d1, sigma, T)
    disc_r = math.exp(-r * T)

    if right == "C":
        return K * T * disc_r * norm_cdf(d2) * 0.01
    else:
        return -K * T * disc_r * norm_cdf(-d2) * 0.01


def dollar_gamma(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    multiplier: float = 1.0,
) -> float:
    """
    Dollar Gamma (Eq. 17) : gamma monétisé.
    DollarGamma = Γ * S² * multiplier

    Pour SPX/US stocks : multiplier = 100.
    """
    return gamma(S, K, T, r, q, sigma) * S ** 2 * multiplier


def dollar_vega(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    multiplier: float = 1.0,
) -> float:
    """
    Dollar Vega (Eq. 18) : vega monétisé.
    DollarVega = Vega * multiplier
    """
    return vega(S, K, T, r, q, sigma) * multiplier


def pnl_attribution(
    S: float, K: float, T: float, r: float, q: float, sigma: float, right: str,
    dS: float, d_sigma: float, dt_days: float,
    S_new: Optional[float] = None,
    sigma_new: Optional[float] = None,
    T_new: Optional[float] = None,
) -> dict:
    """
    Attribution P&L locale (Eq. 19) :
    ΔV ≈ Δ·dS + ½·Γ·(dS)² + Vega·dσ + Θ·dt

    Paramètres
    ----------
    dS      : variation du spot
    d_sigma : variation de vol en absolu (ex: 0.01 = +1 vol point)
    dt_days : temps écoulé en jours calendaires

    S_new, sigma_new, T_new : si fournis, calcule le résidu vs full revalorisation BS.
    """
    from .pricing import bs_price

    d  = delta(S, K, T, r, q, sigma, right)
    g  = gamma(S, K, T, r, q, sigma)
    v  = vega(S, K, T, r, q, sigma)
    th = theta(S, K, T, r, q, sigma, right)

    delta_pnl = d * dS
    gamma_pnl = 0.5 * g * dS ** 2
    # vega est par 1% ; d_sigma en absolu → convertir en points de vol
    vega_pnl  = v * d_sigma * 100.0
    theta_pnl = th * dt_days
    total     = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

    result = {
        "delta_pnl":    round(delta_pnl, 4),
        "gamma_pnl":    round(gamma_pnl, 4),
        "vega_pnl":     round(vega_pnl,  4),
        "theta_pnl":    round(theta_pnl, 4),
        "total_approx": round(total, 4),
    }

    if S_new is not None and sigma_new is not None and T_new is not None:
        price_old = bs_price(S, K, T, r, q, sigma, right)
        price_new = bs_price(S_new, K, T_new, r, q, sigma_new, right)
        full_pnl  = price_new - price_old
        result["full_pnl"] = round(full_pnl, 4)
        result["residual"] = round(full_pnl - total, 4)

    return result


def all_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    right: str, multiplier: float = 1.0,
) -> dict:
    """
    Calcule tous les grecs en un appel.

    Retourne un dict :
      Delta, Gamma, Vega, Theta, Rho, DollarGamma, DollarVega.
    """
    return {
        "Delta":       delta(S, K, T, r, q, sigma, right),
        "Gamma":       gamma(S, K, T, r, q, sigma),
        "Vega":        vega(S, K, T, r, q, sigma),
        "Theta":       theta(S, K, T, r, q, sigma, right),
        "Rho":         rho(S, K, T, r, q, sigma, right),
        "DollarGamma": dollar_gamma(S, K, T, r, q, sigma, multiplier),
        "DollarVega":  dollar_vega(S, K, T, r, q, sigma, multiplier),
    }


def aggregate_greeks(
    positions: list[dict],
    snapshot: pd.DataFrame,
    multiplier: int = 100,
) -> dict:
    """
    Agrège les greeks dollar d'un portefeuille d'options depuis un snapshot.

    Paramètres
    ----------
    positions : liste de dicts, chacun avec les clés :
        symbol   : str  — ex. 'SPX'
        strike   : float — ex. 5500.0
        maturity : str  — ex. '2026-09-19' (format YYYY-MM-DD)
        type     : str  — 'C' ou 'P'
        quantity : int  — nombre de contrats (positif = long, négatif = short)
    snapshot  : DataFrame chargé depuis load_latest_snapshot()
    multiplier: non utilisé directement (les greeks dollar sont déjà dans le snapshot)

    Retourne
    --------
    dict {'DollarDelta', 'DollarGamma', 'DollarVega', 'DollarTheta'}
    Retourne des zéros si aucune position n'est trouvée dans le snapshot.
    """
    totals: dict[str, float] = {
        "DollarDelta": 0.0,
        "DollarGamma": 0.0,
        "DollarVega":  0.0,
        "DollarTheta": 0.0,
    }

    if snapshot.empty:
        logger.warning("aggregate_greeks: snapshot vide.")
        return totals

    for pos in positions:
        mask = (
            (snapshot["Symbol"]   == pos["symbol"]) &
            (snapshot["Strike"]   == float(pos["strike"])) &
            (snapshot["Maturity"] == pos["maturity"]) &
            (snapshot["Type"]     == pos["type"])
        )
        rows = snapshot[mask]
        if rows.empty:
            logger.warning("aggregate_greeks: position introuvable : %s", pos)
            continue
        row = rows.iloc[0]
        qty = int(pos["quantity"])
        for col in totals:
            totals[col] += float(row[col]) * qty

    return totals
