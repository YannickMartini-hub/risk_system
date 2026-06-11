"""
Inversion de la volatilité implicite.
Méthodes : Newton-Raphson (primaire) + Brent via scipy (fallback).
"""

from __future__ import annotations

import math
import logging
from typing import Optional

from scipy.optimize import brentq

from .pricing import bs_price, compute_d1, norm_pdf

logger = logging.getLogger(__name__)


# ── utilitaires internes ──────────────────────────────────────────────────────

def _vega_raw(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega brut dV/dσ (non divisé par 100) — dérivée pour Newton-Raphson."""
    d1 = compute_d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)


def _check_arbitrage_bounds(
    market_price: float, S: float, K: float, T: float,
    r: float, q: float, right: str,
) -> bool:
    """
    Vérifie que market_price respecte les bornes d'arbitrage sans vol.
    Retourne True si le prix est inversible.
    """
    if right == "C":
        intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
        upper = S * math.exp(-q * T)
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
        upper = K * math.exp(-r * T)

    return intrinsic <= market_price <= upper


# ── méthodes d'inversion ─────────────────────────────────────────────────────

def implied_vol_newton(
    S: float, K: float, T: float, r: float, q: float,
    market_price: float, right: str,
    sigma0: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Volatilité implicite par Newton-Raphson.

    Paramètres
    ----------
    sigma0   : vol initiale (50%)
    tol      : tolérance sur l'écart |BS(σ) - market_price|
    max_iter : nombre maximal d'itérations
    bornes   : σ ∈ [0.01, 2.0]

    Retourne sigma ou None si pas de convergence.
    """
    sigma = sigma0

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, q, sigma, right)
        diff  = price - market_price

        if abs(diff) < tol:
            return sigma

        vg = _vega_raw(S, K, T, r, q, sigma)
        if abs(vg) < 1e-12:
            return None

        sigma -= diff / vg
        sigma  = max(0.01, min(sigma, 2.0))

    return None


def implied_vol_brent(
    S: float, K: float, T: float, r: float, q: float,
    market_price: float, right: str,
    lo: float = 1e-4,
    hi: float = 3.0,
) -> Optional[float]:
    """
    Volatilité implicite par méthode de Brent (scipy.optimize.brentq).
    Fallback robuste pour options très OTM ou vega quasi nul.

    Retourne sigma ou None si pas de solution dans [lo, hi].
    """
    try:
        f_lo = bs_price(S, K, T, r, q, lo, right) - market_price
        f_hi = bs_price(S, K, T, r, q, hi, right) - market_price

        if f_lo * f_hi > 0:
            return None

        sigma = brentq(
            lambda s: bs_price(S, K, T, r, q, s, right) - market_price,
            lo, hi, xtol=1e-8, rtol=1e-8, maxiter=200,
        )
        return float(sigma)
    except (ValueError, RuntimeError):
        return None


def implied_vol(
    S: float, K: float, T: float, r: float, q: float,
    market_price: float, right: str,
) -> Optional[float]:
    """
    Volatilité implicite : Newton-Raphson puis Brent en fallback.

    Retourne None proprement si :
      - le prix viole les bornes d'arbitrage (< intrinsèque ou > borne supérieure)
      - aucune solution dans [1e-4, 3.0]
    """
    if not _check_arbitrage_bounds(market_price, S, K, T, r, q, right):
        logger.debug(
            "Prix hors bornes : S=%.2f K=%.2f T=%.4f %s price=%.4f",
            S, K, T, right, market_price,
        )
        return None

    iv = implied_vol_newton(S, K, T, r, q, market_price, right)
    if iv is not None:
        return iv

    logger.debug("Newton échoué, passage à Brent : S=%.2f K=%.2f T=%.4f %s", S, K, T, right)
    return implied_vol_brent(S, K, T, r, q, market_price, right)
