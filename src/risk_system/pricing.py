"""
Pricing Black-Scholes pour options européennes.
Formules de référence : base.ipynb / roadmap Eq. 3, 6–11.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr


# ── fonctions de distribution ────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """
    CDF de la loi normale standard N(x).
    Implémentée via math.erf (stdlib Python, sans dépendance scipy).
    N(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """
    PDF de la loi normale standard n(x).
    n(x) = exp(-x²/2) / sqrt(2π)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Version vectorisée de norm_cdf via scipy.special.ndtr."""
    return ndtr(np.asarray(x, dtype=float))


def norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Version vectorisée de norm_pdf."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ── briques de base ──────────────────────────────────────────────────────────

def forward(S: float, r: float, q: float, T: float) -> float:
    """
    Forward carry-based (Eq. 3 du roadmap).
    F(T) = S * exp((r - q) * T)

    Paramètres
    ----------
    S : spot de référence (mid bid/ask)
    r : taux sans risque continu
    q : dividend yield ou carry implicite
    T : maturité en fractions d'année
    """
    return S * math.exp((r - q) * T)


def log_moneyness(K: float, F: float) -> float:
    """
    Log-moneyness relatif au forward (Eq. 6).
    k = ln(K / F)

    k < 0 : OTM put / ITM call
    k = 0 : ATM forward
    k > 0 : OTM call / ITM put
    """
    return math.log(K / F)


def total_variance(sigma: float, T: float) -> float:
    """
    Variance totale (Eq. 7).
    w = sigma² * T
    """
    return sigma ** 2 * T


def compute_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    d1 (Eq. 8) : [ln(S/K) + (r - q + σ²/2) * T] / (σ * √T)
    """
    sqrt_T = math.sqrt(T)
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)


def compute_d2(d1: float, sigma: float, T: float) -> float:
    """
    d2 (Eq. 9) : d1 - σ * √T

    Note : T passé explicitement (le notebook base.ipynb référençait T depuis
    la portée globale, ce qui est un bug dans un contexte modulaire).
    """
    return d1 - sigma * math.sqrt(T)


# ── prix BS scalaires ────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Prix call européen Black-Scholes (Eq. 10).
    C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    d2 = compute_d2(d1, sigma, T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Prix put européen Black-Scholes (Eq. 11).
    P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    """
    d1 = compute_d1(S, K, T, r, q, sigma)
    d2 = compute_d2(d1, sigma, T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """
    Prix Black-Scholes pour call ('C') ou put ('P').
    """
    if right == "C":
        return bs_call(S, K, T, r, q, sigma)
    elif right == "P":
        return bs_put(S, K, T, r, q, sigma)
    else:
        raise ValueError(f"right doit être 'C' ou 'P', reçu : {right!r}")


# ── versions vectorisées numpy ───────────────────────────────────────────────

def bs_call_vec(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Version vectorisée de bs_call (numpy)."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * np.exp(-q * T) * norm_cdf_vec(d1) - K * np.exp(-r * T) * norm_cdf_vec(d2)


def bs_put_vec(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Version vectorisée de bs_put (numpy)."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * np.exp(-r * T) * norm_cdf_vec(-d2) - S * np.exp(-q * T) * norm_cdf_vec(-d1)
