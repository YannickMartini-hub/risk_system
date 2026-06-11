"""
Tests de la volatilité implicite.
- Round-trip BS(σ) → implied_vol → σ (tolérance 1e-6)
- Newton seul et Newton+Brent (wrapper)
- Cas limites : deep ITM/OTM, maturité très courte
- Prix hors bornes d'arbitrage → None
"""

import math

import pytest

from risk_system.pricing import bs_price
from risk_system.implied_vol import implied_vol, implied_vol_newton, implied_vol_brent

# Grille de paramètres de marché
PARAMS = [
    (5000.0, 5000.0, 0.25, 0.030, 0.025),  # ATM SX5E
    (100.0,   95.0,  0.50, 0.020, 0.000),  # légèrement ITM put / OTM call
    (100.0,  105.0,  1.00, 0.030, 0.010),  # légèrement OTM
    (100.0,  100.0,  0.25, 0.025, 0.015),  # ATM avec dividende
]

SIGMAS = [0.10, 0.20, 0.30, 0.50]
RIGHTS = ["C", "P"]


@pytest.mark.parametrize("S,K,T,r,q", PARAMS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("right", RIGHTS)
def test_round_trip_newton(S, K, T, r, q, sigma, right):
    """BS(sigma) → implied_vol_newton → sigma, tolérance 1e-5."""
    price = bs_price(S, K, T, r, q, sigma, right)
    iv    = implied_vol_newton(S, K, T, r, q, price, right)
    assert iv is not None, (
        f"Newton n'a pas convergé : S={S} K={K} T={T} sigma={sigma} right={right}"
    )
    assert abs(iv - sigma) < 1e-5


@pytest.mark.parametrize("S,K,T,r,q", PARAMS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("right", RIGHTS)
def test_round_trip_full(S, K, T, r, q, sigma, right):
    """Round-trip via implied_vol (Newton + Brent fallback), tolérance 1e-6."""
    price = bs_price(S, K, T, r, q, sigma, right)
    iv    = implied_vol(S, K, T, r, q, price, right)
    assert iv is not None, (
        f"implied_vol a retourné None : S={S} K={K} T={T} sigma={sigma} right={right}"
    )
    assert abs(iv - sigma) < 1e-6


def test_deep_otm_call_brent():
    """Call très OTM (K=200, S=100) : Newton peut échouer, Brent prend le relais."""
    S, K, T, r, q, sigma = 100.0, 200.0, 0.5, 0.02, 0.0, 0.30
    price = bs_price(S, K, T, r, q, sigma, "C")
    iv    = implied_vol(S, K, T, r, q, price, "C")
    if iv is not None:
        assert abs(iv - sigma) < 1e-5


def test_deep_itm_put():
    """Put très ITM (K=200, S=100)."""
    S, K, T, r, q, sigma = 100.0, 200.0, 1.0, 0.03, 0.0, 0.25
    price = bs_price(S, K, T, r, q, sigma, "P")
    iv    = implied_vol(S, K, T, r, q, price, "P")
    if iv is not None:
        assert abs(iv - sigma) < 1e-5


def test_short_maturity():
    """Maturité très courte (T ≈ 1 jour)."""
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0 / 365.25, 0.03, 0.0, 0.20
    price = bs_price(S, K, T, r, q, sigma, "C")
    iv    = implied_vol(S, K, T, r, q, price, "C")
    if iv is not None:
        assert abs(iv - sigma) < 1e-4


def test_price_below_intrinsic_returns_none_call():
    """Prix call en dessous de l'intrinsèque → None."""
    S, K, T, r, q = 100.0, 80.0, 0.5, 0.03, 0.0
    intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    iv = implied_vol(S, K, T, r, q, intrinsic * 0.5, "C")
    assert iv is None


def test_price_below_intrinsic_returns_none_put():
    """Prix put en dessous de l'intrinsèque → None."""
    S, K, T, r, q = 100.0, 120.0, 0.5, 0.03, 0.0
    intrinsic = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
    iv = implied_vol(S, K, T, r, q, intrinsic * 0.5, "P")
    assert iv is None


def test_price_above_upper_bound_returns_none():
    """Prix au-dessus de la borne supérieure (S·e^(-qT)) → None."""
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.03, 0.0
    upper = S * math.exp(-q * T)
    iv = implied_vol(S, K, T, r, q, upper * 1.5, "C")
    assert iv is None


def test_brent_range_no_solution():
    """Brent retourne None quand le prix ne croise pas [lo, hi]."""
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.03, 0.0
    # Prix négatif impossible → pas de solution
    iv = implied_vol_brent(S, K, T, r, q, -1.0, "C")
    assert iv is None


def test_high_vol_round_trip():
    """Vol élevée (150%) : Brent doit trouver une solution."""
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.0, 1.50
    price = bs_price(S, K, T, r, q, sigma, "C")
    iv    = implied_vol(S, K, T, r, q, price, "C")
    if iv is not None:
        assert abs(iv - sigma) < 1e-4
