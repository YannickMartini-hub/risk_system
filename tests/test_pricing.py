"""
Tests du module pricing.
- Parité put-call sur une grille de paramètres
- Dispatcher bs_price
- Cas limites deep ITM/OTM
- Fonctions auxiliaires (forward, log_moneyness, total_variance)
"""

import math

import pytest

from risk_system.pricing import (
    bs_call, bs_put, bs_price,
    forward, log_moneyness, total_variance,
)

# Grille de paramètres (S, K, T, r, q, sigma)
GRID = [
    (5000.0, 5000.0, 0.25, 0.030, 0.025, 0.18),  # ATM SX5E
    (100.0,   95.0,  0.50, 0.020, 0.000, 0.20),  # légèrement OTM put
    (100.0,  110.0,  1.00, 0.040, 0.010, 0.30),  # OTM call
    ( 50.0,   50.0,  0.10, 0.010, 0.000, 0.15),  # ATM, court terme
    (200.0,  150.0,  2.00, 0.050, 0.020, 0.25),  # deep ITM call
]


@pytest.mark.parametrize("S,K,T,r,q,sigma", GRID)
def test_put_call_parity(S, K, T, r, q, sigma):
    """C - P = S·e^(-qT) - K·e^(-rT)."""
    C = bs_call(S, K, T, r, q, sigma)
    P = bs_put(S, K, T, r, q, sigma)
    lhs = C - P
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"Parité violée : LHS={lhs:.8f} RHS={rhs:.8f}"


@pytest.mark.parametrize("S,K,T,r,q,sigma", GRID)
def test_bs_price_dispatcher(S, K, T, r, q, sigma):
    """bs_price dispatche correctement vers bs_call / bs_put."""
    assert bs_price(S, K, T, r, q, sigma, "C") == pytest.approx(bs_call(S, K, T, r, q, sigma))
    assert bs_price(S, K, T, r, q, sigma, "P") == pytest.approx(bs_put(S, K, T, r, q, sigma))


def test_bs_price_invalid_right():
    with pytest.raises(ValueError, match="right"):
        bs_price(100, 100, 0.5, 0.02, 0.0, 0.20, "X")


def test_call_non_negative():
    """Un call ne peut pas être négatif."""
    for S, K, T, r, q, sigma in GRID:
        assert bs_call(S, K, T, r, q, sigma) >= 0.0


def test_put_non_negative():
    """Un put ne peut pas être négatif."""
    for S, K, T, r, q, sigma in GRID:
        assert bs_put(S, K, T, r, q, sigma) >= 0.0


def test_call_deep_itm_close_to_intrinsic():
    """Deep ITM call ≈ S·e^(-qT) - K·e^(-rT) (proche de l'intrinsèque)."""
    S, K, T, r, q, sigma = 200.0, 100.0, 1.0, 0.03, 0.0, 0.20
    C         = bs_call(S, K, T, r, q, sigma)
    intrinsic = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert C >= intrinsic - 1e-6
    assert abs(C - intrinsic) < 5.0


def test_put_deep_itm_close_to_intrinsic():
    """Deep ITM put ≈ K·e^(-rT) - S·e^(-qT)."""
    S, K, T, r, q, sigma = 50.0, 200.0, 1.0, 0.03, 0.0, 0.20
    P         = bs_put(S, K, T, r, q, sigma)
    intrinsic = K * math.exp(-r * T) - S * math.exp(-q * T)
    assert P >= intrinsic - 1e-6


def test_call_increases_with_spot():
    """Le call est croissant avec le spot."""
    K, T, r, q, sigma = 100.0, 0.5, 0.02, 0.0, 0.20
    prices = [bs_call(S, K, T, r, q, sigma) for S in [80, 90, 100, 110, 120]]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


def test_put_decreases_with_spot():
    """Le put est décroissant avec le spot."""
    K, T, r, q, sigma = 100.0, 0.5, 0.02, 0.0, 0.20
    prices = [bs_put(S, K, T, r, q, sigma) for S in [80, 90, 100, 110, 120]]
    assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def test_forward():
    F = forward(100.0, 0.05, 0.02, 1.0)
    assert F == pytest.approx(100.0 * math.exp(0.03), rel=1e-10)


def test_log_moneyness_atm():
    assert log_moneyness(100.0, 100.0) == pytest.approx(0.0, abs=1e-12)


def test_log_moneyness_otm():
    assert log_moneyness(105.0, 100.0) == pytest.approx(math.log(1.05), rel=1e-10)


def test_total_variance():
    assert total_variance(0.20, 0.5) == pytest.approx(0.02, rel=1e-10)
