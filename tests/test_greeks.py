"""
Tests des grecs : analytiques vs différences finies centrées (tolérance 1e-4).
"""

import math

import pytest

from risk_system.pricing import bs_price
from risk_system.greeks import delta, gamma, vega, theta, rho, all_greeks

# Paramètres de test (S, K, T, r, q, sigma, right)
CASES = [
    (5000.0, 5000.0, 0.25, 0.030, 0.025, 0.18, "C"),  # ATM call SX5E
    (5000.0, 5000.0, 0.25, 0.030, 0.025, 0.18, "P"),  # ATM put SX5E
    (100.0,   90.0,  0.50, 0.020, 0.000, 0.20, "P"),  # ITM put
    (100.0,  110.0,  1.00, 0.040, 0.010, 0.30, "C"),  # OTM call long terme
    (100.0,  100.0,  0.25, 0.030, 0.015, 0.25, "C"),  # ATM call avec q
]

FD_TOL = 1e-4


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_delta_vs_fd(S, K, T, r, q, sigma, right):
    """Delta analytique ≈ différence finie centrée sur S."""
    h  = S * 1e-4
    fd = (bs_price(S + h, K, T, r, q, sigma, right) - bs_price(S - h, K, T, r, q, sigma, right)) / (2 * h)
    assert abs(delta(S, K, T, r, q, sigma, right) - fd) < FD_TOL


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_gamma_vs_fd(S, K, T, r, q, sigma, right):
    """Gamma analytique ≈ différence finie du second ordre sur S."""
    h  = S * 1e-3
    fd = (
        bs_price(S + h, K, T, r, q, sigma, right)
        - 2.0 * bs_price(S, K, T, r, q, sigma, right)
        + bs_price(S - h, K, T, r, q, sigma, right)
    ) / h ** 2
    assert abs(gamma(S, K, T, r, q, sigma) - fd) < FD_TOL


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_vega_vs_fd(S, K, T, r, q, sigma, right):
    """Vega (par 1%) analytique ≈ différence finie centrée sur σ divisée par 100."""
    h       = 1e-4
    fd_raw  = (bs_price(S, K, T, r, q, sigma + h, right) - bs_price(S, K, T, r, q, sigma - h, right)) / (2 * h)
    fd_1pct = fd_raw / 100.0  # convention : par 1% de vol
    assert abs(vega(S, K, T, r, q, sigma) - fd_1pct) < FD_TOL


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_theta_vs_fd(S, K, T, r, q, sigma, right):
    """Theta (par jour) analytique ≈ −(dV/dT) / 365 en différences finies."""
    h  = 1e-4
    # theta = -dV/dT / 365 ; FD centrée sur T
    fd = -(bs_price(S, K, T + h, r, q, sigma, right) - bs_price(S, K, T - h, r, q, sigma, right)) / (2 * h * 365.0)
    assert abs(theta(S, K, T, r, q, sigma, right) - fd) < 5 * FD_TOL  # tolérance élargie (non-linéarités)


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_rho_vs_fd(S, K, T, r, q, sigma, right):
    """Rho (par 1%) analytique ≈ différence finie centrée sur r divisée par 100."""
    h       = 1e-4
    fd_raw  = (bs_price(S, K, T, r + h, q, sigma, right) - bs_price(S, K, T, r - h, q, sigma, right)) / (2 * h)
    fd_1pct = fd_raw / 100.0
    assert abs(rho(S, K, T, r, q, sigma, right) - fd_1pct) < FD_TOL


@pytest.mark.parametrize("S,K,T,r,q,sigma,right", CASES)
def test_all_greeks_consistency(S, K, T, r, q, sigma, right):
    """all_greeks() doit être cohérent avec les fonctions individuelles."""
    g = all_greeks(S, K, T, r, q, sigma, right)
    assert g["Delta"] == pytest.approx(delta(S, K, T, r, q, sigma, right), rel=1e-6)
    assert g["Gamma"] == pytest.approx(gamma(S, K, T, r, q, sigma),        rel=1e-6)
    assert g["Vega"]  == pytest.approx(vega(S, K, T, r, q, sigma),         rel=1e-6)
    assert g["Theta"] == pytest.approx(theta(S, K, T, r, q, sigma, right), rel=1e-6)
    assert g["Rho"]   == pytest.approx(rho(S, K, T, r, q, sigma, right),   rel=1e-6)


def test_dollar_gamma_sx5e():
    """DollarGamma avec multiplier=10 vaut bien 10 × Γ × S²."""
    S, K, T, r, q, sigma = 5000.0, 5000.0, 0.25, 0.03, 0.0, 0.18
    g    = all_greeks(S, K, T, r, q, sigma, "C", multiplier=10)
    from risk_system.greeks import gamma as gamma_fn
    g_scalar = gamma_fn(S, K, T, r, q, sigma)
    assert g["DollarGamma"] == pytest.approx(g_scalar * S ** 2 * 10, rel=1e-8)


def test_vega_per_pct_convention():
    """Vega × 100 = raw vega (dV/dσ)."""
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.03, 0.0, 0.20
    h       = 1e-5
    dv_dsig = (bs_price(S, K, T, r, q, sigma + h, "C") - bs_price(S, K, T, r, q, sigma - h, "C")) / (2 * h)
    v       = vega(S, K, T, r, q, sigma)
    assert abs(v * 100 - dv_dsig) < 1e-6
