"""
Construction de la nappe de volatilité implicite.
Interpolation en variance totale w = σ²T (meilleure régularité analytique).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def build_surface(
    df: pd.DataFrame,
    symbol: str,
    n_grid: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit la nappe de volatilité interpolée pour un symbole.

    Les points bruts sont d'abord transformés en variance totale w = σ²T,
    puis interpolés sur une grille régulière (log-moneyness, maturité)
    via scipy.interpolate.griddata (linéaire + nearest pour les bords).
    La surface finale est reconvertie en vol implicite σ = √(w/T).

    Paramètres
    ----------
    df     : DataFrame avec colonnes Symbol, Spot, Strike, T, ImpliedVol
    symbol : sous-jacent à filtrer
    n_grid : résolution de la grille régulière (n_grid × n_grid)

    Retourne
    --------
    X : grille de log-moneyness  (n_grid × n_grid)
    Y : grille de maturités      (n_grid × n_grid)
    Z : grille de vol implicite  (n_grid × n_grid, en décimal)

    Lève ValueError si moins de 4 points valides pour le symbole.
    """
    sub = df[df["Symbol"] == symbol].dropna(subset=["ImpliedVol", "T", "Strike", "Spot"])

    if len(sub) < 4:
        raise ValueError(
            f"Pas assez de points valides pour construire la surface de {symbol} "
            f"({len(sub)} point(s) trouvé(s), minimum 4)."
        )

    # Log-moneyness relatif au spot (approximation du forward)
    k_vals = np.log(sub["Strike"].values / sub["Spot"].values)
    t_vals = sub["T"].values.astype(float)
    # Variance totale : w = σ²T
    w_vals = sub["ImpliedVol"].values.astype(float) ** 2 * t_vals

    # Grille régulière dans l'espace (k, T)
    k_min, k_max = k_vals.min(), k_vals.max()
    t_min, t_max = t_vals.min(), t_vals.max()

    k_grid = np.linspace(k_min, k_max, n_grid)
    t_grid = np.linspace(t_min, t_max, n_grid)
    X, Y = np.meshgrid(k_grid, t_grid)

    points = np.column_stack((k_vals, t_vals))

    # Interpolation linéaire (convexité respectée à l'intérieur)
    W = griddata(points, w_vals, (X, Y), method="linear")

    # Fallback nearest pour les NaN aux bords
    nan_mask = np.isnan(W)
    if nan_mask.any():
        W_nearest = griddata(points, w_vals, (X, Y), method="nearest")
        W[nan_mask] = W_nearest[nan_mask]

    # Clamp variance totale ≥ 0 (pas de vol implicite négative)
    W = np.clip(W, 1e-8, None)

    # Reconversion : σ = √(w / T), T sécurisé contre la division par zéro
    T_safe = np.where(Y > 1e-6, Y, 1e-6)
    Z = np.sqrt(W / T_safe)

    return X, Y, Z
