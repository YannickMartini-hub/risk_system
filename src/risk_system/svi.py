"""
Fit SVI (Stochastic Volatility Inspired) par tranche de maturité.

Modèle : w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

où :
    k = log-moneyness = ln(K/F)
    w = variance totale = sigma_imp^2 * T
    a, b, rho, m, sigma : paramètres SVI

Contraintes naturelles :
    b > 0          (sourire non-plat)
    |rho| < 1      (corrélation strictement bornée)
    sigma > 0      (lissage au minimum)
    a + b*sigma*(1-|rho|) >= 0  (pas de variance négative)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import SETTINGS

logger = logging.getLogger(__name__)

# Paramètres initiaux et bornes (L-BFGS-B)
_SVI_INIT   = [0.04, 0.1, -0.7, 0.0, 0.1]      # a, b, rho, m, sigma
_SVI_BOUNDS = [
    (-0.5,  1.0),   # a
    (1e-4,  2.0),   # b
    (-0.999, 0.999),# rho
    (-1.0,  1.0),   # m
    (1e-4,  2.0),   # sigma
]


def svi_variance(k: np.ndarray, params: dict) -> np.ndarray:
    """
    Évalue w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).
    params : dict avec clés a, b, rho, m, sigma.
    """
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    km = np.asarray(k, dtype=float) - m
    return a + b * (rho * km + np.sqrt(km ** 2 + sig ** 2))


def svi_vol(k: np.ndarray, T: float, params: dict) -> np.ndarray:
    """Vol implicite SVI : sigma = sqrt(w(k) / T). T doit être > 0."""
    w = svi_variance(k, params)
    w = np.clip(w, 1e-8, None)
    return np.sqrt(w / T)


def fit_svi(k_arr: np.ndarray, w_arr: np.ndarray) -> dict | None:
    """
    Ajuste le modèle SVI sur les points (k, w) d'une tranche de maturité.

    Paramètres
    ----------
    k_arr : log-moneyness ln(K/F), tableau 1-D
    w_arr : variance totale sigma^2 * T, tableau 1-D

    Retourne
    --------
    dict {'a', 'b', 'rho', 'm', 'sigma'} ou None si < 4 points ou divergence.
    """
    k_arr = np.asarray(k_arr, dtype=float)
    w_arr = np.asarray(w_arr, dtype=float)

    valid = np.isfinite(k_arr) & np.isfinite(w_arr) & (w_arr > 0)
    k_arr, w_arr = k_arr[valid], w_arr[valid]

    if len(k_arr) < 4:
        return None

    def _loss(params):
        a, b, rho, m, sig = params
        km = k_arr - m
        w_pred = a + b * (rho * km + np.sqrt(km ** 2 + sig ** 2))
        # Pénalise les variances négatives
        neg = np.sum(np.maximum(-w_pred, 0) ** 2)
        # Pénalise la contrainte de non-arbitrage butterfly
        a_floor = a + b * sig * (1 - abs(rho))
        penalty = max(-a_floor, 0) ** 2
        return np.sum((w_pred - w_arr) ** 2) + 1e4 * neg + 1e4 * penalty

    try:
        res = minimize(
            _loss, _SVI_INIT, method="L-BFGS-B", bounds=_SVI_BOUNDS,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
    except Exception as exc:
        logger.debug("fit_svi: optimisation échouée : %s", exc)
        return None

    if not res.success and res.fun > 1e-2:
        logger.debug("fit_svi: convergence insuffisante (fun=%.4e).", res.fun)
        return None

    a, b, rho, m, sigma = res.x
    return {"a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma)}


def fit_svi_surface(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ajuste SVI par tranche de maturité pour un symbole donné.

    Paramètres
    ----------
    df     : snapshot avec colonnes ImpliedVol, T, Strike, Forward (ou Spot), Symbol
    symbol : ticker à filtrer

    Retourne un DataFrame :
        Symbol | Maturity | T | a | b | rho | m | sigma_svi | n_points | rmse
    Retourne un DataFrame vide si pas assez de données.
    """
    sub = df[df["Symbol"] == symbol].dropna(subset=["ImpliedVol", "T", "Strike"])
    sub = sub[(sub["ImpliedVol"] >= 0.02) & (sub["ImpliedVol"] <= 1.50)]

    if sub.empty:
        logger.warning("[%s] Pas de données pour le fit SVI.", symbol)
        return pd.DataFrame()

    # Log-moneyness forward-based
    fwd = sub["Forward"].values if "Forward" in sub.columns else sub["Spot"].values
    sub = sub.copy()
    sub["_k"] = np.log(sub["Strike"].values / fwd)
    sub["_w"] = sub["ImpliedVol"].values ** 2 * sub["T"].values

    rows: list[dict] = []

    for mat, grp in sub.groupby("Maturity"):
        T_val  = float(grp["T"].iloc[0])
        k_arr  = grp["_k"].values
        w_arr  = grp["_w"].values

        params = fit_svi(k_arr, w_arr)
        if params is None:
            logger.debug("[%s %s] Fit SVI échoué (%d pts).", symbol, mat, len(k_arr))
            continue

        # RMSE en vol implicite
        w_fit  = svi_variance(k_arr, params)
        w_fit  = np.clip(w_fit, 1e-8, None)
        iv_fit = np.sqrt(w_fit / T_val)
        iv_obs = np.sqrt(np.clip(w_arr, 1e-8, None) / T_val)
        rmse   = float(np.sqrt(np.mean((iv_fit - iv_obs) ** 2)))

        rows.append({
            "Symbol":    symbol,
            "Maturity":  mat,
            "T":         T_val,
            "a":         params["a"],
            "b":         params["b"],
            "rho":       params["rho"],
            "m":         params["m"],
            "sigma_svi": params["sigma"],
            "n_points":  len(k_arr),
            "rmse":      rmse,
        })

    result = pd.DataFrame(rows)
    logger.info("[%s] SVI fitté sur %d/%d tranches.", symbol, len(result),
                sub["Maturity"].nunique())
    return result


# ── persistence des paramètres SVI ────────────────────────────────────────────

def save_svi_params(df_params: pd.DataFrame, out_dir: Optional[Path] = None) -> Path:
    """Sauvegarde les paramètres SVI horodatés dans data/parquet/."""
    out_dir = out_dir or SETTINGS.parquet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"svi_params_{ts}.parquet"
    df_params.to_parquet(path, engine="pyarrow", index=False)
    logger.info("SVI params sauvegardés : %s (%d lignes)", path, len(df_params))
    return path


def load_latest_svi_params(parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    """Charge les paramètres SVI les plus récents disponibles."""
    parquet_dir = parquet_dir or SETTINGS.parquet_dir
    files = sorted(parquet_dir.glob("svi_params_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])
