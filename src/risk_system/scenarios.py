"""
Moteur de scénarios de stress (Étape 12).
Génération de grilles de chocs (Spot, Vol, Temps) et calcul du PnL
par revalorisation complète et approximation des Grecques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from .pricing import bs_price
from .greeks import pnl_attribution

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    spot_shift_pct: float
    vol_shift_abs: float
    time_roll_days: float
    description: str
    version: str = "1.0"

def generate_standard_grid() -> List[Scenario]:
    """
    Génère une grille de stress standardisée (Spot x Volatilité).
    Spot : de -20% à +20% (par pas de 10%)
    Volatilité : de -5 pts à +20 pts (par pas de 5 pts)
    Temps : +1 jour
    """
    scenarios = []
    spot_shifts = [-0.20, -0.10, 0.0, 0.10, 0.20]
    vol_shifts = [-0.05, 0.0, 0.05, 0.10, 0.20]
    
    for s_shift in spot_shifts:
        for v_shift in vol_shifts:
            sid = f"S_{s_shift*100:+.0f}%_V_{v_shift*100:+.0f}bp"
            desc = f"Choc Spot {s_shift*100:+.0f}%, Choc Vol {v_shift*100:+.0f} pts, +1 Jour"
            scenarios.append(Scenario(
                scenario_id=sid,
                spot_shift_pct=s_shift,
                vol_shift_abs=v_shift,
                time_roll_days=1.0,
                description=desc
            ))
    
    # Ajout d'un scénario de krach extrême (ex: 1987 ou 2008)
    scenarios.append(Scenario(
        scenario_id="KRACH_EXTREME",
        spot_shift_pct=-0.30,
        vol_shift_abs=0.40,
        time_roll_days=1.0,
        description="Choc extrême : Spot -30%, Vol +40 points"
    ))
    
    return scenarios

def run_scenario_engine(
    positions: List[dict], 
    snapshot: pd.DataFrame, 
    scenarios: List[Scenario],
    r: float, 
    q: float
) -> pd.DataFrame:
    """
    Exécute les scénarios de stress sur un portefeuille.
    
    Paramètres
    ----------
    positions : Liste de dictionnaires contenant 'symbol', 'strike', 'maturity', 'type', 'quantity'.
    snapshot  : Le DataFrame des conditions de marché (load_latest_snapshot).
    scenarios : Liste d'objets Scenario à appliquer.
    
    Retourne
    --------
    Un DataFrame contenant le PnL détaillé par ligne et par scénario.
    """
    if snapshot.empty or not positions:
        logger.warning("Snapshot vide ou portefeuille vide. Fin de l'analyse.")
        return pd.DataFrame()

    results = []

    for pos in positions:
        # Recherche de la ligne de marché correspondante
        mask = (
            (snapshot["Symbol"] == pos["symbol"]) &
            (snapshot["Strike"] == float(pos["strike"])) &
            (snapshot["Maturity"] == pos["maturity"]) &
            (snapshot["Type"] == pos["type"])
        )
        rows = snapshot[mask]
        
        if rows.empty:
            logger.warning(f"Position introuvable dans le snapshot : {pos}")
            continue
            
        row = rows.iloc[0]
        S0 = float(row["Spot"])
        K = float(row["Strike"])
        T0 = float(row["T"])
        sigma0 = float(row["ImpliedVol"])
        right = row["Type"]
        qty = int(pos["quantity"])
        c_mult = float(row.get("ContractMultiplier", 10.0 if pos["symbol"] == "SX5E" else 100.0))

        # Boucle sur chaque scénario pour cette position
        for sc in scenarios:
            # 1. Calcul des nouveaux paramètres
            S_new = S0 * (1.0 + sc.spot_shift_pct)
            sigma_new = max(0.001, sigma0 + sc.vol_shift_abs) # La vol ne peut pas être négative ou nulle
            T_new = max(0.0001, T0 - (sc.time_roll_days / 365.25))

            # 2. Revalorisation complète (Full Repricing)
            price_old = bs_price(S0, K, T0, r, q, sigma0, right)
            price_new = bs_price(S_new, K, T_new, r, q, sigma_new, right)
            full_pnl_unit = price_new - price_old
            full_pnl_total = full_pnl_unit * qty * c_mult

            # 3. Approximation locale par les Grecques (Taylor)
            dS = S_new - S0
            d_sigma = sigma_new - sigma0
            
            approx_greeks = pnl_attribution(
                S=S0, K=K, T=T0, r=r, q=q, sigma=sigma0, right=right,
                dS=dS, d_sigma=d_sigma, dt_days=sc.time_roll_days,
                S_new=S_new, sigma_new=sigma_new, T_new=T_new
            )
            
            approx_pnl_total = approx_greeks["total_approx"] * qty * c_mult

            results.append({
                "ScenarioID": sc.scenario_id,
                "Symbol": pos["symbol"],
                "Strike": K,
                "Type": right,
                "Qty": qty,
                "S0": S0,
                "S_new": S_new,
                "Vol_new": sigma_new,
                "Full_PnL_EUR": full_pnl_total,
                "Approx_PnL_EUR": approx_pnl_total,
                "Delta_PnL_EUR": approx_greeks["delta_pnl"] * qty * c_mult,
                "Gamma_PnL_EUR": approx_greeks["gamma_pnl"] * qty * c_mult,
                "Vega_PnL_EUR": approx_greeks["vega_pnl"] * qty * c_mult,
                "Theta_PnL_EUR": approx_greeks["theta_pnl"] * qty * c_mult,
                "Residual_EUR": full_pnl_total - approx_pnl_total
            })

    df_results = pd.DataFrame(results)
    return df_results

def get_worst_case_scenarios(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les PnL par scénario au niveau du portefeuille global 
    et identifie les pires pertes.
    """
    if df_results.empty:
        return pd.DataFrame()
        
    portfolio_pnl = df_results.groupby("ScenarioID").agg(
        Total_Full_PnL=("Full_PnL_EUR", "sum"),
        Total_Approx_PnL=("Approx_PnL_EUR", "sum")
    ).reset_index()
    
    # Tri du pire scénario au meilleur
    return portfolio_pnl.sort_values(by="Total_Full_PnL", ascending=True).reset_index(drop=True)