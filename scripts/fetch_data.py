#!/usr/bin/env python3
"""
Génère un snapshot Parquet de DÉMONSTRATION (données synthétiques) pour
développer et tester l'app Streamlit sans connexion IBKR.

Les prix sont produits par Black-Scholes avec un smile de volatilité
paramétrique réaliste (skew négatif + sourire), puis repassés dans la
chaîne normale implied_vol → grecs → filtre delta ±30, exactement comme
les vraies données. Le fichier est nommé snapshot_DEMO_*.parquet pour
qu'il soit impossible de le confondre avec un vrai snapshot.

Usage :
    python scripts/make_demo_snapshot.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from risk_system.config import SETTINGS
from risk_system.pricing import bs_price
from risk_system.implied_vol import implied_vol
from risk_system.greeks import all_greeks

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("make_demo_snapshot")

# ── univers de démo : (symbole, spot, vol ATM, pas de strike) ────────────────
UNIVERSE = [
    ("SX5E", 5450.0, 0.165, 50.0),
    ("ASML", 880.0, 0.30, 10.0),
    ("LVMH", 620.0, 0.26, 10.0),
    ("TTE",  62.0,  0.22, 1.0),
    ("SAP",  255.0, 0.24, 5.0),
    ("AIR",  192.0, 0.25, 2.0),
]

N_EXPIRIES = 6   # échéances mensuelles (3e vendredi)
R, Q = SETTINGS.r, SETTINGS.q


def third_friday(year: int, month: int) -> datetime:
    """3e vendredi du mois (échéance standard des options)."""
    d = datetime(year, month, 1)
    fridays = [d + timedelta(days=i) for i in range(31)
               if (d + timedelta(days=i)).month == month
               and (d + timedelta(days=i)).weekday() == 4]
    return fridays[2]


def next_expiries(n: int) -> list[datetime]:
    """Les n prochaines échéances mensuelles futures."""
    today = datetime.now()
    out: list[datetime] = []
    y, m = today.year, today.month
    while len(out) < n:
        e = third_friday(y, m)
        if e > today:
            out.append(e)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def smile_vol(atm_vol: float, k_log: float, T: float) -> float:
    """
    Smile paramétrique réaliste pour actions/indices :
    skew négatif (puts OTM plus chers) + convexité, atténués avec √T.
    σ(k,T) = σ_ATM + a·k/√T + b·k²/√T   avec a<0.
    """
    a, b = -0.12, 0.35
    sqrt_t = math.sqrt(max(T, 1e-3))
    vol = atm_vol + a * k_log / sqrt_t + b * k_log * k_log / sqrt_t
    return min(max(vol, 0.05), 1.50)


def main() -> None:
    records: list[dict] = []
    expiries = next_expiries(N_EXPIRIES)

    for symbol, spot, atm_vol, k_step in UNIVERSE:
        multiplier = SETTINGS.MULTIPLIER if symbol == "SX5E" else 1
        n_sym = 0

        for exp in expiries:
            T = (exp - datetime.now()).days / SETTINGS.T_BASIS
            if T <= 0:
                continue

            # grille de strikes ±20% autour du spot, alignée sur k_step
            lo = math.floor(spot * 0.80 / k_step) * k_step
            hi = math.ceil(spot * 1.20 / k_step) * k_step
            strikes = [lo + i * k_step for i in range(int((hi - lo) / k_step) + 1)]

            for K in strikes:
                k_log = math.log(K / spot)
                sigma_true = smile_vol(atm_vol, k_log, T)

                for right in ("C", "P"):
                    # prix "marché" = prix BS au smile + micro-spread
                    mid = bs_price(spot, K, T, R, Q, sigma_true, right)
                    if mid < 0.05:          # options sans valeur : ignorées
                        continue
                    half_spread = max(0.01, 0.01 * mid)
                    bid, ask = mid - half_spread, mid + half_spread

                    # chaîne normale : on RE-déduit l'IV depuis le prix,
                    # comme pour les vraies données (validation round-trip)
                    iv = implied_vol(spot, K, T, R, Q, mid, right)
                    if iv is None:
                        continue

                    g = all_greeks(spot, K, T, R, Q, iv,
                                   right=right, multiplier=multiplier)
                    if not (SETTINGS.DELTA_MIN <= g["Delta"] <= SETTINGS.DELTA_MAX):
                        continue

                    records.append({
                        "Symbol":     symbol,
                        "Spot":       spot,
                        "Strike":     float(K),
                        "Maturity":   exp.strftime("%Y-%m-%d"),
                        "T":          T,
                        "Type":       right,
                        "Bid":        round(bid, 2),
                        "Ask":        round(ask, 2),
                        "Mid":        round(mid, 2),
                        "ImpliedVol": iv,
                        "Delta":      g["Delta"],
                        "Gamma":      g["Gamma"],
                        "Vega":       g["Vega"],
                        "Theta":      g["Theta"],
                        "Rho":        g["Rho"],
                    })
                    n_sym += 1

        logger.info("[%s] %d lignes de démo.", symbol, n_sym)

    df = pd.DataFrame(records)
    SETTINGS.parquet_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SETTINGS.parquet_dir / f"snapshot_DEMO_{ts}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Snapshot de démo : %s (%d lignes)", path, len(df))
    logger.warning("Données SYNTHÉTIQUES — à remplacer par un vrai run fetch_data.py.")


if __name__ == "__main__":
    main()