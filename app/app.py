"""
Page 1 — Option Chain (Prix du marché).

Tableau style salle de marché : Calls | Strike | Puts
avec IV, Delta, Gamma, Vega, Theta pour chaque option.
Lit le dernier snapshot Parquet disponible — aucune connexion IBKR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_system.market_data import list_snapshots

st.set_page_config(
    page_title="Risk System — Option Chain",
    page_icon="📊",
    layout="wide",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Risk System")

snapshots = list_snapshots()
if not snapshots:
    st.title("Option Chain")
    st.warning(
        "Aucun snapshot disponible dans `data/parquet/`.\n\n"
        "Lancez d'abord la collecte :\n```bash\npython scripts/fetch_data.py\n```"
    )
    st.stop()

snapshot_names = [p.name for p in snapshots]
selected_name  = st.sidebar.selectbox("Snapshot", snapshot_names)
selected_path  = next(p for p in snapshots if p.name == selected_name)


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


df_all = _load(str(selected_path))

symbols    = sorted(df_all["Symbol"].unique())
symbol     = st.sidebar.selectbox("Sous-jacent", symbols)
df_sym     = df_all[df_all["Symbol"] == symbol]
maturities = sorted(df_sym["Maturity"].unique())
maturity   = st.sidebar.selectbox("Maturité", maturities)

df = df_sym[df_sym["Maturity"] == maturity].copy()

# ── en-tête ───────────────────────────────────────────────────────────────────
spot_val = float(df["Spot"].iloc[0]) if not df.empty else 0.0
T_val    = float(df["T"].iloc[0])    if not df.empty else 0.0

st.title(f"Option Chain — {symbol}")
st.caption(
    f"Spot : **{spot_val:,.2f}** | Maturité : **{maturity}** | "
    f"T : **{T_val:.4f} an** | Snapshot : `{selected_name}`"
)

if df.empty:
    st.info("Aucune donnée pour cette sélection.")
    st.stop()

# ── construction du tableau option chain ──────────────────────────────────────
calls = df[df["Type"] == "C"].set_index("Strike").sort_index()
puts  = df[df["Type"] == "P"].set_index("Strike").sort_index()
all_strikes = sorted(set(calls.index) | set(puts.index))

atm_strike = min(all_strikes, key=lambda k: abs(k - spot_val))

rows = []
for k in all_strikes:
    c = calls.loc[k] if k in calls.index else pd.Series(dtype=float)
    p = puts.loc[k]  if k in puts.index  else pd.Series(dtype=float)

    def _get(series: pd.Series, col: str) -> float:
        return float(series[col]) if col in series.index and pd.notna(series[col]) else float("nan")

    rows.append({
        "Bid C":  _get(c, "Bid"),
        "Ask C":  _get(c, "Ask"),
        "Mid C":  _get(c, "Mid"),
        "IV C %": _get(c, "ImpliedVol") * 100 if not c.empty else float("nan"),
        "Δ C":    _get(c, "Delta"),
        "Γ C":    _get(c, "Gamma"),
        "V C":    _get(c, "Vega"),
        "Θ C":    _get(c, "Theta"),
        "Strike": k,
        "_atm":   k == atm_strike,
        "Bid P":  _get(p, "Bid"),
        "Ask P":  _get(p, "Ask"),
        "Mid P":  _get(p, "Mid"),
        "IV P %": _get(p, "ImpliedVol") * 100 if not p.empty else float("nan"),
        "Δ P":    _get(p, "Delta"),
        "Γ P":    _get(p, "Gamma"),
        "V P":    _get(p, "Vega"),
        "Θ P":    _get(p, "Theta"),
    })

chain_df   = pd.DataFrame(rows)
display_df = chain_df.drop(columns=["_atm"])
atm_flags  = chain_df["_atm"].values

# ── style ──────────────────────────────────────────────────────────────────────
# Formats numériques appliqués via le Styler (et non column_config :
# le mélange Styler + NumberColumn provoquait un affichage tronqué du
# Gamma, ex. "056044" au lieu de "0.0560").
_FMT = {
    "Strike": "{:.0f}",
    "Bid C": "{:.2f}", "Ask C": "{:.2f}", "Mid C": "{:.2f}",
    "Bid P": "{:.2f}", "Ask P": "{:.2f}", "Mid P": "{:.2f}",
    "IV C %": "{:.1f}", "IV P %": "{:.1f}",
    "Δ C": "{:.4f}",   "Δ P": "{:.4f}",
    "Γ C": "{:.4f}",   "Γ P": "{:.4f}",
    "V C": "{:.2f}",   "V P": "{:.2f}",
    "Θ C": "{:.2f}",   "Θ P": "{:.2f}",
}

_ATM_STYLE = "background-color: #5c4d00; color: #ffe97f; font-weight: bold"
_STRIKE_STYLE = "font-weight: bold"


def _highlight(row):
    """Ligne ATM : fond ambre + texte clair (lisible sur thème sombre)."""
    if atm_flags[row.name]:
        return [_ATM_STYLE] * len(row)
    return [_STRIKE_STYLE if col == "Strike" else "" for col in row.index]


styled = (
    display_df.style
    .apply(_highlight, axis=1)
    .format(_FMT, na_rep="")        # cellules vides au lieu de None/NaN
)

st.dataframe(
    styled,
    use_container_width=True,
    height=600,
)

st.caption(
    "IV en % | Vega par 1% de vol | Theta par jour calendaire | "
    "Strike ATM surligné | Delta filtré ±30% (seules les options OTM "
    "apparaissent : puts sous le spot, calls au-dessus)"
)