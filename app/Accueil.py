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
        "C Bid":  _get(c, "Bid"),
        "C Ask":  _get(c, "Ask"),
        "C Mid":  _get(c, "Mid"),
        "C IV %": _get(c, "ImpliedVol") * 100 if not c.empty else float("nan"),
        "C Δ":    _get(c, "Delta"),
        "C Γ":    _get(c, "Gamma"),
        "C V":    _get(c, "Vega"),
        "C Θ":    _get(c, "Theta"),
        "Strike": k,
        "_atm":   k == atm_strike,
        "P Bid":  _get(p, "Bid"),
        "P Ask":  _get(p, "Ask"),
        "P Mid":  _get(p, "Mid"),
        "P IV %": _get(p, "ImpliedVol") * 100 if not p.empty else float("nan"),
        "P Δ":    _get(p, "Delta"),
        "P Γ":    _get(p, "Gamma"),
        "P V":    _get(p, "Vega"),
        "P Θ":    _get(p, "Theta"),
    })

chain_df = pd.DataFrame(rows)
display_df = chain_df.drop(columns=["_atm"])

atm_flags = chain_df["_atm"].values


def _highlight(row):
    return (
        ["background-color: #fffacd; font-weight: bold"] * len(row)
        if atm_flags[row.name]
        else [""] * len(row)
    )


styled = display_df.style.apply(_highlight, axis=1)

st.dataframe(
    styled,
    use_container_width=True,
    height=600,
    column_config={
        "Strike":  st.column_config.NumberColumn("Strike", format="%.0f"),
        "C Bid":   st.column_config.NumberColumn("Bid C",  format="%.2f"),
        "C Ask":   st.column_config.NumberColumn("Ask C",  format="%.2f"),
        "C Mid":   st.column_config.NumberColumn("Mid C",  format="%.2f"),
        "C IV %":  st.column_config.NumberColumn("IV C %", format="%.1f"),
        "C Δ":     st.column_config.NumberColumn("Δ C",    format="%.4f"),
        "C Γ":     st.column_config.NumberColumn("Γ C",    format="%.6f"),
        "C V":     st.column_config.NumberColumn("V C",    format="%.2f"),
        "C Θ":     st.column_config.NumberColumn("Θ C",    format="%.2f"),
        "P Bid":   st.column_config.NumberColumn("Bid P",  format="%.2f"),
        "P Ask":   st.column_config.NumberColumn("Ask P",  format="%.2f"),
        "P Mid":   st.column_config.NumberColumn("Mid P",  format="%.2f"),
        "P IV %":  st.column_config.NumberColumn("IV P %", format="%.1f"),
        "P Δ":     st.column_config.NumberColumn("Δ P",    format="%.4f"),
        "P Γ":     st.column_config.NumberColumn("Γ P",    format="%.6f"),
        "P V":     st.column_config.NumberColumn("V P",    format="%.2f"),
        "P Θ":     st.column_config.NumberColumn("Θ P",    format="%.2f"),
    },
)

st.caption(
    "IV en % | Vega par 1% de vol | Theta par jour calendaire | "
    "Strike ATM surligné en jaune | Delta filtré ±30%"
)
