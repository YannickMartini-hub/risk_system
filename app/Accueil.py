"""
Page 1 — Option Chain (Prix du marché).

Tableau style salle de marché : Calls | Strike | Puts
avec volume, IV, grecs en valeur et en €.
Lit le dernier snapshot Parquet disponible — aucune connexion IBKR.
"""

from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_system.market_data import list_snapshots

_TENORS = [("1w", 7), ("2w", 14), ("1m", 30), ("2m", 60), ("3m", 91),
           ("4m", 122), ("6m", 182), ("9m", 273), ("1y", 365), ("18m", 548), ("2y", 730)]


def _tenor_label(mat_str: str) -> str:
    days = (_date.fromisoformat(mat_str) - _date.today()).days
    if days <= 0:
        return mat_str
    return min(_TENORS, key=lambda x: abs(x[1] - days))[0]


def _maturity_options(maturities: list[str]) -> list[tuple[str, str]]:
    labeled = [(mat, _tenor_label(mat)) for mat in maturities]
    counts: dict[str, int] = {}
    for _, lbl in labeled:
        counts[lbl] = counts.get(lbl, 0) + 1
    return [(f"{lbl} ({mat})", mat) if counts[lbl] > 1 else (lbl, mat)
            for mat, lbl in labeled]


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
maturities  = sorted(df_sym["Maturity"].unique())
mat_opts    = _maturity_options(maturities)
mat_labels  = [lbl for lbl, _ in mat_opts]
mat_dates   = [dt  for _, dt  in mat_opts]
mat_label   = st.sidebar.selectbox("Maturité", mat_labels)
maturity    = mat_dates[mat_labels.index(mat_label)]

df = df_sym[df_sym["Maturity"] == maturity].copy()

# ── en-tête ───────────────────────────────────────────────────────────────────
spot_val = float(df["Spot"].iloc[0]) if not df.empty else 0.0
T_val    = float(df["T"].iloc[0])    if not df.empty else 0.0

st.title(f"Option Chain — {symbol}")
st.caption(
    f"Spot : **{spot_val:,.2f}** | Maturité : **{mat_label}** ({maturity}) | "
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

    def _iv(series: pd.Series) -> float:
        v = _get(series, "ImpliedVol")
        return v * 100 if not pd.isna(v) else float("nan")

    rows.append({
        # ── CALLS ────────────────────────────────────────────────────────
        "Vol C":   _get(c, "Volume"),
        "Bid C":   _get(c, "Bid"),
        "Ask C":   _get(c, "Ask"),
        "Mid C":   _get(c, "Mid"),
        "IV C %":  _iv(c),
        "Δ C":     _get(c, "Delta"),
        "Γ C":     _get(c, "Gamma"),
        "V C":     _get(c, "Vega"),
        "Θ C":     _get(c, "Theta"),
        "€Δ C":    _get(c, "DollarDelta"),
        "€Γ C":    _get(c, "DollarGamma"),
        "€V C":    _get(c, "DollarVega"),
        "€Θ C":    _get(c, "DollarTheta"),
        # ── STRIKE ───────────────────────────────────────────────────────
        "Strike":  k,
        "_atm":    k == atm_strike,
        # ── PUTS ─────────────────────────────────────────────────────────
        "Vol P":   _get(p, "Volume"),
        "Bid P":   _get(p, "Bid"),
        "Ask P":   _get(p, "Ask"),
        "Mid P":   _get(p, "Mid"),
        "IV P %":  _iv(p),
        "Δ P":     _get(p, "Delta"),
        "Γ P":     _get(p, "Gamma"),
        "V P":     _get(p, "Vega"),
        "Θ P":     _get(p, "Theta"),
        "€Δ P":    _get(p, "DollarDelta"),
        "€Γ P":    _get(p, "DollarGamma"),
        "€V P":    _get(p, "DollarVega"),
        "€Θ P":    _get(p, "DollarTheta"),
    })

chain_df  = pd.DataFrame(rows)
# Convertit toutes les colonnes numériques en float64 pour que na_rep="—" fonctionne
_num_cols = [c for c in chain_df.columns if c not in ("Strike", "_atm")]
chain_df[_num_cols] = chain_df[_num_cols].apply(pd.to_numeric, errors="coerce")
display_df = chain_df.drop(columns=["_atm"])
atm_flags  = chain_df["_atm"].values

# ── style ──────────────────────────────────────────────────────────────────────
_FMT = {
    "Strike":  "{:.0f}",
    # marché
    "Vol C":  "{:.0f}",   "Vol P":  "{:.0f}",
    "Bid C":  "{:.2f}",   "Bid P":  "{:.2f}",
    "Ask C":  "{:.2f}",   "Ask P":  "{:.2f}",
    "Mid C":  "{:.2f}",   "Mid P":  "{:.2f}",
    "IV C %": "{:.1f}",   "IV P %": "{:.1f}",
    # grecs en valeur
    "Δ C":    "{:.4f}",   "Δ P":    "{:.4f}",
    "Γ C":    "{:.6f}",   "Γ P":    "{:.6f}",
    "V C":    "{:.2f}",   "V P":    "{:.2f}",
    "Θ C":    "{:.2f}",   "Θ P":    "{:.2f}",
    # grecs en €
    "€Δ C":   "{:.0f}",   "€Δ P":   "{:.0f}",
    "€Γ C":   "{:.2f}",   "€Γ P":   "{:.2f}",
    "€V C":   "{:.2f}",   "€V P":   "{:.2f}",
    "€Θ C":   "{:.2f}",   "€Θ P":   "{:.2f}",
}

_ATM_STYLE    = "background-color: #5c4d00; color: #ffe97f; font-weight: bold"
_STRIKE_STYLE = "font-weight: bold"


def _highlight(row):
    if atm_flags[row.name]:
        return [_ATM_STYLE] * len(row)
    styles = []
    for col in row.index:
        if col == "Strike":
            styles.append(_STRIKE_STYLE)
        else:
            styles.append("")
    return styles


styled = (
    display_df.style
    .apply(_highlight, axis=1)
    .format(_FMT, na_rep="—")
)

st.dataframe(
    styled,
    use_container_width=True,
    height=620,
)

st.caption(
    "**Grecs en valeur** : Δ (sans unité) | Γ (par point²) | "
    "Vega (par 1% de vol) | Θ (par jour cal.) — "
    "**Grecs en €** : €Δ = Δ × Spot × mult | €Γ = Γ × Spot² × mult | "
    "€Vega = Vega × mult | €Θ = Θ × mult — "
    "Strike ATM surligné"
)
