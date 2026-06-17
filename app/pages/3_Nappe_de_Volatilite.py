"""
Page 3 — Nappe de Volatilité 3D.

Surface Plotly interactive (rotation/zoom) interpolée en variance totale σ²T.
Log-moneyness forward-based k = ln(K/F).
Axe X : log-moneyness ou strike (toggle).
Overlay courbe SVI par maturité si paramètres disponibles.
Points bruts optionnels.
"""

from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from risk_system.surface import build_surface
from risk_system.market_data import list_snapshots
from risk_system.svi import load_latest_svi_params, svi_vol

st.set_page_config(page_title="Risk System — Surface", layout="wide")
st.title("Nappe de Volatilité Implicite")

# ── snapshot ──────────────────────────────────────────────────────────────────
snapshots = list_snapshots()
if not snapshots:
    st.warning("Aucun snapshot. Lancez `python scripts/fetch_data.py`.")
    st.stop()

snapshot_names = [p.name for p in snapshots]
selected_name  = st.sidebar.selectbox("Snapshot", snapshot_names)
selected_path  = next(p for p in snapshots if p.name == selected_name)


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def _load_svi(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path else pd.DataFrame()


df_all = _load(str(selected_path))

# Charge les paramètres SVI les plus récents (si disponibles)
svi_files = sorted(selected_path.parent.glob("svi_params_*.parquet"), reverse=True)
df_svi    = _load_svi(str(svi_files[0])) if svi_files else pd.DataFrame()

# ── sélecteurs ────────────────────────────────────────────────────────────────
symbols = sorted(df_all["Symbol"].unique())
symbol  = st.sidebar.selectbox("Sous-jacent", symbols)

x_axis    = st.sidebar.radio("Axe X", ["Log-Moneyness ln(K/F)", "Strike"], index=0)
show_raw  = st.sidebar.checkbox("Afficher les points de marché", value=True)
show_svi  = st.sidebar.checkbox("Overlay SVI", value=not df_svi.empty)
n_grid    = st.sidebar.slider("Résolution de la grille", 20, 100, 50, step=5)

# ── construction de la surface ────────────────────────────────────────────────
df_sym = df_all[df_all["Symbol"] == symbol].copy()

try:
    X, Y, Z = build_surface(df_sym, symbol, n_grid=n_grid)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Forward de référence (première valeur disponible par maturité, fallback Spot)
if "Forward" in df_sym.columns:
    fwd_ref = float(df_sym.groupby("Maturity")["Forward"].first().mean())
else:
    fwd_ref = float(df_sym["Spot"].iloc[0])

# Conversion axe X si besoin
use_strike = x_axis == "Strike"
if use_strike:
    X_display = fwd_ref * np.exp(X)
    x_label   = "Strike"
else:
    X_display = X
    x_label   = "Log-Moneyness k=ln(K/F)"

# ── figure Plotly ─────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Surface(
    x=X_display,
    y=Y,
    z=Z * 100,           # en pourcentage
    colorscale="Viridis",
    opacity=0.85,
    colorbar=dict(title="IV (%)"),
    name="Surface interpolée",
    hovertemplate=(
        f"{x_label}: %{{x:.4f}}<br>"
        "Maturité: %{y:.4f} an<br>"
        "IV: %{z:.2f}%<extra></extra>"
    ),
))

# ── points de marché bruts ────────────────────────────────────────────────────
if show_raw:
    fwd_col = df_sym["Forward"].values if "Forward" in df_sym.columns else df_sym["Spot"].values
    k_raw   = np.log(df_sym["Strike"].values / fwd_col)
    x_raw   = fwd_ref * np.exp(k_raw) if use_strike else k_raw
    fig.add_trace(go.Scatter3d(
        x=x_raw,
        y=df_sym["T"].values,
        z=df_sym["ImpliedVol"].values * 100,
        mode="markers",
        marker=dict(size=3, color="red", opacity=0.7),
        name="Points de marché",
        hovertemplate=(
            f"{x_label}: %{{x:.4f}}<br>"
            "T: %{y:.4f}<br>"
            "IV: %{z:.2f}%<extra></extra>"
        ),
    ))

# ── overlay SVI par maturité ─────────────────────────────────────────────────
if show_svi and not df_svi.empty:
    svi_sym = df_svi[df_svi["Symbol"] == symbol]
    if not svi_sym.empty:
        k_range = np.linspace(X.min(), X.max(), 80)
        for _, svi_row in svi_sym.iterrows():
            T_val  = float(svi_row["T"])
            params = {c: float(svi_row[c]) for c in ("a", "b", "rho", "m", "sigma_svi")}
            params["sigma"] = params.pop("sigma_svi")
            iv_curve = svi_vol(k_range, T_val, params) * 100
            x_curve  = fwd_ref * np.exp(k_range) if use_strike else k_range
            fig.add_trace(go.Scatter3d(
                x=x_curve,
                y=np.full_like(k_range, T_val),
                z=iv_curve,
                mode="lines",
                line=dict(color="orange", width=4),
                name=f"SVI {svi_row['Maturity']}",
                hovertemplate=(
                    f"SVI — T={T_val:.3f}an<br>"
                    f"{x_label}: %{{x:.4f}}<br>"
                    "IV SVI: %{z:.2f}%<extra></extra>"
                ),
            ))

fig.update_layout(
    scene=dict(
        xaxis_title=x_label,
        yaxis_title="Maturité (ans)",
        zaxis_title="Vol implicite (%)",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        xaxis=dict(backgroundcolor="rgb(245,245,245)"),
        yaxis=dict(backgroundcolor="rgb(245,245,245)"),
        zaxis=dict(backgroundcolor="rgb(245,245,245)"),
    ),
    height=680,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(x=0.02, y=0.98),
    title=dict(
        text=f"Surface de vol — {symbol}",
        x=0.5, xanchor="center",
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ── statistiques rapides ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Min IV", f"{df_sym['ImpliedVol'].min()*100:.1f}%")
with col2:
    st.metric("Médiane IV", f"{df_sym['ImpliedVol'].median()*100:.1f}%")
with col3:
    st.metric("Max IV", f"{df_sym['ImpliedVol'].max()*100:.1f}%")

svi_status = f"{len(df_svi[df_svi['Symbol']==symbol])} tranches SVI" if not df_svi.empty else "pas de params SVI"
st.caption(
    f"Interpolation en variance totale σ²T | log-moneyness k=ln(K/F) | "
    f"grille {n_grid}×{n_grid} | {svi_status} | {symbol} | {selected_name}"
)
