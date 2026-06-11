"""
Page 3 — Nappe de Volatilité 3D.

Surface Plotly interactive (rotation/zoom) interpolée en variance totale σ²T.
Axe X : log-moneyness ou strike (toggle). Points bruts optionnels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from risk_system.surface import build_surface
from risk_system.market_data import list_snapshots

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


df_all = _load(str(selected_path))

# ── sélecteurs ────────────────────────────────────────────────────────────────
symbols = sorted(df_all["Symbol"].unique())
symbol  = st.sidebar.selectbox("Sous-jacent", symbols)

x_axis   = st.sidebar.radio("Axe X", ["Log-Moneyness", "Strike"], index=0)
show_raw = st.sidebar.checkbox("Afficher les points bruts", value=True)
n_grid   = st.sidebar.slider("Résolution de la grille", 20, 100, 50, step=5)

# ── construction de la surface ────────────────────────────────────────────────
df_sym = df_all[df_all["Symbol"] == symbol].copy()

try:
    X, Y, Z = build_surface(df_sym, symbol, n_grid=n_grid)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Conversion axe X si besoin
if x_axis == "Strike":
    spot_ref  = float(df_sym["Spot"].iloc[0])
    X_display = spot_ref * np.exp(X)
    x_label   = "Strike"
else:
    X_display = X
    x_label   = "Log-Moneyness"

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

if show_raw:
    k_raw = np.log(df_sym["Strike"].values / df_sym["Spot"].values)
    x_raw = df_sym["Spot"].values * np.exp(k_raw) if x_axis == "Strike" else k_raw
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

st.caption(
    f"Interpolation en variance totale σ²T | grille {n_grid}×{n_grid} | "
    f"{symbol} | {selected_name}"
)
