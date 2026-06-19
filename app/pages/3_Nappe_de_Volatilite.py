import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from risk_system.market_data import load_latest_snapshot

st.set_page_config(page_title="Risk System - Nappe de Volatilité", layout="wide", page_icon="📈")

st.title("📈 Analyse des Nappes de Volatilité Implicite")
st.write("Cartographie et modélisation des structures par terme de la volatilité.")

df = load_latest_snapshot()

if df.empty:
    st.warning("⚠️ Aucun snapshot trouvé. Veuillez exécuter l'extraction de données.")
    st.stop()

available_tickers = sorted(df["Symbol"].unique())
default_idx = available_tickers.index("SX5E") if "SX5E" in available_tickers else 0
selected_ticker = st.selectbox("Sélectionnez la surface à analyser :", available_tickers, index=default_idx)

df_surface = df[df["Symbol"] == selected_ticker].copy()
spot_price = df_surface["Spot"].iloc[0]

# ── LOGIQUE DES TENORS (Cohérence avec l'Accueil) ─────────────────────────────
def map_days_to_tenor(t_years):
    days = round(t_years * 365.25)
    if days <= 20: return "2W"
    elif days <= 45: return "1M"
    elif days <= 120: return "3M"
    elif days <= 220: return "6M"
    elif days <= 310: return "9M"
    elif days <= 420: return "12M"
    elif days <= 600: return "18M"
    elif days <= 800: return "24M"
    else: return "36M"

# Application de la fonction pour créer le label de légende propre
df_surface["Tenor"] = df_surface["T"].apply(map_days_to_tenor)
df_surface["LegendLabel"] = df_surface["Tenor"] + " (" + df_surface["Maturity"] + ")"

# ── SELECTION DE LA METRIQUE DE L'AXE X ───────────────────────────────────────
st.sidebar.subheader("📐 Paramètres de l'Axe Horizontal")
x_axis_type = st.sidebar.radio(
    "Métrique de l'axe X :",
    ["Strike Absolu", "Moneyness (K/S)", "Log-Moneyness ln(K/S)"]
)

# Préparation des variables de coordonnées
df_surface["Moneyness"] = df_surface["Strike"] / spot_price
df_surface["LogMoneyness"] = np.log(df_surface["Moneyness"])

if x_axis_type == "Strike Absolu":
    x_col = "Strike"
    x_label = "Prix d'Exercice (Strike)"
elif x_axis_type == "Moneyness (K/S)":
    x_col = "Moneyness"
    x_label = "Moneyness (K / S)"
else:
    x_col = "LogMoneyness"
    x_label = "Log-Moneyness ln(K / S)"

# ── SELECTION DES OPTIONS OUT-OF-THE-MONEY (OTM) ──────────────────────────────
filter_otm = st.sidebar.checkbox("Filtrer uniquement les options OTM (Recommandé)", value=True)

if filter_otm:
    df_surface = df_surface[
        ((df_surface["Type"] == "P") & (df_surface["Strike"] < spot_price)) |
        ((df_surface["Type"] == "C") & (df_surface["Strike"] >= spot_price))
    ]

# ── FILTRES QUALITÉ ───────────────────────────────────────────────────────────
st.sidebar.subheader("🛡️ Filtrage des Valeurs Aberrantes")
max_iv_slider = st.sidebar.slider("Volatilité Implicite Max Acceptée (%)", min_value=50, max_value=250, value=90)

df_surface = df_surface[
    (df_surface["Volume"] >= 0) & 
    (df_surface["ImpliedVol"] >= 0.01) & 
    (df_surface["ImpliedVol"] <= max_iv_slider / 100.0)
]

if df_surface.empty:
    st.error("Aucune donnée ne correspond aux critères sélectionnés.")
    st.stop()

# ── GRAPHIC 3D SURFACE ────────────────────────────────────────────────────────
st.subheader(f"Surface de Volatilité 3D — {selected_ticker} (Spot: {spot_price:.2f} €)")

fig_3d = go.Figure(data=[go.Mesh3d(
    x=df_surface[x_col],
    y=df_surface['T'] * 365.25,
    z=df_surface['ImpliedVol'] * 100,
    intensity=df_surface['ImpliedVol'] * 100,
    colorscale='Viridis',
    opacity=0.88
)])

fig_3d.update_layout(
    scene=dict(
        xaxis_title=x_label,
        yaxis_title="Maturité (Jours)",
        zaxis_title="Volatilité Implicite (%)"
    ),
    margin=dict(l=10, r=10, b=10, t=10),
    height=600
)
st.plotly_chart(fig_3d, use_container_width=True)

# ── GRAPHIC 2D SMILES ─────────────────────────────────────────────────────────
st.subheader("Coupes Transversales : Structures par Strike / Monnaie")

# Extraction des labels uniques triés par le temps (T) pour respecter l'ordre chronologique
unique_mats = df_surface[['Maturity', 'T', 'LegendLabel']].drop_duplicates().sort_values('T')

fig_2d = go.Figure()

for _, row in unique_mats.iterrows():
    mat = row["Maturity"]
    label = row["LegendLabel"]
    
    df_mat = df_surface[df_surface["Maturity"] == mat].sort_values(x_col)
    
    if len(df_mat) > 3:
        fig_2d.add_trace(go.Scatter(
            x=df_mat[x_col],
            y=df_mat["ImpliedVol"] * 100,
            mode='lines+markers',
            name=label  # Utilisation du nouveau label "Tenor (Date)"
        ))

# Repère vertical pour l'ATM
if x_axis_type == "Strike Absolu":
    ref_line = spot_price
elif x_axis_type == "Moneyness (K/S)":
    ref_line = 1.0
else:
    ref_line = 0.0

fig_2d.add_vline(x=ref_line, line_dash="dash", line_color="red", annotation_text="Niveau ATM")

fig_2d.update_layout(
    xaxis_title=x_label,
    yaxis_title="Volatilité Implicite (%)",
    hovermode="x unified",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
st.plotly_chart(fig_2d, use_container_width=True)