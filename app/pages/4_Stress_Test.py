# pages/4_Stress_Test.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from risk_system.market_data import load_latest_snapshot
from risk_system.scenarios import generate_standard_grid, run_scenario_engine, get_worst_case_scenarios

st.set_page_config(page_title="Risk System - Stress Test", layout="wide", page_icon="⚡")

st.title("⚡ Moteur de Scénarios de Stress (Étape 12)")
st.write("Évaluation institutionnelle des risques : Full Repricing vs Approximation locale par les Grecques.")

# ── 1. CONDITIONS DE MARCHÉ ET RECUPÉRATION DU STATE ──────────────────────────
r = st.session_state.get("r", 0.035)
q = st.session_state.get("q", 0.0)

st.sidebar.markdown("### 🌐 Environnement")
st.sidebar.markdown(f"Taux sans risque ($r$) : **{r:.3%}**")
st.sidebar.markdown(f"Rendement div. ($q$) : **{q:.3%}**")

snapshot = load_latest_snapshot()

if snapshot.empty:
    st.warning("⚠️ Aucun snapshot trouvé. Veuillez lancer l'extraction de données.")
    st.stop()

# ── 2. SÉLECTION / CONFIGURATION D'UN PORTEFEUILLE DE TEST ────────────────────
st.subheader("📁 Composition du Portefeuille à Scénariser")

# Pour l'exemple, on liste les options disponibles sur le SX5E pour créer une position
df_sx5e = snapshot[snapshot["Symbol"] == "SX5E"]
if df_sx5e.empty:
    st.error("Aucune donnée disponible pour l'indice SX5E.")
    st.stop()

available_maturities = sorted(df_sx5e["Maturity"].unique())

col1, col2, col3, col4 = st.columns(4)
with col1:
    mat_selected = st.selectbox("Échéance de l'option :", available_maturities)
with col2:
    df_mat = df_sx5e[df_sx5e["Maturity"] == mat_selected]
    strikes_avail = sorted(df_mat["Strike"].unique())
    # On centre par défaut sur le strike le plus proche du Spot
    spot_ref = df_mat["Spot"].iloc[0]
    default_strike_idx = int(np.abs(np.array(strikes_avail) - spot_ref).argmin())
    strike_selected = st.selectbox("Strike :", strikes_avail, index=default_strike_idx)
with col3:
    type_selected = st.radio("Type :", ["C", "P"], horizontal=True)
with col4:
    qty_selected = st.number_input("Quantité (Positif = Long, Négatif = Short) :", value=10, step=1)

# Structuration du portefeuille pour le moteur de scénarios
# Note : On peut ajouter plusieurs lignes à cette liste si nécessaire
portfolio = [{
    "symbol": "SX5E",
    "strike": strike_selected,
    "maturity": mat_selected,
    "type": type_selected,
    "quantity": qty_selected
}]

st.info(f"Position actuelle : **{qty_selected}** contrats sur SX5E {type_selected} {strike_selected} ({mat_selected})")

# ── 3. EXÉCUTION DU MOTEUR DE SCÉNARIOS ───────────────────────────────────────
scenarios_grid = generate_standard_grid()

with st.spinner("Calcul de la matrice de stress en cours..."):
    df_details = run_scenario_engine(portfolio, snapshot, scenarios_grid, r, q)
    df_summary = get_worst_case_scenarios(df_details)

if df_summary.empty:
    st.error("Erreur lors de l'exécution des scénarios.")
    st.stop()

# ── 4. AFFICHAGE DES RÉSULTATS (WORST-CASE) ───────────────────────────────────
st.subheader("🚨 Analyse des Pires Scénarios (Worst-Case)")

worst_sc = df_summary.iloc[0]
best_sc = df_summary.iloc[-1]

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        label=f"Pire Scénario : {worst_sc['ScenarioID']}", 
        value=f"{worst_sc['Total_Full_PnL']:,.0f} €",
        delta=f"Approximé: {worst_sc['Total_Approx_PnL']:,.0f} €",
        delta_color="inverse"
    )
with m2:
    st.metric(
        label=f"Meilleur Scénario : {best_sc['ScenarioID']}", 
        value=f"{best_sc['Total_Full_PnL']:,.0f} €"
    )
with m3:
    # Calcul du résidu moyen ou max pour valider la qualité de l'approximation
    max_residual = df_details["Residual_EUR"].abs().max()
    st.metric(label="Erreur d'approximation Max (Résidu)", value=f"{max_residual:,.0f} €")

# ── 5. VISUALISATION DE LA MATRICE DE STRESS ──────────────────────────────────
st.subheader("📊 Grille Globale des Risques (Spot vs Vol)")

# Pivot pour obtenir une belle matrice à afficher
# On extrait les variations de Spot et Vol depuis l'ID ou on les ré-analyse pour un affichage propre
df_details["Choc Spot"] = df_details["S_new"] / df_details["S0"] - 1.0
df_details["Choc Vol (pts)"] = df_details["Vol_new"] - (df_details["S_new"] * 0.0) # Ajustement cosmétique

# Représentation graphique du PnL du portefeuille selon les scénarios
fig_scenarios = px.bar(
    df_details, 
    x="ScenarioID", 
    y=["Delta_PnL_EUR", "Gamma_PnL_EUR", "Vega_PnL_EUR", "Theta_PnL_EUR", "Residual_EUR"],
    title="Attribution du PnL par Facteur de Risque (€)",
    labels={"value": "PnL (€)", "variable": "Attribution"},
    barmode="relative"
)
st.plotly_chart(fig_scenarios, use_container_width=True)

# Affichage de la table de données complète pour l'audit opérationnel (Conforme Partie III)
st.subheader("📋 Rapport de Triage des Scénarios")
st.dataframe(
    df_details[[
        "ScenarioID", "Qty", "Full_PnL_EUR", "Approx_PnL_EUR", 
        "Delta_PnL_EUR", "Gamma_PnL_EUR", "Vega_PnL_EUR", "Theta_PnL_EUR", "Residual_EUR"
    ]].style.format({
        "Full_PnL_EUR": "{:,.0f} €", "Approx_PnL_EUR": "{:,.0f} €",
        "Delta_PnL_EUR": "{:,.0f} €", "Gamma_PnL_EUR": "{:,.0f} €",
        "Vega_PnL_EUR": "{:,.0f} €", "Theta_PnL_EUR": "{:,.0f} €",
        "Residual_EUR": "{:,.0f} €"
    }), 
    use_container_width=True
)