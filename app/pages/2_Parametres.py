import streamlit as st
from risk_system.config import SETTINGS

st.set_page_config(page_title="Risk System - Paramètres", layout="wide", page_icon="⚙️")

st.title("⚙️ Configuration des Paramètres Économiques")
st.write("Ajustement des hypothèses de valorisation pour la session en cours.")

st.subheader("Conventions Monétaires (Zone Euro)")

# Paramétrage des taux calés sur l'environnement de marché Euro
r_input = st.number_input(
    "Taux sans risque monétaire (r) — Ex: ESTER / Bunds Allemands", 
    min_value=0.0, 
    max_value=0.10, 
    value=SETTINGS.r,  # Utilise la valeur par défaut de notre config.py révisée
    step=0.0005, 
    format="%.4f"
)

q_input = st.number_input(
    "Taux de rendement des dividendes estimé (q)", 
    min_value=0.0, 
    max_value=0.10, 
    value=SETTINGS.q, 
    step=0.0005, 
    format="%.4f"
)

# Injection dans le state applicatif globale pour les autres onglets
st.session_state["r"] = r_input
st.session_state["q"] = q_input

st.success(f"Hypothèses enregistrées : Taux d'intérêt = {r_input:.3%} | Dividendes = {q_input:.3%}")

st.info(
    "💡 **Spécificité Institutionnelle :** Le multiplicateur de calcul pour l'indice Euro Stoxx 50 "
    "est configuré à **10 €** par point d'indice (contrat OESX sur EUREX), tandis que les actions composantes "
    "disposent d'un multiplicateur standardisé à **100**. Le moteur de calcul applique automatiquement ces échelles "
    "lors de l'évaluation des Dollar-Grecs."
)