import streamlit as st
import pandas as pd
import numpy as np
from risk_system.market_data import load_latest_snapshot
from risk_system.implied_vol import implied_vol
from risk_system.greeks import all_greeks

st.set_page_config(page_title="Risk System - Accueil", layout="wide", page_icon="📊")

st.title("📊 Système de Gestion des Risques — Euro Stoxx 50")
st.write("Visualisation dynamique des chaînes d'options au standard institutionnel (% et €).")

# ── 1. GESTION DES PARAMÈTRES ET DE LA RÉACTIVITÉ ─────────────────────────────
r = st.session_state.get("r", 0.035)
q = st.session_state.get("q", 0.0)

st.sidebar.markdown("### 🌐 Conditions de Marché Actuelles")
st.sidebar.markdown(f"Taux sans risque ($r$) : **{r:.3%}**")
st.sidebar.markdown(f"Rendement div. ($q$) : **{q:.3%}**")

if st.sidebar.button("🔄 Forcer le recalcul global"):
    st.cache_data.clear()
    st.rerun()

# ── 2. CHARGEMENT ET FILTRAGE DE L'UNIVERS ────────────────────────────────────
df_raw = load_latest_snapshot()

if df_raw.empty:
    st.warning("⚠️ Aucun snapshot de marché trouvé. Veuillez lancer l'extraction via `scripts/fetch_data.py`.")
    st.stop()

available_tickers = sorted(df_raw["Symbol"].unique())
default_idx = available_tickers.index("SX5E") if "SX5E" in available_tickers else 0

selected_ticker = st.selectbox("Sélectionnez un sous-jacent :", available_tickers, index=default_idx)
df_ticker = df_raw[df_raw["Symbol"] == selected_ticker].copy()

if df_ticker.empty:
    st.error("Aucune donnée disponible pour ce symbole.")
    st.stop()

spot_price = df_ticker["Spot"].iloc[0]
st.metric(label=f"Prix Spot de Référence ({selected_ticker})", value=f"{spot_price:,.2f} €")

# ── 3. MOTEUR DE RECALCUL DYNAMIQUE DES GRECS (% ET €) ────────────────────────
@st.cache_data
def recalculate_chain_with_greeks(df_data, r_rate, q_dividend):
    records = []
    for _, row in df_data.iterrows():
        c_mult = row.get("ContractMultiplier", 100.0) if pd.notna(row.get("ContractMultiplier")) else 10.0 if row["Symbol"] == "SX5E" else 100.0
        
        iv = implied_vol(row["Spot"], row["Strike"], row["T"], r_rate, q_dividend, row["Mid"], row["Type"])
        if iv is None or not np.isfinite(iv):
            continue
            
        g = all_greeks(row["Spot"], row["Strike"], row["T"], r_rate, q_dividend, iv, right=row["Type"], multiplier=c_mult)
        
        records.append({
            "Strike": row["Strike"], "Maturity": row["Maturity"], "T": row["T"], "Type": row["Type"],
            "Bid": row["Bid"], "Ask": row["Ask"], "Mid": row["Mid"], "Volume": row["Volume"],
            "ImpliedVol": iv,
            "Delta_%": g["Delta"], "Gamma_%": g["Gamma"], "Vega_%": g["Vega"], "Theta_%": g["Theta"],
            "Delta_EUR": g["Delta"] * row["Spot"] * c_mult,
            "Gamma_EUR": g["DollarGamma"],
            "Vega_EUR": g["DollarVega"],
            "Theta_EUR": g["Theta"] * c_mult
        })
    return pd.DataFrame(records)

with st.spinner("Calcul des structures de risques en cours..."):
    df_recalc = recalculate_chain_with_greeks(df_ticker, r, q)

# ── 4. SÉLECTION DU TENOR INTERMÉDIAIRE ───────────────────────────────────────
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

df_recalc["Tenor"] = df_recalc["T"].apply(map_days_to_tenor)
df_recalc["DropdownLabel"] = df_recalc["Tenor"] + " (" + df_recalc["Maturity"] + ")"

unique_maturities_df = df_recalc[['Maturity', 'T', 'DropdownLabel']].drop_duplicates().sort_values('T')
dropdown_options = unique_maturities_df["DropdownLabel"].tolist()

selected_label = st.selectbox("Sélectionnez une échéance (Tenor) :", dropdown_options)
df_mat = df_recalc[df_recalc["DropdownLabel"] == selected_label]

calls = df_mat[df_mat["Type"] == "C"].sort_values("Strike")
puts = df_mat[df_mat["Type"] == "P"].sort_values("Strike")

cols_to_merge = ['Strike', 'Bid', 'Ask', 'Mid', 'ImpliedVol', 'Delta_%', 'Delta_EUR', 'Gamma_%', 'Gamma_EUR', 'Vega_%', 'Vega_EUR', 'Theta_%', 'Theta_EUR']

# LA CORRECTION EST ICI : how='outer' permet de garder les strikes orphelins (uniquement Call ou uniquement Put)
chain = pd.merge(
    calls[cols_to_merge], puts[cols_to_merge],
    on='Strike', how='outer', suffixes=('_Call', '_Put')
).sort_values("Strike")

if chain.empty:
    st.warning("Aucune donnée disponible pour cette maturité.")
    st.stop()

# ── 5. ALIGNEMENT DES COLONNES STANDARDISÉES : CALLS | STRIKE | PUTS ──────────
call_display_cols = [
    'Delta_%_Call', 'Delta_EUR_Call', 'Gamma_%_Call', 'Gamma_EUR_Call', 
    'Vega_%_Call', 'Vega_EUR_Call', 'Theta_%_Call', 'Theta_EUR_Call', 
    'ImpliedVol_Call', 'Bid_Call', 'Ask_Call', 'Mid_Call'
]

put_display_cols = [
    'Mid_Put', 'Ask_Put', 'Bid_Put', 'ImpliedVol_Put', 
    'Delta_%_Put', 'Delta_EUR_Put', 'Gamma_%_Put', 'Gamma_EUR_Put', 
    'Vega_%_Put', 'Vega_EUR_Put', 'Theta_%_Put', 'Theta_EUR_Put'
]

chain = chain[call_display_cols + ['Strike'] + put_display_cols]

# L'ASTUCE : On ajoute un espace " " à la fin des noms de colonnes des Puts pour duper Pandas.
rename_dict = {
    'Strike': 'Strike',
    'Bid_Call': 'Bid', 'Ask_Call': 'Ask', 'Mid_Call': 'Mid', 'ImpliedVol_Call': 'IV',
    'Delta_%_Call': 'Δ (%)', 'Delta_EUR_Call': 'Δ (€)', 'Gamma_%_Call': 'Γ (%)', 'Gamma_EUR_Call': 'Γ (€)',
    'Vega_%_Call': '𝒱 (%)', 'Vega_EUR_Call': '𝒱 (€)', 'Theta_%_Call': 'Θ (%)', 'Theta_EUR_Call': 'Θ (€)',
    'Bid_Put': 'Bid ', 'Ask_Put': 'Ask ', 'Mid_Put': 'Mid ', 'ImpliedVol_Put': 'IV ',
    'Delta_%_Put': 'Δ (%) ', 'Delta_EUR_Put': 'Δ (€) ', 'Gamma_%_Put': 'Γ (%) ', 'Gamma_EUR_Put': 'Γ (€) ',
    'Vega_%_Put': '𝒱 (%) ', 'Vega_EUR_Put': '𝒱 (€) ', 'Theta_%_Put': 'Θ (%) ', 'Theta_EUR_Put': 'Θ (€) '
}
chain_renamed = chain.rename(columns=rename_dict)

# ── 6. FORMATAGE EN TEXTE PUR (ANTI-STREAMLIT NONE) ───────────────────────────
def format_to_string(val, fmt_string):
    """Convertit brutalement la valeur en texte. Si vide, renvoie un tiret strict."""
    # On traque toutes les formes possibles de vide (NaN, None, "<NA>")
    if pd.isna(val) or val is None or str(val).lower().strip() in ['nan', 'none', '<na>', '']:
        return "-"
    try:
        return fmt_string.format(float(val))
    except (ValueError, TypeError):
        return "-"

# Dictionnaire des formats pour chaque colonne
format_dict = {
    'Strike': '{:.1f}',
    'Bid': '{:.2f} €', 'Ask': '{:.2f} €', 'Mid': '{:.2f} €', 'IV': '{:.2%}',
    'Δ (%)': '{:.1%}', 'Γ (%)': '{:.3f}', '𝒱 (%)': '{:.2f}', 'Θ (%)': '{:.2f}',
    'Δ (€)': '{:.0f} €', 'Γ (€)': '{:.0f} €', '𝒱 (€)': '{:.0f} €', 'Θ (€)': '{:.0f} €',
    
    # Colonnes Puts (avec l'espace invisible)
    'Bid ': '{:.2f} €', 'Ask ': '{:.2f} €', 'Mid ': '{:.2f} €', 'IV ': '{:.2%}',
    'Δ (%) ': '{:.1%}', 'Γ (%) ': '{:.3f}', '𝒱 (%) ': '{:.2f}', 'Θ (%) ': '{:.2f}',
    'Δ (€) ': '{:.0f} €', 'Γ (€) ': '{:.0f} €', '𝒱 (€) ': '{:.0f} €', 'Θ (€) ': '{:.0f} €'
}

# On écrase les nombres par des chaînes de caractères définitives
for col, fmt in format_dict.items():
    if col in chain_renamed.columns:
        chain_renamed[col] = chain_renamed[col].apply(lambda x: format_to_string(x, fmt))

# ── 7. APPLICATION DU STYLING GRAPHIQUE ───────────────────────────────────────
strikes_arr = chain['Strike'].dropna().to_numpy()
if len(strikes_arr) > 0:
    atm_strike = strikes_arr[np.abs(strikes_arr - spot_price).argmin()]
else:
    atm_strike = None

def style_desk_matrix(df_matrix):
    styles = np.full(df_matrix.shape, '', dtype=object)
    strike_idx = 12  # Le Strike est la 13ème colonne (index 12)

    for r_idx in range(df_matrix.shape[0]):
        strike_val = df_matrix.iloc[r_idx, strike_idx]
        
        # Sécurité : on ignore si le strike a été transformé en tiret
        if strike_val == "-":
            continue
            
        current_strike = float(strike_val)
        is_atm = (current_strike == atm_strike)
        
        for c_idx in range(df_matrix.shape[1]):
            if is_atm:
                styles[r_idx, c_idx] = 'background-color: rgba(255, 193, 7, 0.35); color: black; font-weight: bold; border-top: 1px solid #ffc107; border-bottom: 1px solid #ffc107;'
            else:
                if c_idx == strike_idx:
                    styles[r_idx, c_idx] = 'background-color: #e9ecef; font-weight: bold; color: #495057; text-align: center;'
                elif c_idx < strike_idx:
                    styles[r_idx, c_idx] = 'background-color: rgba(52, 152, 219, 0.04); color: #2c3e50;'
                else:
                    styles[r_idx, c_idx] = 'background-color: rgba(231, 76, 60, 0.04); color: #2c3e50;'

    return pd.DataFrame(styles, index=df_matrix.index, columns=df_matrix.columns)

# Plus besoin de .format() ici car tout est déjà converti en texte dans le dataframe !
formatted_chain = chain_renamed.style.apply(style_desk_matrix, axis=None)

st.dataframe(formatted_chain, use_container_width=True, height=650)
st.caption("🔵 Sections Bleues : Options Calls | ⚪ Axe Central : Strikes | 🔴 Sections Rouges : Options Puts. "
           "La ligne jaune indique le niveau At-The-Money (ATM). Les tirets (-) indiquent une absence de cotation au marché.")