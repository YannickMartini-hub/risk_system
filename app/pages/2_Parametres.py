"""
Page 2 — Paramètres & Recalcul.

Sliders pour r, q et base de calcul.
Au changement, recalcule vols implicites + grecs à partir des prix Mid stockés.
Démontre : retrouver σ qui price l'option, puis les grecs.
Les paramètres sont stockés dans st.session_state pour les autres pages.
"""

from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from risk_system.implied_vol import implied_vol
from risk_system.greeks import all_greeks
from risk_system.market_data import list_snapshots
from risk_system.config import SETTINGS

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

st.set_page_config(page_title="Risk System — Paramètres", layout="wide")
st.title("Paramètres de marché & Recalcul")

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


df_raw = _load(str(selected_path))

# ── sélecteurs de sous-jacent / maturité ──────────────────────────────────────
symbols    = sorted(df_raw["Symbol"].unique())
symbol     = st.sidebar.selectbox("Sous-jacent", symbols)
df_sym     = df_raw[df_raw["Symbol"] == symbol]
maturities  = sorted(df_sym["Maturity"].unique())
mat_opts    = _maturity_options(maturities)
mat_labels  = [lbl for lbl, _ in mat_opts]
mat_dates   = [dt  for _, dt  in mat_opts]
mat_label   = st.sidebar.selectbox("Maturité", mat_labels)
maturity    = mat_dates[mat_labels.index(mat_label)]

# ── paramètres ────────────────────────────────────────────────────────────────
st.header("Paramètres de calcul")
col1, col2, col3 = st.columns(3)

with col1:
    r_pct = st.slider(
        "Taux sans risque r (%)", 0.0, 10.0,
        float(st.session_state.get("r", SETTINGS.r)) * 100,
        step=0.1,
    )
    r = r_pct / 100.0

with col2:
    q_pct = st.slider(
        "Dividend yield q (%)", 0.0, 5.0,
        float(st.session_state.get("q", SETTINGS.q)) * 100,
        step=0.1,
    )
    q = q_pct / 100.0

with col3:
    day_base = st.selectbox(
        "Base theta (jours/an)", [365, 252, 360],
        index=[365, 252, 360].index(int(st.session_state.get("day_base", 365))),
    )

# Persistance dans session_state pour les autres pages
st.session_state["r"]        = r
st.session_state["q"]        = q
st.session_state["day_base"] = day_base

st.caption(f"r = **{r*100:.2f}%**  |  q = **{q*100:.2f}%**  |  base theta = **{day_base} j/an**")

# ── recalcul ──────────────────────────────────────────────────────────────────
st.header(f"Options recalculées — {symbol} / {mat_label}")

df = df_sym[df_sym["Maturity"] == maturity].copy()
if df.empty:
    st.info("Aucune option pour cette sélection.")
    st.stop()

multiplier = SETTINGS.MULTIPLIER if symbol == "SX5E" else 1
results: list[dict] = []

for _, row in df.iterrows():
    iv = implied_vol(float(row.Spot), float(row.Strike), float(row.T), r, q, float(row.Mid), row.Type)
    if iv is None:
        continue
    g = all_greeks(
        float(row.Spot), float(row.Strike), float(row.T), r, q, iv,
        right=row.Type, multiplier=multiplier,
    )
    # Theta recalculé selon la base choisie
    theta_rebased = g["Theta"] * 365.0 / day_base

    results.append({
        "Type":    row.Type,
        "Strike":  int(row.Strike),
        "Mid":     round(float(row.Mid), 4),
        "IV %":    round(iv * 100, 2),
        "Delta":   round(g["Delta"], 4),
        "Gamma":   round(g["Gamma"], 6),
        "Vega":    round(g["Vega"],  4),
        "Theta":   round(theta_rebased, 4),
        "Rho":     round(g["Rho"],   4),
        "$ Gamma": round(g["DollarGamma"], 2),
        "$ Vega":  round(g["DollarVega"],  4),
    })

if results:
    df_out = pd.DataFrame(results).sort_values(["Type", "Strike"])
    st.dataframe(
        df_out,
        use_container_width=True,
        column_config={
            "Strike": st.column_config.NumberColumn("Strike", format="%d"),
            "Mid":    st.column_config.NumberColumn("Mid",    format="%.4f"),
            "IV %":   st.column_config.NumberColumn("IV %",   format="%.2f"),
            "Delta":  st.column_config.NumberColumn("Δ",      format="%.4f"),
            "Gamma":  st.column_config.NumberColumn("Γ",      format="%.6f"),
            "Vega":   st.column_config.NumberColumn("V",      format="%.4f"),
            "Theta":  st.column_config.NumberColumn("Θ",      format="%.4f"),
            "Rho":    st.column_config.NumberColumn("ρ",      format="%.4f"),
        },
    )
    st.caption(
        "IV recalculée de zéro depuis le prix Mid avec les paramètres ci-dessus. "
        "Theta exprimé sur la base sélectionnée."
    )
else:
    st.info("Aucune option recalculable avec ces paramètres (prix hors bornes d'arbitrage ?).")
