# risk_system

Système de pricing d'options sur l'**Euro Stoxx 50 (SX5E)** et ses 50 composants.  
Projet étudiant — Albert School, cours Algo Trading.

---

## Architecture

```
risk_system/
├── src/risk_system/
│   ├── config.py          # Dataclass Settings (IBKR, marché, chemins)
│   ├── pricing.py         # BS : norm_cdf/pdf, forward, d1/d2, bs_call/put
│   ├── greeks.py          # Delta, Gamma, Vega, Theta, Rho, DollarGamma, P&L attribution
│   ├── implied_vol.py     # Newton-Raphson + Brent (fallback), bornes arbitrage
│   ├── market_data.py     # Extraction IBKR batch, filtre delta ±30%, parquet
│   └── surface.py         # Nappe de vol en variance totale (griddata)
├── scripts/
│   └── fetch_data.py      # CLI extraction (argparse, --dry-run)
├── app/
│   ├── Accueil.py         # Page 1 — Option Chain style salle de marché
│   └── pages/
│       ├── 2_Parametres.py            # Page 2 — Paramètres & recalcul IV
│       └── 3_Nappe_de_Volatilite.py   # Page 3 — Surface 3D Plotly
├── tests/
│   ├── test_pricing.py    # Parité put-call, cas limites
│   ├── test_greeks.py     # Grecs analytiques vs différences finies
│   └── test_implied_vol.py # Round-trip BS → IV → BS
├── data/
│   ├── reference/
│   │   └── euro_stoxx_50_tickers.xlsx
│   └── parquet/           # Snapshots horodatés (gitignored)
└── docs/
    └── industrial_roadmap_volatility_infrastructure_v4.pdf
```

---

## Installation

```bash
git clone <repo-url>
cd risk_system

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .
```

---

## Récupération des données

**Prérequis :** IB Gateway ou TWS actif sur `127.0.0.1:4002` (paper ou live).

```bash
# Tous les symboles (SX5E + 50 composants Euro Stoxx 50)
python scripts/fetch_data.py

# Symboles spécifiques
python scripts/fetch_data.py --symbols SX5E ASML.AS LVMH.PA

# Limiter les maturités
python scripts/fetch_data.py --expiries-max 4

# Simuler sans connexion (test de configuration)
python scripts/fetch_data.py --dry-run
```

Les snapshots sont sauvegardés dans `data/parquet/snapshot_YYYYMMDD_HHMMSS.parquet`.

---

## Lancement de l'application

```bash
streamlit run app/Accueil.py
```

Naviguez entre les trois pages via la barre latérale Streamlit :

| Page | Contenu |
|------|---------|
| **Option Chain** | Tableau calls \| strike \| puts avec IV, Δ, Γ, V, Θ ; ATM surligné |
| **Paramètres** | Sliders r / q / base theta, recalcul IV + grecs depuis les prix Mid |
| **Nappe de Volatilité** | Surface 3D interactive (Plotly), toggle log-moneyness / strike |

---

## Tests

```bash
pytest tests/ -v
```

---

## Méthodologie

### Black-Scholes (Eq. 10–11, roadmap)

```
C = S·e^(-qT)·N(d1) − K·e^(-rT)·N(d2)
P = K·e^(-rT)·N(−d2) − S·e^(-qT)·N(−d1)

d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T
```

`norm_cdf` implémentée *from scratch* via `math.erf` (pas de dépendance scipy).

### Grecs — Conventions

| Grec | Formule | Convention |
|------|---------|-----------|
| Delta | e^(-qT)·N(±d1) | unité |
| Gamma | e^(-qT)·n(d1) / (S·σ·√T) | unité |
| **Vega** | S·e^(-qT)·n(d1)·√T / 100 | **par 1% de vol** |
| **Theta** | ∂V/∂t / 365 | **par jour calendaire** |
| **Rho** | K·T·e^(-rT)·N(±d2) × 0.01 | **par 1% de taux** |
| DollarGamma | Γ × S² × mult | mult=10 pour SX5E |
| DollarVega | V × mult | mult=10 pour SX5E |

### Volatilité Implicite

1. **Newton-Raphson** : `σ₀ = 0.5`, tolérance `1e-6`, max 100 itérations, borne `σ ∈ [0.01, 2.0]`.
2. **Brent** (fallback) via `scipy.optimize.brentq` sur `[1e-4, 3.0]` — robuste pour vega ≈ 0 (options très OTM).
3. Vérification des **bornes d'arbitrage** avant inversion : `intrinsèque ≤ prix ≤ borne supérieure`.

### Filtre Delta ±30%

Seules les options vérifiant `|Δ| ≤ 0.30` sont conservées dans le snapshot.  
**Pré-filtrage** des strikes (avant toute requête réseau) sur la fenêtre :
```
[S·e^(−2·σ_ref·√T), S·e^(+2·σ_ref·√T)] ∩ [0.8·S, 1.2·S]
```
avec `σ_ref = 25%`.

### Nappe de Volatilité

Interpolation en **variance totale** `w = σ²T` (meilleure régularité analytique, absence d'arbitrage calendaire) via `scipy.interpolate.griddata` (linéaire à l'intérieur, *nearest* aux bords).  
Reconversion : `σ(k, T) = √(w / T)`.
