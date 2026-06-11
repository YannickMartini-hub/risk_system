# CLAUDE.md — risk_system

Contexte complet du projet pour Claude Code.
Lis ce fichier en entier avant toute intervention.

---

## Projet

Cours **Algo Trading — Albert School**.  
Système de pricing d'options sur l'**Euro Stoxx 50 (SX5E)** et ses 23 composants
disponibles dans le fichier de référence.  
Backend données : **Interactive Brokers** via `ib_insync`, Gateway sur `127.0.0.1:4002`.  
Frontend : **Streamlit** multi-pages (lecture seule des parquets — jamais de connexion IBKR dans l'app).

---

## Architecture

```
risk_system/
├── src/risk_system/          ← package Python (pip install -e .)
│   ├── __init__.py
│   ├── config.py             ← Settings dataclass (singleton SETTINGS)
│   ├── pricing.py            ← Black-Scholes scalaire + numpy vectorisé
│   ├── greeks.py             ← tous les grecs + all_greeks() + pnl_attribution()
│   ├── implied_vol.py        ← Newton-Raphson + Brent fallback
│   ├── market_data.py        ← extraction IBKR, parquet in/out
│   └── surface.py            ← nappe de vol (variance totale)
├── scripts/
│   └── fetch_data.py         ← CLI argparse (--symbols, --expiries-max, --dry-run)
├── app/
│   ├── Accueil.py            ← Page 1 : Option Chain
│   └── pages/
│       ├── 2_Parametres.py   ← Page 2 : Paramètres & recalcul IV
│       └── 3_Nappe_de_Volatilite.py  ← Page 3 : Surface 3D Plotly
├── tests/
│   ├── test_pricing.py       ← parité put-call, cas limites
│   ├── test_greeks.py        ← grecs analytiques vs FD (tol 1e-4)
│   └── test_implied_vol.py   ← round-trip BS→IV (tol 1e-6), bornes arbitrage
├── data/
│   ├── reference/
│   │   └── euro_stoxx_50_tickers.xlsx   ← 23 tickers + en-tête "Sigle"
│   └── parquet/              ← snapshots horodatés (gitignored)
├── docs/
│   └── industrial_roadmap_volatility_infrastructure_v4.pdf
├── pyproject.toml            ← build-backend = setuptools.build_meta
├── requirements.txt
└── .gitignore                ← ignore data/parquet/, __pycache__, .venv/
```

---

## Formules mathématiques de référence

Source : `base.ipynb` (supprimé) + roadmap PDF.
**Ne jamais modifier les conventions** sans raison explicite.

### Black-Scholes (Eq. 10–11)

```
d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T

C = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
P = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

`norm_cdf` implémentée **from scratch** via `math.erf` (pas scipy) :
```python
0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
```

### Grecs — conventions IMPÉRATIVES

| Grec | Convention | Facteur |
|------|-----------|---------|
| Delta | e^(−qT)·N(±d1) | — |
| Gamma | e^(−qT)·n(d1) / (S·σ·√T) | — |
| **Vega** | S·e^(−qT)·n(d1)·√T | **/ 100** (par 1% de vol) |
| **Theta** | formule standard | **/ 365** (par jour calendaire) |
| **Rho** | K·T·e^(−rT)·N(±d2) | **× 0.01** (par 1% de taux) |
| DollarGamma | Γ × S² × multiplier | mult=10 pour SX5E |
| DollarVega | Vega × multiplier | mult=10 pour SX5E |

`pnl_attribution` : `vega_pnl = v * d_sigma * 100.0`
(vega est par 1%, d_sigma est en absolu → on convertit en points de vol).

### compute_d2 — bug historique corrigé

Dans `base.ipynb`, `compute_d2(d1, sigma)` référençait `T` depuis la portée
globale (bug silencieux en REPL). Dans `pricing.py`, T est un paramètre explicite :
```python
def compute_d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)
```
Ne jamais revenir à l'ancienne signature.

---

## Module par module

### `config.py`

```python
SETTINGS = Settings()   # singleton importable directement
```

Champs clés :
- `host="127.0.0.1"`, `port=4002`, `client_id=1`
- `r=0.025`, `q=0.0`
- `DELTA_MIN=-0.30`, `DELTA_MAX=0.30`
- `MULTIPLIER=10` (SX5E Eurex)
- `T_BASIS=365.25` (ACT/365.25 pour annualiser T)
- `THETA_BASIS=365.0` (base theta par jour calendaire)
- Chemins pathlib relatifs à la racine du projet (`Path(__file__).parents[2]`)

### `pricing.py`

Fonctions scalaires (module `math`) :
`norm_cdf`, `norm_pdf`, `forward`, `log_moneyness`, `total_variance`,
`compute_d1`, `compute_d2`, `bs_call`, `bs_put`, `bs_price(right='C'|'P')`

Fonctions vectorisées numpy :
`norm_cdf_vec` (via `scipy.special.ndtr`), `norm_pdf_vec`,
`bs_call_vec`, `bs_put_vec`

### `greeks.py`

Toutes les fonctions prennent `(S, K, T, r, q, sigma, right)`.
`gamma` et `vega` ne prennent pas `right` (identiques C et P).

```python
all_greeks(S, K, T, r, q, sigma, right, multiplier=1.0) -> dict
# clés : Delta, Gamma, Vega, Theta, Rho, DollarGamma, DollarVega
```

### `implied_vol.py`

```python
implied_vol(S, K, T, r, q, market_price, right) -> float | None
```
1. Vérifie bornes arbitrage : `intrinsèque ≤ prix ≤ borne sup`
2. Newton-Raphson : σ₀=0.5, tol=1e-6, max 100 iter, bornes [0.01, 2.0]
3. Brent fallback : `scipy.brentq` sur [1e-4, 3.0]
4. Retourne `None` proprement si pas de solution

### `market_data.py`

**SX5E** : `Index("ESTX50", "EUREX", "EUR")` + options `Option("ESTX50", ..., exchange="EUREX", multiplier="10")`  
**Composants** : `Stock(symbol, "SMART", "EUR")`

Pré-filtrage strikes avant requêtes :
```python
fenêtre = [max(S·e^(−2·0.25·√T), 0.80·S), min(S·e^(+2·0.25·√T), 1.20·S)]
```

Batch : lots de 50 contrats, `ib.sleep(2.0)` après chaque lot, `cancelMktData` après lecture.  
Connexion unique réutilisée pour tous les symboles.  
Sorties : `save_snapshot()`, `load_latest_snapshot()`, `list_snapshots()` dans `data/parquet/`.

Colonnes du parquet :
`Symbol, Spot, Strike, Maturity (YYYY-MM-DD), T, Type (C/P),`
`Bid, Ask, Mid, ImpliedVol, Delta, Gamma, Vega, Theta, Rho`

### `surface.py`

```python
build_surface(df, symbol, n_grid=50) -> (X, Y, Z)
# X = log-moneyness (n_grid×n_grid)
# Y = maturité en années
# Z = vol implicite en décimal
```
Interpolation en **variance totale** `w = σ²T`, `griddata` linéaire + nearest (bords),
reconversion `σ = √(w/T)`.

---

## Scripts & App

### `scripts/fetch_data.py`

```bash
python scripts/fetch_data.py                        # tous les symboles
python scripts/fetch_data.py --symbols SX5E AIR ALV # symboles spécifiques
python scripts/fetch_data.py --expiries-max 4       # limite maturités
python scripts/fetch_data.py --dry-run              # sans connexion IBKR
```

`_load_tickers()` : `pd.read_excel(..., header=0)` — la première ligne du fichier
est l'en-tête "Sigle", pas un ticker. Déduplication en conservant l'ordre.

### App Streamlit

```bash
streamlit run app/Accueil.py
```

- **Page 1** `Accueil.py` : option chain calls|strike|puts, IV en %, grecs,
  ATM surligné en jaune (`style.apply`), sélecteur snapshot/symbole/maturité.
- **Page 2** `2_Parametres.py` : sliders r/q/base theta, recalcul complet IV + grecs
  depuis les prix Mid stockés, résultats en tableau. Les paramètres sont stockés dans
  `st.session_state["r"]`, `["q"]`, `["day_base"]`.
- **Page 3** `3_Nappe_de_Volatilite.py` : `plotly.graph_objects.Surface` colorscale
  Viridis, toggle axe X log-moneyness/strike, scatter3d points bruts optionnel.

L'app **ne se connecte jamais à IBKR** : elle lit uniquement les parquets via
`@st.cache_data`.

---

## Tests

```bash
pytest tests/ -v     # 125 tests, tous verts
```

| Fichier | Ce qui est testé |
|---------|-----------------|
| `test_pricing.py` | Parité put-call (tol 1e-8), dispatcher, deep ITM/OTM, monotonie, forward |
| `test_greeks.py` | Delta/Gamma/Vega/Theta/Rho analytiques vs FD centrée (tol 1e-4), all_greeks, DollarGamma, convention vega/1% |
| `test_implied_vol.py` | Round-trip Newton (tol 1e-5) et Newton+Brent (tol 1e-6) sur grille 4×4×2, deep OTM/ITM, T court, prix hors bornes → None |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e .
pip install scipy pandas pyarrow pytest   # si non installés par pip install -e .
```

`pyproject.toml` utilise `setuptools.build_meta` (compatible avec toutes les
versions de setuptools, y compris le venv Python 3.13 existant).

---

## Points d'attention pour les futures modifications

1. **Conventions grecs** : vega/1%, theta/365, rho/1% — ne pas changer.
2. **compute_d2** : toujours passer T explicitement.
3. **SX5E** : contrat `Index("ESTX50", "EUREX", "EUR")`, multiplier 10,
   options sur `"EUREX"`. Pas un `Stock`.
4. **Aucun `print`** dans `src/risk_system/` — uniquement `logging`.
5. **Aucun chemin absolu** — tout via `pathlib` et `SETTINGS`.
6. **App Streamlit** : aucune logique métier dans l'app, elle appelle uniquement
   `src/risk_system`. Aucune connexion IBKR.
7. **data/parquet/** est gitignored — ne pas committer de snapshots.
8. **Excel tickers** : 23 tickers (pas 50 — le fichier est partiel), en-tête
   "Sigle" en ligne 0 → `header=0` dans `pd.read_excel`.
