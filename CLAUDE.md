# CLAUDE.md — risk_system

Contexte complet du projet pour Claude Code.
Lis ce fichier en entier avant toute intervention.

---

## Projet

Cours **Algo Trading — Albert School**.  
Système de pricing d'options sur le **S&P 500 (SPX)** et ses 50 plus grandes
composantes (top 50 par capitalisation boursière).  
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
│   ├── greeks.py             ← grecs + all_greeks() + pnl_attribution() + aggregate_greeks()
│   ├── implied_vol.py        ← Newton-Raphson + Brent fallback
│   ├── market_data.py        ← extraction IBKR, parquet in/out, forward put-call parity
│   ├── surface.py            ← nappe de vol (variance totale, log-moneyness forward-based)
│   ├── svi.py                ← fit SVI paramétrique par tranche + save/load params
│   └── qc.py                 ← contrôle qualité snapshots (5 checks + run_all_checks)
├── scripts/
│   ├── fetch_data.py         ← CLI argparse (--symbols, --expiries-max, --dry-run)
│   └── create_universe.py    ← script one-shot pour générer sp500_top50_universe.xlsx
├── app/
│   ├── Accueil.py            ← Page 1 : Option Chain (filtre delta 0.20–0.80)
│   └── pages/
│       ├── 2_Parametres.py   ← Page 2 : Paramètres & recalcul IV
│       └── 3_Nappe_de_Volatilite.py  ← Page 3 : Surface 3D + overlay SVI
├── tests/
│   ├── test_pricing.py       ← parité put-call, cas limites
│   ├── test_greeks.py        ← grecs analytiques vs FD (tol 1e-4)
│   └── test_implied_vol.py   ← round-trip BS→IV (tol 1e-6), bornes arbitrage
├── data/
│   ├── reference/
│   │   └── sp500_top50_universe.xlsx   ← 51 lignes : SPX + top 50 (Ticker, Company, Sector…)
│   └── parquet/              ← snapshots horodatés (gitignored)
│       ├── snapshot_YYYYMMDD_HHMMSS.parquet
│       └── svi_params_YYYYMMDD_HHMMSS.parquet
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

**Note importante** : Black-Scholes européen utilisé pour TOUT, y compris les options
sur actions US (style américain). C'est une approximation intentionnelle et explicite,
acceptable pour les grecs de premier ordre et les maturités courtes.

### Grecs — conventions IMPÉRATIVES

| Grec | Convention | Facteur |
|------|-----------|---------|
| Delta | e^(−qT)·N(±d1) | — |
| Gamma | e^(−qT)·n(d1) / (S·σ·√T) | — |
| **Vega** | S·e^(−qT)·n(d1)·√T | **/ 100** (par 1% de vol) |
| **Theta** | formule standard | **/ 365** (par jour calendaire) |
| **Rho** | K·T·e^(−rT)·N(±d2) | **× 0.01** (par 1% de taux) |
| DollarDelta | Δ × S × multiplier | mult=100 pour SPX/US |
| DollarGamma | Γ × S² × multiplier | mult=100 pour SPX/US |
| DollarVega | Vega × multiplier | mult=100 pour SPX/US |
| DollarTheta | Theta × multiplier | mult=100 pour SPX/US |

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

### SVI (Stochastic Volatility Inspired)

```
w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
```
où `k = ln(K/F)` (log-moneyness forward-based), `w = σ²T` (variance totale).

Contraintes : `b > 0`, `|rho| < 1`, `sigma > 0`, `a + b*sigma*(1-|rho|) >= 0`.

---

## Module par module

### `config.py`

```python
SETTINGS = Settings()   # singleton importable directement
```

Champs clés :
- `host="127.0.0.1"`, `port=4002`, `client_id=1`
- `r=0.045`, `q=0.0`  ← taux US (Fed funds / 10Y ~ 4.5%)
- `DELTA_ABS_MIN=0.20`, `DELTA_ABS_MAX=0.80`
- `MULTIPLIER=100` (SPX + US stocks, OCC)
- `T_BASIS=365.25` (ACT/365.25 pour annualiser T)
- `THETA_BASIS=365.0` (base theta par jour calendaire)
- `tickers_file` → `data/reference/sp500_top50_universe.xlsx`
- Chemins pathlib relatifs à la racine du projet (`Path(__file__).parents[2]`)

### `pricing.py`

Fonctions scalaires (module `math`) :
`norm_cdf`, `norm_pdf`, `forward`, `log_moneyness`, `total_variance`,
`compute_d1`, `compute_d2`, `bs_call`, `bs_put`, `bs_price(right='C'|'P')`

Fonctions vectorisées numpy :
`norm_cdf_vec` (via `scipy.special.ndtr`), `norm_pdf_vec`,
`bs_call_vec`, `bs_put_vec`

### `greeks.py`

Toutes les fonctions scalaires prennent `(S, K, T, r, q, sigma, right)`.
`gamma` et `vega` ne prennent pas `right` (identiques C et P).

```python
all_greeks(S, K, T, r, q, sigma, right, multiplier=1.0) -> dict
# clés : Delta, Gamma, Vega, Theta, Rho, DollarGamma, DollarVega

aggregate_greeks(positions, snapshot, multiplier=100) -> dict
# positions : [{'symbol', 'strike', 'maturity', 'type', 'quantity'}, ...]
# clés résultat : DollarDelta, DollarGamma, DollarVega, DollarTheta
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

**SPX** : `Index("SPX", "CBOE", "USD")` + options sur CBOE (tradingClass="SPX"), multiplier=100.
**Actions US** : `Stock(symbol, "SMART", "USD")`.
**BRK.B** : mapping `_IBKR_SYMBOL_MAP = {"BRK.B": "BRK B"}` → `Stock("BRK B", "SMART", "USD")`.

Sélection paramètre d'options :
- SPX → préférer CBOE + tradingClass="SPX" (AM-settled mensuel)
- Stocks → n'importe quel exchange non-SMART (OCC)

Heures de marché : `_in_us_hours()` → lun-ven 09h30-16h00 ET (`ZoneInfo("America/New_York")`).

Batch : lots de 50 contrats, `ib.sleep(2.0)` après chaque lot, `cancelMktData` après lecture.
Prix : bid/ask mid (si spread ≤ 100%) → last → close.
Champ `RefType` : `"bid_ask"` / `"last"` / `"close"`.

Forward par parité put-call :
```
F(T) ≈ K + e^(rT) * (C_mid - P_mid)
```
Calculé sur les 5 strikes les plus proches de l'ATM avec calls ET puts disponibles.
Stocké dans colonne `Forward` du snapshot. Fallback : `Forward = Spot`.

Colonnes du parquet :
`Symbol, Spot, Strike, Maturity (YYYY-MM-DD), T, Type (C/P),`
`Bid, Ask, Mid, Volume, RefType, ImpliedVol,`
`Delta, Gamma, Vega, Theta, Rho,`
`DollarDelta, DollarGamma, DollarVega, DollarTheta, Forward`

### `surface.py`

```python
build_surface(df, symbol, n_grid=50) -> (X, Y, Z)
# X = log-moneyness k=ln(K/F)  (n_grid×n_grid) — forward-based
# Y = maturité en années
# Z = vol implicite en décimal
```
Fallback sur Spot si colonne Forward absente (compatibilité anciens snapshots).
Interpolation en **variance totale** `w = σ²T`, `griddata` linéaire + nearest (bords),
reconversion `σ = √(w/T)`. Lissage Gaussien sigma=1.0.

### `svi.py`

```python
fit_svi(k_arr, w_arr) -> dict | None
# {'a', 'b', 'rho', 'm', 'sigma'} ou None si < 4 pts ou divergence

svi_variance(k, params) -> np.ndarray
svi_vol(k, T, params) -> np.ndarray

fit_svi_surface(df, symbol) -> pd.DataFrame
# Symbol | Maturity | T | a | b | rho | m | sigma_svi | n_points | rmse

save_svi_params(df_params) -> Path   # svi_params_YYYYMMDD_HHMMSS.parquet
load_latest_svi_params() -> pd.DataFrame
```
Optimisation L-BFGS-B avec contraintes de non-arbitrage.

### `qc.py`

```python
run_all_checks(df, r=0.045) -> dict[str, pd.DataFrame]
# clés : 'spread_pct', 'staleness', 'chain_coverage', 'put_call_parity', 'calendar_spread'
```
- `check_spread_pct` : spread > 50% du mid
- `check_staleness` : RefType == "close"
- `check_chain_coverage` : < 5 strikes par (Symbol, Maturity, Type)
- `check_put_call_parity` : |F/S - 1| > 5%
- `check_calendar_spread` : IV décroissante avec T (arbitrage calendaire)

---

## Scripts & App

### `scripts/fetch_data.py`

```bash
python scripts/fetch_data.py                          # tous (51 symboles)
python scripts/fetch_data.py --symbols SPX AAPL MSFT  # symboles spécifiques
python scripts/fetch_data.py --expiries-max 4         # limite maturités
python scripts/fetch_data.py --dry-run                # sans connexion IBKR
```

`_load_tickers()` : `pd.read_excel(..., header=0)` → colonne `Ticker`.
SPX est en ligne 1 du fichier Excel (pas de prepend manuel).

### `scripts/create_universe.py`

Script one-shot : génère `data/reference/sp500_top50_universe.xlsx`.
Colonnes : `Ticker, Company, Sector, SecType, Exchange, Currency, OptStyle, Note`.

### App Streamlit

```bash
streamlit run app/Accueil.py
```

- **Page 1** `Accueil.py` : option chain calls|strike|puts, IV en %, grecs en valeur et en $,
  **filtre delta 0.20 ≤ |Δ| ≤ 0.80** appliqué dans la couche display (parquet garde tout),
  ATM surligné en jaune, sélecteur snapshot/symbole/maturité (affichage tenor 1w/1m/3m…).
- **Page 2** `2_Parametres.py` : sliders r/q (défaut r=4.5%)/base theta, recalcul IV + grecs,
  paramètres stockés dans `st.session_state`.
- **Page 3** `3_Nappe_de_Volatilite.py` : `plotly.graph_objects.Surface` colorscale Viridis,
  log-moneyness forward-based k=ln(K/F), toggle axe X, overlay SVI (orange) si params dispo,
  scatter3d points bruts optionnel.

L'app **ne se connecte jamais à IBKR** : elle lit uniquement les parquets via `@st.cache_data`.

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
pip install scipy pandas pyarrow pytest openpyxl   # si non installés par pip install -e .
```

`pyproject.toml` utilise `setuptools.build_meta`.

---

## Points d'attention pour les futures modifications

1. **Conventions grecs** : vega/1%, theta/365, rho/1% — ne pas changer.
2. **compute_d2** : toujours passer T explicitement.
3. **SPX** : contrat `Index("SPX", "CBOE", "USD")`, multiplier 100.
   Options sur CBOE, tradingClass="SPX" pour les mensuels AM-settled.
4. **BRK.B** : symbole IBKR = `"BRK B"` (avec espace), pas `"BRK.B"`.
5. **Aucun `print`** dans `src/risk_system/` — uniquement `logging`.
6. **Aucun chemin absolu** — tout via `pathlib` et `SETTINGS`.
7. **App Streamlit** : aucune logique métier dans l'app, elle appelle uniquement
   `src/risk_system`. Aucune connexion IBKR.
8. **data/parquet/** est gitignored — ne pas committer de snapshots.
9. **Excel univers** : 51 lignes (SPX + 50 actions), colonne `Ticker` en header.
10. **Filtre delta** : appliqué dans la couche DISPLAY uniquement (Accueil.py).
    Le parquet garde tous les strikes — nécessaire pour la nappe de vol.
11. **Forward** : calculé par parité put-call dans fetch_symbol(), stocké dans `Forward`.
    surface.py utilise `Forward` si présent, sinon fallback `Spot`.
12. **SVI** : params stockés dans `svi_params_*.parquet` séparé du snapshot principal.
    Le fit est fait post-collecte, pas pendant la collecte.
