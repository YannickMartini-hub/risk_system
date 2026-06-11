from ib_insync import *
import pandas as pd
from datetime import date
import os


##Extraction des données

Excelsigle = r"C:\Users\bouil\Desktop\Cours\Data\Algo Trading\Euro Stoxx 50 - Sigles et Entreprises.xlsx"
OUTPUT_DIR = rf"C:\Users\bouil\Desktop\Cours\Data\Algo Trading\Parquet_{date.today()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Crée le dossier s'il n'existe pas

from ib_insync import *
import pandas as pd
from datetime import datetime

def get_options_data(symbol):
    """
    Récupère les données des options pour un symbole donné, incluant :
    - Spot (prix du sous-jacent)
    - Strike
    - Maturité
    - Type (Call/Put)
    - Prix du marché (Market_Price)

    Paramètres :
    - symbol : Symbole du sous-jacent (ex: "AAPL")

    Retourne :
    - DataFrame avec les colonnes : Spot, Strike, Maturité, Type, Market_Price
    """
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4002, clientId=2, timeout=10)
        print(f"Connecté à IBKR Gateway pour {symbol}")

        Lspot, Lstrike, Lmaturite, Lright, Lmarket_price, vega, rho, delta, gamma, theta = [], [], [], [], [], [], [], [], [], []

        # Récupérer le contrat du sous-jacent
        stock = Stock(symbol, 'SMART', 'EUR')
        contracts = ib.qualifyContracts(stock)

        if not contracts:
            print(f"Contrat introuvable pour {symbol}")
            return pd.DataFrame()  # Retourne un DataFrame vide

        stock = contracts[0]
        print(f"Contrat qualifié : {stock}")

        # Récupération du spot via historique
        bars = ib.reqHistoricalData(
            stock,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        if bars:
            spot = bars[-1].close
            print(f"Spot ({symbol}): {spot}")
        else:
            print(f"Aucun historique disponible pour {symbol}")
            return pd.DataFrame()

        # Récupérer les paramètres des options
        options_params = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        ib.sleep(1)  # Attendre la réception des données

        if not options_params:
            print(f"Aucune option trouvée pour {symbol}")
            return pd.DataFrame()

        print(f"{len(options_params)} ensembles d'options trouvés")

        # Récupérer les prix du marché pour chaque option
        for param in options_params:
            for expiry in param.expirations:
                for strike in param.strikes:
                    for right in ['C', 'P']:
                        # Créer le contrat de l'option
                        option_contract = Option(
                            symbol=symbol,
                            expiry=expiry,
                            strike=strike,
                            right=right,
                            exchange='SMART',
                            currency='EUR'
                        )
                        ib.qualifyContracts(option_contract)

                        # Récupérer le prix du marché
                        ticker = ib.reqMktData(option_contract, '', False, False)
                        ib.sleep(0.1)  # Éviter les limitations de taux de requêtes

                        # Extraire le prix (midpoint entre bid et ask si marketPrice n'est pas disponible)
                        if hasattr(ticker, 'marketPrice') and ticker.marketPrice is not None:
                            market_price = ticker.marketPrice
                        elif hasattr(ticker, 'bid') and hasattr(ticker, 'ask') and ticker.bid is not None and ticker.ask is not None:
                            market_price = (ticker.bid + ticker.ask) / 2
                        else:
                            market_price = None
                            print(f"Prix non disponible pour {symbol} {right}{strike} {expiry}")

                        # Ajouter les données à la liste
                        Lspot.append(spot)
                        Lstrike.append(strike)
                        Lmaturite.append(expiry)
                        Lright.append(right)
                        Lmarket_price.append(market_price)
                        delta.append(None)
                        gamma.append(None)
                        theta.append(None)
                        vega.append(None)
                        rho.append(None)

        # Créer le DataFrame
        df_options = pd.DataFrame({
            'Spot': Lspot,
            'Strike': Lstrike,
            'Maturité': Lmaturite,
            'Type': Lright,
            'Market_Price': Lmarket_price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,  # Par jour
            "Vega": vega,    # Pour 1% de volatilité
            "Rho": rho       # Pour 1% de taux

        })

        # Supprimer les lignes où le prix du marché est manquant
        df_options = df_options.dropna(subset=['Market_Price'])

        print(f"✅ {symbol} : {len(df_options)} lignes enregistrées avec prix du marché")

    except Exception as e:
        print(f"Erreur pour {symbol}: {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur
    finally:
        ib.disconnect()
    return df_options

## Calcule des grecs

import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, M, r, sigma, q=0, option_type="C"):
    """
    Calcule les Grecs pour une option européenne (Black-Scholes).

    Paramètres :
    - S : Prix du sous-jacent
    - K : Strike
    - T : Temps jusqu'à l'échéance (en années)
    - r : Taux sans risque (annuel)
    - sigma : Volatilité implicite (annuelle)
    - q : Taux de dividende (annuel, défaut=0)
    - option_type : "Call" ou "Put"
    """
    #convertion de la maturité en temps jusqu'a echéance :
    M = M[0:4]+"-"+M[4:6]+"-"+M[6:8]
    Mat_date = (datetime.strptime(M,"%Y-%m-%d") - datetime.now()).days
    T = Mat_date/365.25

    # Calcul de d1 et d2
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Fonctions normales
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_prime_d1 = norm.pdf(d1)  # Densité normale

    # Grecs
    if option_type == "C":
        delta = np.exp(-q * T) * N_d1
        gamma = np.exp(-q * T) * N_prime_d1 / (S * sigma * np.sqrt(T))
        theta = (-S * np.exp(-q * T) * N_prime_d1 * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * N_d2 -
                 q * S * np.exp(-q * T) * N_d1) / 365  # Thêta par jour
        vega = S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T) * 0.01  # Vega pour 1% de volatilité
        rho = K * T * np.exp(-r * T) * N_d2 * 0.01  # Rho pour 1% de taux
    elif option_type == "P":
        delta = np.exp(-q * T) * (N_d1 - 1)
        gamma = np.exp(-q * T) * N_prime_d1 / (S * sigma * np.sqrt(T))
        theta = (-S * np.exp(-q * T) * N_prime_d1 * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2) +
                 q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        vega = S * np.exp(-q * T) * N_prime_d1 * np.sqrt(T) * 0.01
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
    else:
        raise ValueError("option_type doit être 'Call' ou 'Put'")

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,  # Par jour
        "Vega": vega,    # Pour 1% de volatilité
        "Rho": rho       # Pour 1% de taux
    }

# Exemple d'utilisation
#S = 100      # Prix du sous-jacent
#K = 105      # Strike
#T = 0.5      # 6 mois jusqu'à l'échéance
#r = 0.02     # Taux sans risque 2%
#sigma = 0.25 # Volatilité implicite 25%
#q = 0.01     # Dividende 1%

#grecs_call = black_scholes_greeks(S, K, T, r, sigma, q, "Call")
#grecs_put = black_scholes_greeks(S, K, T, r, sigma, q, "Put")

#print("Grecs pour un Call :", grecs_call)
#print("Grecs pour un Put :", grecs_put)

##Calcule de Sigma

import numpy as np
from scipy.stats import norm

def implied_volatility_newton(S, K, M, r, market_price, q=0, option_type="C", max_iter=100, tol=1e-6):
    """
    Calcule la volatilité implicite (sigma) d'une option avec la méthode de Newton-Raphson.

    Paramètres :
    - S : Prix du sous-jacent (float)
    - K : Strike de l'option (float)
    - T : Temps jusqu'à l'échéance en années (float, ex: 0.5 pour 6 mois)
    - r : Taux sans risque annuel (float, ex: 0.02 pour 2%)
    - market_price : Prix de marché de l'option (float)
    - q : Taux de dividende annuel (float, défaut=0)
    - option_type : Type d'option ("Call" ou "Put", défaut="Call")
    - max_iter : Nombre maximal d'itérations (int, défaut=100)
    - tol : Tolérance pour la convergence (float, défaut=1e-6)

    Retourne :
    - sigma : Volatilité implicite en décimal (ex: 0.25 pour 25%)
    - None : Si la méthode ne converge pas
    """
    sigma = 0.5  # Valeur initiale pour sigma

    #convertion de la maturité en temps jusqu'a echéance :
    M = M[0:4]+"-"+M[4:6]+"-"+M[6:8]
    Mat_date = (datetime.strptime(M,"%Y-%m-%d") - datetime.now()).days
    T = Mat_date/365.25

    for _ in range(max_iter):
        # Calcul de d1 et d2
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calcul du prix théorique de l'option
        if option_type == "C":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # Put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        # Calcul du Vega (dérivée du prix par rapport à sigma)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Différence entre le prix théorique et le prix de marché
        diff = price - market_price

        # Vérification de la convergence
        if abs(diff) < tol:
            return sigma

        # Mise à jour de sigma avec la méthode de Newton-Raphson
        sigma = sigma - diff / vega

        # Vérification que sigma reste dans un intervalle raisonnable
        if sigma <= 0.01 or sigma >= 2.0:
            break

    return None  # La méthode n'a pas convergé

# Fonction principale
def main():
    df = pd.read_excel(Excelsigle)
    symbols = df.iloc[:, 0].tolist()

    for symbol in symbols:
        df_options = get_options_data(symbol)
        if df_options.empty:
            continue

        # Ajouter les colonnes pour les Grecs
        df_options[['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Sigma']] = pd.NA

        for index, row in df_options.iterrows():
            S = row['Spot']
            K = row['Strike']
            M = row['Maturité']
            option_type = row['Type']
            r = 0.025
            market_price = row['Market_Price']
            q = 0.0

            # Calculer la volatilité implicite
            sigma = implied_volatility_newton(S, K, M, r, market_price, q, option_type)
            if sigma is None:
                print(f"Impossible de calculer sigma pour {symbol} {option_type}{K} {M}")
                continue

            # Calculer les Grecs
            grecs = black_scholes_greeks(S, K, M, r, sigma, q, option_type)

            # Mettre à jour le DataFrame
            df_options.at[index, 'Delta'] = grecs['Delta']
            df_options.at[index, 'Gamma'] = grecs['Gamma']
            df_options.at[index, 'Theta'] = grecs['Theta']
            df_options.at[index, 'Vega'] = grecs['Vega']
            df_options.at[index, 'Rho'] = grecs['Rho']
            df_options.at[index, 'Sigma'] = sigma

        # Sauvegarder le DataFrame
        safe_symbol = symbol.replace('.', '_')
        filepath = os.path.join(OUTPUT_DIR, f"Data_{safe_symbol}_{date.today()}.parquet")
        df_options.to_parquet(filepath, engine="pyarrow", index=False)
        print(f"Fichier sauvegardé : {filepath}")

if __name__ == "__main__":
    main()


print("HW")
main