"""
Script one-shot : génère data/reference/euro_stoxx_50_universe.xlsx
SX5E (index) + les composants de l'Euro Stoxx 50.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_ROOT = Path(__file__).resolve().parents[1]

# Liste extraite de votre fichier CSV
UNIVERSE = [
    # Ticker, IBKR_Symbol, PrimaryExchange, Currency, Company, Country, Sector, OptStyle, SecType
    ("SX5E",     "SX5E",  "EUREX",    "EUR", "Euro Stoxx 50",         "Index",       "Index",                  "European", "IND"),
    ("ADS.DE",   "ADS",   "IBIS",     "EUR", "Adidas",                "Germany",     "Consumer Discretionary", "American", "STK"),
    ("ADYEN.AS", "ADYEN", "AEB",      "EUR", "Adyen",                 "Netherlands", "Financials",             "American", "STK"),
    ("AD.AS",    "AD",    "AEB",      "EUR", "Ahold Delhaize",        "Netherlands", "Consumer Staples",       "American", "STK"),
    ("AI.PA",    "AI",    "SBF",      "EUR", "Air Liquide",           "France",      "Materials",              "American", "STK"),
    ("AIR.PA",   "AIR",   "SBF",      "EUR", "Airbus",                "France",      "Industrials",            "American", "STK"),
    ("ALV.DE",   "ALV",   "IBIS",     "EUR", "Allianz",               "Germany",     "Financials",             "American", "STK"),
    ("ABI.BR",   "ABI",   "ENEXT.BE", "EUR", "Anheuser-Busch InBev",  "Belgium",     "Consumer Staples",       "American", "STK"),
    ("ASML.AS",  "ASML",  "AEB",      "EUR", "ASML Holding",          "Netherlands", "Information Technology", "American", "STK"),
    ("CS.PA",    "CS",    "SBF",      "EUR", "AXA",                   "France",      "Financials",             "American", "STK"),
    ("BAS.DE",   "BAS",   "IBIS",     "EUR", "BASF",                  "Germany",     "Materials",              "American", "STK"),
    ("BAYN.DE",  "BAYN",  "IBIS",     "EUR", "Bayer",                 "Germany",     "Health Care",            "American", "STK"),
    ("BBVA.MC",  "BBVA",  "BM",       "EUR", "Banco Bilbao",          "Spain",       "Financials",             "American", "STK"),
    ("SAN.MC",   "SAN",   "BM",       "EUR", "Banco Santander",       "Spain",       "Financials",             "American", "STK"),
    ("BNP.PA",   "BNP",   "SBF",      "EUR", "BNP Paribas",           "France",      "Financials",             "American", "STK"),
    ("CRG.IE",   "CRG",   "ENEXT.IE", "EUR", "CRH",                   "Ireland",     "Materials",              "American", "STK"),
    ("DTE.DE",   "DTE",   "IBIS",     "EUR", "Deutsche Telekom",      "Germany",     "Communication Services", "American", "STK"),
    ("DPW.DE",   "DPW",   "IBIS",     "EUR", "DHL Group",             "Germany",     "Industrials",            "American", "STK"),
    ("ENEL.MI",  "ENEL",  "BVME",     "EUR", "Enel",                  "Italy",       "Utilities",              "American", "STK"),
    ("ENI.MI",   "ENI",   "BVME",     "EUR", "Eni",                   "Italy",       "Energy",                 "American", "STK"),
    ("EL.PA",    "EL",    "SBF",      "EUR", "EssilorLuxottica",      "France",      "Health Care",            "American", "STK"),
    ("RMS.PA",   "RMS",   "SBF",      "EUR", "Hermès",                "France",      "Consumer Discretionary", "American", "STK"),
    ("IBE.MC",   "IBE",   "BM",       "EUR", "Iberdrola",             "Spain",       "Utilities",              "American", "STK"),
    ("ITX.MC",   "ITX",   "BM",       "EUR", "Inditex",               "Spain",       "Consumer Discretionary", "American", "STK"),
    ("IFX.DE",   "IFX",   "IBIS",     "EUR", "Infineon Technologies", "Germany",     "Information Technology", "American", "STK"),
    ("INGA.AS",  "INGA",  "AEB",      "EUR", "ING Groep",             "Netherlands", "Financials",             "American", "STK"),
    ("ISP.MI",   "ISP",   "BVME",     "EUR", "Intesa Sanpaolo",       "Italy",       "Financials",             "American", "STK"),
    ("OR.PA",    "OR",    "SBF",      "EUR", "L'Oréal",               "France",      "Consumer Staples",       "American", "STK"),
    ("MC.PA",    "MC",    "SBF",      "EUR", "LVMH",                  "France",      "Consumer Discretionary", "American", "STK"),
    ("MBG.DE",   "MBG",   "IBIS",     "EUR", "Mercedes-Benz Group",   "Germany",     "Consumer Discretionary", "American", "STK"),
    ("MUV2.DE",  "MUV2",  "IBIS",     "EUR", "Munich Re",             "Germany",     "Financials",             "American", "STK"),
    ("PRX.AS",   "PRX",   "AEB",      "EUR", "Prosus",                "Netherlands", "Consumer Discretionary", "American", "STK"),
    ("RHM.DE",   "RHM",   "IBIS",     "EUR", "Rheinmetall",           "Germany",     "Industrials",            "American", "STK"),
    ("SAF.PA",   "SAF",   "SBF",      "EUR", "Safran",                "France",      "Industrials",            "American", "STK"),
    ("SGO.PA",   "SGO",   "SBF",      "EUR", "Saint-Gobain",          "France",      "Industrials",            "American", "STK"),
    ("SAN.PA",   "SAN",   "SBF",      "EUR", "Sanofi",                "France",      "Health Care",            "American", "STK"),
    ("SAP.DE",   "SAP",   "IBIS",     "EUR", "SAP",                   "Germany",     "Information Technology", "American", "STK"),
    ("SU.PA",    "SU",    "SBF",      "EUR", "Schneider Electric",    "France",      "Industrials",            "American", "STK"),
    ("SIE.DE",   "SIE",   "IBIS",     "EUR", "Siemens",               "Germany",     "Industrials",            "American", "STK"),
    ("STMPA.PA", "STMPA", "SBF",      "EUR", "STMicroelectronics",    "France",      "Information Technology", "American", "STK"),
    ("TTE.PA",   "TTE",   "SBF",      "EUR", "TotalEnergies",         "France",      "Energy",                 "American", "STK"),
    ("DG.PA",    "DG",    "SBF",      "EUR", "Vinci",                 "France",      "Industrials",            "American", "STK"),
    ("VOW3.DE",  "VOW3",  "IBIS",     "EUR", "Volkswagen",            "Germany",     "Consumer Discretionary", "American", "STK")
]

def main() -> None:
    out = _ROOT / "data" / "reference" / "euro_stoxx_50_universe.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    columns = ["Ticker", "IBKR_Symbol", "PrimaryExchange", "Currency", "Company", "Country", "Sector", "OptStyle", "SecType"]
    df = pd.DataFrame(UNIVERSE, columns=columns)
    
    df.to_excel(out, index=False)
    print(f"Univers généré avec succès ({len(df)} symboles) dans : {out}")

if __name__ == "__main__":
    main()