#!/usr/bin/env python3
"""
Script one-shot : génère data/reference/sp500_top50_universe.xlsx
SPX (index) + top 50 S&P 500 par capitalisation boursière.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]

UNIVERSE = [
    # Ticker, Company, Sector, SecType, Exchange, Currency, OptStyle, Note
    ("SPX",   "S&P 500 Index",              "Index",                "IND", "CBOE",  "USD", "European", ""),
    ("AAPL",  "Apple",                       "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("MSFT",  "Microsoft",                   "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("NVDA",  "NVIDIA",                      "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("AMZN",  "Amazon",                      "Consumer Discret.",    "STK", "SMART", "USD", "American",  ""),
    ("GOOGL", "Alphabet",                    "Comm. Services",       "STK", "SMART", "USD", "American",  ""),
    ("META",  "Meta Platforms",              "Comm. Services",       "STK", "SMART", "USD", "American",  ""),
    ("BRK.B", "Berkshire Hathaway B",        "Financials",           "STK", "SMART", "USD", "American",  "IBKR symbol: BRK B"),
    ("LLY",   "Eli Lilly",                   "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("AVGO",  "Broadcom",                    "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("TSLA",  "Tesla",                       "Consumer Discret.",    "STK", "SMART", "USD", "American",  ""),
    ("JPM",   "JPMorgan Chase",              "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("UNH",   "UnitedHealth",                "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("V",     "Visa",                        "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("XOM",   "ExxonMobil",                  "Energy",               "STK", "SMART", "USD", "American",  ""),
    ("MA",    "Mastercard",                  "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("COST",  "Costco",                      "Consumer Staples",     "STK", "SMART", "USD", "American",  ""),
    ("HD",    "Home Depot",                  "Industrials",          "STK", "SMART", "USD", "American",  ""),
    ("PG",    "Procter & Gamble",            "Consumer Staples",     "STK", "SMART", "USD", "American",  ""),
    ("JNJ",   "Johnson & Johnson",           "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("ABBV",  "AbbVie",                      "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("NFLX",  "Netflix",                     "Comm. Services",       "STK", "SMART", "USD", "American",  ""),
    ("MRK",   "Merck",                       "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("CRM",   "Salesforce",                  "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("CVX",   "Chevron",                     "Energy",               "STK", "SMART", "USD", "American",  ""),
    ("WMT",   "Walmart",                     "Consumer Staples",     "STK", "SMART", "USD", "American",  ""),
    ("BAC",   "Bank of America",             "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("KO",    "Coca-Cola",                   "Consumer Staples",     "STK", "SMART", "USD", "American",  ""),
    ("AMD",   "Advanced Micro Devices",      "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("PEP",   "PepsiCo",                     "Consumer Staples",     "STK", "SMART", "USD", "American",  ""),
    ("ACN",   "Accenture",                   "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("MCD",   "McDonald's",                  "Consumer Discret.",    "STK", "SMART", "USD", "American",  ""),
    ("LIN",   "Linde",                       "Materials",            "STK", "SMART", "USD", "American",  ""),
    ("CSCO",  "Cisco",                       "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("ORCL",  "Oracle",                      "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("TMO",   "Thermo Fisher",               "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("ABT",   "Abbott",                      "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("IBM",   "IBM",                         "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("GS",    "Goldman Sachs",               "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("CAT",   "Caterpillar",                 "Industrials",          "STK", "SMART", "USD", "American",  ""),
    ("INTU",  "Intuit",                      "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("ISRG",  "Intuitive Surgical",          "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("AMGN",  "Amgen",                       "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("SPGI",  "S&P Global",                  "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("BKNG",  "Booking Holdings",            "Consumer Discret.",    "STK", "SMART", "USD", "American",  ""),
    ("AXP",   "American Express",            "Financials",           "STK", "SMART", "USD", "American",  ""),
    ("NOW",   "ServiceNow",                  "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("UNP",   "Union Pacific",               "Industrials",          "STK", "SMART", "USD", "American",  ""),
    ("AMAT",  "Applied Materials",           "Technology",           "STK", "SMART", "USD", "American",  ""),
    ("GILD",  "Gilead Sciences",             "Healthcare",           "STK", "SMART", "USD", "American",  ""),
    ("DE",    "Deere & Company",             "Industrials",          "STK", "SMART", "USD", "American",  ""),
]


def main() -> None:
    out = _ROOT / "data" / "reference" / "sp500_top50_universe.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        UNIVERSE,
        columns=["Ticker", "Company", "Sector", "SecType", "Exchange", "Currency", "OptStyle", "Note"],
    )

    df.to_excel(out, index=False)
    print(f"Univers créé : {out}  ({len(df)} lignes)")


if __name__ == "__main__":
    main()
