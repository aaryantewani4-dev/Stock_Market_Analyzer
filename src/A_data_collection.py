"""
================================================================================
STOCK & CRYPTOCURRENCY DATA COLLECTION
================================================================================
Purpose: Download historical price data from Yahoo Finance for analysis
Author: Your Name
Date: February 2026

This script downloads:
1. Apple (AAPL) stock data - For stock market prediction
2. Bitcoin (BTC-USD) data - For cryptocurrency prediction
================================================================================
"""

import yfinance as yf
import pandas as pd

# ================================================================================
# CONFIGURATION: Update these paths to match your system
# ================================================================================
AAPL_OUTPUT_PATH = '../data/aaple_stock.csv'
BITCOIN_OUTPUT_PATH = '../data/bitcoin.csv'


# ================================================================================
# SECTION 1: APPLE (AAPL) STOCK DATA COLLECTION
# ================================================================================
print("\n" + "="*80)
print("DOWNLOADING APPLE (AAPL) STOCK DATA")
print("="*80)

# Download AAPL data from Yahoo Finance
# Parameters:
#   - Ticker: 'AAPL' (Apple Inc.)
#   - Start: 2019-01-01 (beginning of data collection)
#   - End: 2025-12-31 (end of data collection)
aapl_data = yf.download(
    'AAPL',
    start='2019-01-01',
    end='2025-12-31',
    progress=True  # Show download progress
)

# Display basic information about downloaded data
print(f"\n✅ AAPL Data Downloaded Successfully!")
print(f"   Records: {len(aapl_data)}")
print(f"   Date Range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
print(f"   Columns: {list(aapl_data.columns)}")

# Save to CSV file for later use
aapl_data.to_csv(AAPL_OUTPUT_PATH)
print(f"   Saved to: {AAPL_OUTPUT_PATH}")


# ================================================================================
# SECTION 2: BITCOIN (BTC-USD) DATA COLLECTION
# ================================================================================
print("\n" + "="*80)
print("DOWNLOADING BITCOIN (BTC-USD) DATA")
print("="*80)

# Download Bitcoin data from Yahoo Finance
# Parameters:
#   - Ticker: 'BTC-USD' (Bitcoin in USD)
#   - Start: 2021-01-01 (Bitcoin data starts later than stocks)
#   - End: 2026-02-12 (most recent data available)
bitcoin_data = yf.download(
    'BTC-USD',
    start='2021-01-01',
    end='2026-02-12',
    progress=True  # Show download progress
)

# Display basic information about downloaded data
print(f"\n✅ Bitcoin Data Downloaded Successfully!")
print(f"   Records: {len(bitcoin_data)}")
print(f"   Date Range: {bitcoin_data.index[0]} to {bitcoin_data.index[-1]}")
print(f"   Columns: {list(bitcoin_data.columns)}")

# Display sample of Bitcoin data
print("\nSample Bitcoin Data (First 5 rows):")
print(bitcoin_data.head())

# Save to CSV file for later use
bitcoin_data.to_csv(BITCOIN_OUTPUT_PATH)
print(f"\n   Saved to: {BITCOIN_OUTPUT_PATH}")


# ================================================================================
# DATA STRUCTURE EXPLANATION
# ================================================================================
"""
Both datasets contain the following columns:
- Date: Trading date (index)
- Open: Opening price of the day
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing price of the day
- Adj Close: Adjusted closing price (accounts for splits, dividends)
- Volume: Number of shares/coins traded

Note: 
- AAPL: Prices in USD, volume in number of shares
- BTC-USD: Prices in USD, volume in number of bitcoins
"""

print("\n" + "="*80)
print("DATA COLLECTION COMPLETE!")
print("="*80)
print(f"\nNext Steps:")
print(f"1. Run 2_feature_engineering.py to create technical indicators")
print(f"2. Choose which asset to analyze (AAPL or Bitcoin)")
print(f"3. Uncomment the appropriate lines in subsequent scripts")
