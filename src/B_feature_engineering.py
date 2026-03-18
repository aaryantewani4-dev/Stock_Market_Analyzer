"""
================================================================================
FEATURE ENGINEERING FOR STOCK/CRYPTO PREDICTION
================================================================================
Purpose: Create technical indicators and features for machine learning
Author: Your Name
Date: February 2026

This script transforms raw price data into machine learning features:
- Returns-based features (percentage changes - scale independent)
- Technical indicators (RSI, Moving Averages, Momentum, Volatility)
- Price ratios (relative to moving averages)
- Volume indicators

Key Concept: We use RETURNS (%) instead of absolute prices to make the model
scale-independent and able to work at any price level.
================================================================================
"""

import pandas as pd
import numpy as np

# Set pandas display options to see all columns
pd.set_option('display.max_columns', None)


# ================================================================================
# CONFIGURATION: Choose which asset to analyze
# ================================================================================
# AAPL (Apple Stock) - UNCOMMENT THIS LINE TO USE:
INPUT_FILE = '../data/aaple_stock.csv'
OUTPUT_FILE = '../data/aaple_stock_features_returns.csv'
ASSET_NAME = "AAPL"

# BITCOIN - UNCOMMENT THESE LINES TO USE INSTEAD:
# INPUT_FILE = '../data/bitcoin.csv'
# OUTPUT_FILE = '../data/bitcoin_features.csv'
# ASSET_NAME = "Bitcoin"


# ================================================================================
# LOAD DATA
# ================================================================================
print("\n" + "="*80)
print(f"FEATURE ENGINEERING FOR {ASSET_NAME}")
print("="*80)

df = pd.read_csv(INPUT_FILE)
print(f"\n✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Date range: {df['Price'].iloc[0]} to {df['Price'].iloc[-1]}")


# ================================================================================
# SECTION 1: PERCENTAGE RETURNS (Core Feature)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 1: Calculating Percentage Returns")
print("-"*80)

# Calculate daily percentage returns
# Formula: (Today's Price - Yesterday's Price) / Yesterday's Price * 100
# This makes the model scale-independent (works at $50 or $500)
df['Returns'] = df['Close'].pct_change() * 100

print(f"✅ Returns calculated")
print(f"   Mean daily return: {df['Returns'].mean():.4f}%")
print(f"   Std deviation: {df['Returns'].std():.4f}%")
print(f"   Range: {df['Returns'].min():.2f}% to {df['Returns'].max():.2f}%")


# ================================================================================
# SECTION 2: MOVING AVERAGES (Trend Indicators)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 2: Calculating Moving Averages")
print("-"*80)

# Calculate moving averages for different time windows
# These help identify trends and support/resistance levels

# 5-day MA: Short-term trend
df['MA_5'] = df['Close'].rolling(window=5).mean()

# 10-day MA: Medium-short term trend
df['MA_10'] = df['Close'].rolling(window=10).mean()

# 20-day MA: Medium-term trend (approximately 1 month of trading days)
df['MA_20'] = df['Close'].rolling(window=20).mean()

print(f"✅ Moving averages calculated (5, 10, 20 days)")


# ================================================================================
# SECTION 3: PRICE-TO-MA RATIOS (Scale-Independent Position Indicators)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 3: Calculating Price-to-MA Ratios")
print("-"*80)

# These ratios tell us where current price is relative to its moving average
# Values > 1: Price above MA (bullish)
# Values < 1: Price below MA (bearish)
# These are SCALE-INDEPENDENT (work at any price level)

df['Price_to_MA5'] = df['Close'] / df['MA_5']
df['Price_to_MA10'] = df['Close'] / df['MA_10']
df['Price_to_MA20'] = df['Close'] / df['MA_20']

print(f"✅ Price-to-MA ratios calculated")
print(f"   These indicate if price is above/below its moving averages")


# ================================================================================
# SECTION 4: MOMENTUM INDICATORS (Rate of Change)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 4: Calculating Momentum Indicators")
print("-"*80)

# Momentum shows the rate of price change over N days
# Positive momentum: Price going up
# Negative momentum: Price going down

# 5-day momentum (1 week)
df['Momentum_5'] = df['Close'].pct_change(5) * 100

# 10-day momentum (2 weeks)
df['Momentum_10'] = df['Close'].pct_change(10) * 100

print(f"✅ Momentum indicators calculated (5, 10 days)")
print(f"   Shows rate of price change over time windows")


# ================================================================================
# SECTION 5: VOLATILITY MEASURES (Risk Indicators)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 5: Calculating Volatility Measures")
print("-"*80)

# Volatility = Standard deviation of returns
# High volatility = More price swings (higher risk)
# Low volatility = Stable prices (lower risk)

# 5-day volatility (short-term risk)
df['Volatility_5'] = df['Returns'].rolling(window=5).std()

# 10-day volatility (medium-term risk)
df['Volatility_10'] = df['Returns'].rolling(window=10).std()

print(f"✅ Volatility measures calculated (5, 10 days)")
print(f"   Mean 5-day volatility: {df['Volatility_5'].mean():.4f}%")
print(f"   Mean 10-day volatility: {df['Volatility_10'].mean():.4f}%")


# ================================================================================
# SECTION 6: PRICE RANGE INDICATOR (Intraday Volatility)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 6: Calculating Price Range Indicator")
print("-"*80)

# High-Low spread as percentage of closing price
# Shows how much price moved within the day
# Higher values = More intraday volatility
df['High_Low_Spread_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100

print(f"✅ High-Low spread calculated")
print(f"   Average daily range: {df['High_Low_Spread_Pct'].mean():.2f}%")


# ================================================================================
# SECTION 7: VOLUME INDICATORS (Trading Activity)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 7: Calculating Volume Indicators")
print("-"*80)

# Volume indicates trading activity and market interest
# High volume = Strong conviction in price movement
# Low volume = Weak conviction

# Volume change (% change from previous day)
df['Volume_Change'] = df['Volume'].pct_change() * 100

# 5-day average volume
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

# Volume ratio (current volume vs 5-day average)
# Values > 1: Above average volume
# Values < 1: Below average volume
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

print(f"✅ Volume indicators calculated")
print(f"   Shows trading activity levels")


# ================================================================================
# SECTION 8: RSI (Relative Strength Index)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 8: Calculating RSI (Relative Strength Index)")
print("-"*80)

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)

    RSI is a momentum indicator that measures overbought/oversold conditions:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI 30-70: Neutral zone

    Parameters:
    -----------
    data : pd.Series
        Price data (typically closing prices)
    window : int
        Lookback period (default: 14 days, industry standard)

    Returns:
    --------
    pd.Series : RSI values (0-100 scale)
    """
    # Calculate price changes
    delta = data.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Calculate relative strength
    rs = gain / loss

    # Calculate RSI (0-100 scale)
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Calculate RSI with 14-day window (industry standard)
df['RSI'] = calculate_rsi(df['Close'], window=14)

print(f"✅ RSI calculated (14-day window)")
print(f"   Mean RSI: {df['RSI'].mean():.2f}")
print(f"   Current RSI: {df['RSI'].iloc[-1]:.2f}")
if df['RSI'].iloc[-1] > 70:
    print(f"   ⚠️ Currently OVERBOUGHT (RSI > 70)")
elif df['RSI'].iloc[-1] < 30:
    print(f"   ⚠️ Currently OVERSOLD (RSI < 30)")
else:
    print(f"   ✅ Currently in NEUTRAL zone")


# ================================================================================
# SECTION 9: LAGGED RETURNS (Memory/Historical Context)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 9: Creating Lagged Return Features")
print("-"*80)

# Lagged returns provide historical context
# These tell the model what happened in recent days
# Helps capture momentum and mean reversion patterns

# Yesterday's return
df['Returns_Lag_1'] = df['Returns'].shift(1)

# 2 days ago return
df['Returns_Lag_2'] = df['Returns'].shift(2)

# 3 days ago return
df['Returns_Lag_3'] = df['Returns'].shift(3)

print(f"✅ Lagged returns created (1, 2, 3 days)")
print(f"   These provide historical context for prediction")


# ================================================================================
# SECTION 10: SAVE CLOSE PRICE FOR LATER RECONSTRUCTION
# ================================================================================
print("\n" + "-"*80)
print("SECTION 10: Saving Close Price for Price Reconstruction")
print("-"*80)

# Keep the actual closing price in a separate column
# We'll need this later to convert predicted returns back to prices
df['Close_Price'] = df['Close']

print(f"✅ Close price saved for later use")


# ================================================================================
# SECTION 11: CREATE TARGET VARIABLE
# ================================================================================
print("\n" + "-"*80)
print("SECTION 11: Creating Target Variable")
print("-"*80)

# Target: Next day's percentage return
# We shift by -1 to get tomorrow's return for today's features
# This is what we're trying to predict!
df['Target_Return'] = df['Returns'].shift(-1)

print(f"✅ Target variable created: Next day's return")
print(f"   This is what the model will learn to predict")


# ================================================================================
# SECTION 12: CLEAN DATA (Remove NaN values)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 12: Cleaning Data (Removing NaN values)")
print("-"*80)

# Count NaN before cleaning
nan_count_before = df.isnull().sum().sum()
rows_before = len(df)

# Remove rows with NaN (created by rolling calculations and shifts)
df = df.dropna()

rows_after = len(df)
rows_removed = rows_before - rows_after

print(f"✅ Data cleaned")
print(f"   Rows before: {rows_before}")
print(f"   Rows removed: {rows_removed}")
print(f"   Rows after: {rows_after}")
print(f"   NaN values removed: {nan_count_before}")


# ================================================================================
# SECTION 13: DISPLAY STATISTICS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 13: Target Variable Statistics")
print("-"*80)

print(f"\nNext Day Return (Target) Statistics:")
print(df['Target_Return'].describe())

print(f"\nDistribution:")
print(f"   Positive returns (up days): {(df['Target_Return'] > 0).sum()} ({(df['Target_Return'] > 0).sum()/len(df)*100:.1f}%)")
print(f"   Negative returns (down days): {(df['Target_Return'] < 0).sum()} ({(df['Target_Return'] < 0).sum()/len(df)*100:.1f}%)")


# ================================================================================
# SECTION 14: SAVE PROCESSED DATA
# ================================================================================
print("\n" + "-"*80)
print("SECTION 14: Saving Processed Data")
print("-"*80)

# Save the fully processed dataset with all features
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Processed data saved!")
print(f"   Output file: {OUTPUT_FILE}")
print(f"   Total features created: {len(df.columns) - len(pd.read_csv(INPUT_FILE).columns)}")


# ================================================================================
# SUMMARY
# ================================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print("="*80)
print(f"\nFinal Dataset:")
print(f"   Asset: {ASSET_NAME}")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"\nFeatures Created:")
print(f"   ✅ 1 Return feature")
print(f"   ✅ 3 Price-to-MA ratios")
print(f"   ✅ 2 Momentum indicators")
print(f"   ✅ 2 Volatility measures")
print(f"   ✅ 1 Price range indicator")
print(f"   ✅ 3 Volume indicators")
print(f"   ✅ 1 RSI indicator")
print(f"   ✅ 3 Lagged returns")
print(f"   ✅ 1 Target variable (next day return)")
print(f"\nTotal: 15 input features + 1 target variable")
print(f"\nNext Step: Run 3_feature_loading.py to prepare for model training")
