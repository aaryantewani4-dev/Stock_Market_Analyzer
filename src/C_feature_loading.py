"""
================================================================================
FEATURE LOADING & DATA PREPARATION
================================================================================
Purpose: Load engineered features and prepare data for machine learning
Author: Your Name
Date: February 2026

This script:
1. Loads the processed dataset with features
2. Selects relevant features for modeling
3. Splits data into training and testing sets (chronological)
4. Scales features using StandardScaler
5. Exports prepared data for model training

Key Concept: We use chronological split (no shuffling) because stock/crypto
data is time-series - we train on past data, test on future data.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set pandas display options to see all data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ================================================================================
# CONFIGURATION: Choose which asset to analyze
# ================================================================================
# AAPL (Apple Stock) - UNCOMMENT THIS LINE TO USE:
INPUT_FILE = '../data/aaple_stock_features_returns.csv'
ASSET_NAME = "AAPL"

# BITCOIN - UNCOMMENT THESE LINES TO USE INSTEAD:
# INPUT_FILE = '../data/bitcoin_features.csv'
# ASSET_NAME = "Bitcoin"


# ================================================================================
# SECTION 1: LOAD PROCESSED DATA
# ================================================================================
print("\n" + "="*80)
print(f"LOADING FEATURES FOR {ASSET_NAME}")
print("="*80)

# Load the dataset with all engineered features
df = pd.read_csv(INPUT_FILE)

print(f"\n✅ Data loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Date range: {df['Price'].iloc[0]} to {df['Price'].iloc[-1]}")

# Verify data quality
nan_check = df.isnull().any().any()
print(f"   Any NaN values? {nan_check}")
if nan_check:
    print(f"   ⚠️ WARNING: NaN values detected! Data cleaning required.")
else:
    print(f"   ✅ Data is clean (no NaN values)")


# ================================================================================
# SECTION 2: DEFINE FEATURE SET
# ================================================================================
print("\n" + "-"*80)
print("SECTION 2: Defining Feature Set for Model")
print("-"*80)

# Select features for machine learning model
# All features are SCALE-INDEPENDENT (work at any price level)
feature_column = [
    # ---- Current Market State ----
    'Returns',              # Today's return (%)
    
    # ---- Price Position (relative to moving averages) ----
    'Price_to_MA5',         # Price / 5-day MA (ratio)
    'Price_to_MA10',        # Price / 10-day MA (ratio)
    'Price_to_MA20',        # Price / 20-day MA (ratio)
    
    # ---- Momentum (rate of change) ----
    'Momentum_5',           # 5-day momentum (%)
    'Momentum_10',          # 10-day momentum (%)
    
    # ---- Volatility (risk measures) ----
    'Volatility_5',         # 5-day volatility (std of returns)
    'Volatility_10',        # 10-day volatility (std of returns)
    
    # ---- Price Range ----
    'High_Low_Spread_Pct',  # Intraday price range (%)
    
    # ---- Volume (trading activity) ----
    'Volume_Change',        # Volume change vs previous day (%)
    'Volume_Ratio',         # Current volume / 5-day average volume
    
    # ---- Technical Indicator ----
    'RSI',                  # Relative Strength Index (0-100)
    
    # ---- Historical Context (memory) ----
    'Returns_Lag_1',        # Yesterday's return
    'Returns_Lag_2',        # Return from 2 days ago
    'Returns_Lag_3'         # Return from 3 days ago
]

print(f"✅ Feature set defined: {len(feature_column)} features")
print(f"\nFeature Categories:")
print(f"   • Current state: 1 feature")
print(f"   • Price position: 3 features")
print(f"   • Momentum: 2 features")
print(f"   • Volatility: 2 features")
print(f"   • Price range: 1 feature")
print(f"   • Volume: 2 features")
print(f"   • Technical: 1 feature")
print(f"   • Historical: 3 features")


# ================================================================================
# SECTION 3: PREPARE FEATURES AND TARGET
# ================================================================================
print("\n" + "-"*80)
print("SECTION 3: Preparing Features and Target Variable")
print("-"*80)

# Features (X): Input variables for the model
X = df[feature_column]

# Target (y): What we want to predict (next day's percentage return)
y = df['Target_Return']

# Additional data we'll need later:
# - Dates: For plotting and analysis
# - Close prices: For converting predicted returns back to prices
dates = df['Price']
close_prices = df['Close_Price']

print(f"✅ Data prepared")
print(f"   Features (X) shape: {X.shape}")
print(f"   Target (y) shape: {y.shape}")
print(f"   Dates available: {len(dates)}")
print(f"   Close prices available: {len(close_prices)}")


# ================================================================================
# SECTION 4: TRAIN-TEST SPLIT (Chronological)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 4: Splitting Data (Train/Test)")
print("-"*80)

# Split data: 80% training, 20% testing
# IMPORTANT: shuffle=False preserves chronological order
#            We train on PAST data, test on FUTURE data
#            This is how we'd use the model in real trading!

X_train, X_test, y_train, y_test, dates_train, dates_test, close_train, close_test = train_test_split(
    X,              # Features
    y,              # Target
    dates,          # Dates (for plotting)
    close_prices,   # Close prices (for reconstruction)
    test_size=0.2,  # 20% for testing
    shuffle=False,  # DON'T shuffle (preserve time order)
    random_state=42 # For reproducibility
)

print(f"✅ Data split complete")
print(f"\n   Training Set:")
print(f"      Samples: {X_train.shape[0]}")
print(f"      Date range: {dates_train.iloc[0]} to {dates_train.iloc[-1]}")
print(f"      Mean return: {y_train.mean():.4f}%")
print(f"      Std return: {y_train.std():.4f}%")

print(f"\n   Testing Set:")
print(f"      Samples: {X_test.shape[0]}")
print(f"      Date range: {dates_test.iloc[0]} to {dates_test.iloc[-1]}")
print(f"      Mean return: {y_test.mean():.4f}%")
print(f"      Std return: {y_test.std():.4f}%")

# Calculate split ratio
train_ratio = len(X_train) / len(X) * 100
test_ratio = len(X_test) / len(X) * 100
print(f"\n   Split ratio: {train_ratio:.1f}% train / {test_ratio:.1f}% test")


# ================================================================================
# SECTION 5: FEATURE SCALING (Standardization)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 5: Feature Scaling (Standardization)")
print("-"*80)

# Why scale features?
# - Different features have different ranges (Returns: -10 to +10, RSI: 0 to 100)
# - ML algorithms work better when features are on similar scales
# - StandardScaler: transforms features to have mean=0, std=1

# Create scaler
scaler = StandardScaler()

# Fit scaler on TRAINING data only (prevent data leakage)
# Then transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using StandardScaler")
print(f"   Method: Zero mean, unit variance")
print(f"   Scaler fitted on: Training data only")
print(f"   Applied to: Both train and test data")

print(f"\n   Scaled Data Shapes:")
print(f"      X_train_scaled: {X_train_scaled.shape}")
print(f"      X_test_scaled: {X_test_scaled.shape}")

# Verify scaling worked
print(f"\n   Verification (Train Set After Scaling):")
print(f"      Mean: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"      Std: {X_train_scaled.std():.6f} (should be ~1)")


# ================================================================================
# DATA PREPARATION COMPLETE
# ================================================================================
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)

print(f"\n📊 Summary for {ASSET_NAME}:")
print(f"   Total samples: {len(X)}")
print(f"   Training samples: {len(X_train)} ({train_ratio:.1f}%)")
print(f"   Testing samples: {len(X_test)} ({test_ratio:.1f}%)")
print(f"   Number of features: {len(feature_column)}")
print(f"   Target variable: Next day percentage return")

print(f"\n✅ Exported Variables (available for import):")
print(f"   • X_train, X_test (unscaled features)")
print(f"   • X_train_scaled, X_test_scaled (scaled features)")
print(f"   • y_train, y_test (target returns)")
print(f"   • dates_train, dates_test (dates)")
print(f"   • close_train, close_test (closing prices)")
print(f"   • feature_column (list of feature names)")
print(f"   • scaler (fitted StandardScaler)")
print(f"   • df (original dataframe)")

print(f"\n🚀 Ready for Model Training!")
print(f"   Next step: Run 4_model_training.py")
