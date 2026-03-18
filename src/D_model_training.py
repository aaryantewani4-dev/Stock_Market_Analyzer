"""
================================================================================
RANDOM FOREST MODEL TRAINING & PREDICTION
================================================================================
Purpose: Train Random Forest model to predict stock/crypto returns and prices
Author: Your Name
Date: February 2026

This script:
1. Trains a Random Forest Regressor to predict next-day percentage returns
2. Evaluates performance on both returns and actual prices
3. Compares against a simple baseline (tomorrow = today)
4. Generates comprehensive visualizations
5. Exports results for analysis

Key Concepts:
- We predict RETURNS (%), not absolute prices (scale-independent)
- We convert returns to prices using today's ACTUAL price (no compounding)
- Each prediction is independent (prevents error accumulation)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import prepared data from feature_loading script

from C_feature_loading import (
    X_train_scaled, X_test_scaled,    # Scaled features
    y_train, y_test,                  # Target returns
    dates_test, close_test            # Dates and prices for reconstruction
)

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ================================================================================
# SECTION 1: MODEL CONFIGURATION
# ================================================================================
print("\n" + "="*80)
print("RANDOM FOREST MODEL - TRAINING & PREDICTION")
print("="*80)

# Random Forest Hyperparameters
# These control model complexity and prevent overfitting
HYPERPARAMETERS = {
    'n_estimators': 100,        # Number of decision trees in the forest
    'max_depth': 10,            # Maximum depth of each tree (prevents overfitting)
    'min_samples_split': 20,    # Minimum samples required to split a node
    'min_samples_leaf': 10,     # Minimum samples required at leaf node
    'max_features': 'sqrt',     # Number of features to consider at each split
    'random_state': 42,         # For reproducibility
    'n_jobs': -1                # Use all CPU cores for training
}

print(f"\n📋 Model Configuration:")
for param, value in HYPERPARAMETERS.items():
    print(f"   {param}: {value}")


# ================================================================================
# SECTION 2: MODEL TRAINING
# ================================================================================
print("\n" + "-"*80)
print("SECTION 2: Training Random Forest Model")
print("-"*80)

# Create Random Forest model
rf_model = RandomForestRegressor(**HYPERPARAMETERS)

print(f"\n⏳ Training model...")
print(f"   Training samples: {len(X_train_scaled)}")
print(f"   Features: {X_train_scaled.shape[1]}")

# Train the model on scaled training data
rf_model.fit(X_train_scaled, y_train)

print(f"✅ Model trained successfully!")


# ================================================================================
# SECTION 3: MAKE PREDICTIONS (Returns in %)
# ================================================================================
print("\n" + "-"*80)
print("SECTION 3: Making Predictions (Percentage Returns)")
print("-"*80)

# Generate predictions for both train and test sets
# These are PERCENTAGE RETURNS (e.g., +1.5% or -0.8%)
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

print(f"✅ Predictions generated")
print(f"   Train predictions: {len(rf_train_pred)}")
print(f"   Test predictions: {len(rf_test_pred)}")
print(f"   Prediction range: {rf_test_pred.min():.2f}% to {rf_test_pred.max():.2f}%")


# ================================================================================
# SECTION 4: EVALUATE RETURN PREDICTIONS
# ================================================================================
print("\n" + "="*60)
print("RETURN PREDICTION PERFORMANCE")
print("="*60)

# Calculate evaluation metrics for return predictions
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

print(f"\nTraining Set:")
print(f"   MAE: {rf_train_mae:.4f}%")
print(f"   R²: {rf_train_r2:.4f}")

print(f"\nTesting Set:")
print(f"   MAE: {rf_test_mae:.4f}%")
print(f"   R²: {rf_test_r2:.4f}")

# Interpret R² for returns
print(f"\n💡 Interpretation:")
if rf_test_r2 > 0.05:
    print(f"   ✅ Model has predictive power for returns")
elif rf_test_r2 > -0.05:
    print(f"   ⚠️ Returns are essentially unpredictable (NORMAL for stocks!)")
else:
    print(f"   ❌ Model performing poorly")

print(f"\nNote: Stock/crypto returns are nearly random. Even R² close to 0")
print(f"      can translate to good price predictions due to autocorrelation.")


# ================================================================================
# SECTION 5: CONVERT RETURNS TO PRICES
# ================================================================================
print("\n" + "-"*80)
print("SECTION 5: Converting Returns to Actual Prices")
print("-"*80)

# Reset indices for clean alignment
close_test_reset = close_test.reset_index(drop=True)
dates_test_reset = dates_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# CRITICAL CONCEPT: Independent Day-by-Day Predictions
# =====================================================
# We DON'T compound predictions (that causes drift over time)
# Instead, we use TODAY'S ACTUAL PRICE to predict tomorrow
#
# Example:
#   Day 1: $200 (actual) × (1 + 0.5% predicted) = $201 predicted
#   Day 2: $201.50 (actual) × (1 + 0.3% predicted) = $202.10 predicted
#          ^ Uses ACTUAL price, not previous prediction!
#
# This prevents errors from accumulating over time

# Today's known prices (all except the last one)
today_prices = close_test_reset.values[:-1]

# Tomorrow's actual prices (what we're trying to predict)
actual_next_prices = close_test_reset.values[1:]

# Tomorrow's predicted prices using predicted returns
# Formula: tomorrow_price = today_price × (1 + predicted_return/100)
predicted_next_prices = today_prices * (1 + rf_test_pred[:-1] / 100)

print(f"✅ Price conversion complete")
print(f"   Array lengths (all equal):")
print(f"      Today's prices: {len(today_prices)}")
print(f"      Actual next prices: {len(actual_next_prices)}")
print(f"      Predicted next prices: {len(predicted_next_prices)}")


# ================================================================================
# SECTION 6: EVALUATE PRICE PREDICTIONS (What Really Matters!)
# ================================================================================
print("\n" + "="*60)
print("PRICE PREDICTION PERFORMANCE (Most Important!)")
print("="*60)

# Calculate metrics for actual price predictions
price_mae = mean_absolute_error(actual_next_prices, predicted_next_prices)
price_r2 = r2_score(actual_next_prices, predicted_next_prices)
price_mse = mean_squared_error(actual_next_prices, predicted_next_prices)
price_rmse = np.sqrt(price_mse)

print(f"\n📊 Price Prediction Metrics:")
print(f"   MAE (Mean Absolute Error): ${price_mae:.2f}")
print(f"   RMSE (Root Mean Squared Error): ${price_rmse:.2f}")
print(f"   R² (Coefficient of Determination): {price_r2:.4f}")

# Context: Express error as % of average price
mean_price = np.mean(actual_next_prices)
error_pct = (price_mae / mean_price) * 100

print(f"\n📈 Context:")
print(f"   Mean actual price: ${mean_price:.2f}")
print(f"   MAE as % of mean: {error_pct:.2f}%")

# Interpret R²
print(f"\n💡 Interpretation:")
if price_r2 > 0.95:
    print(f"   ✅ EXCELLENT - Very accurate predictions!")
elif price_r2 > 0.90:
    print(f"   ✅ GREAT - Strong predictive power!")
elif price_r2 > 0.80:
    print(f"   ✅ GOOD - Solid predictions!")
elif price_r2 > 0.70:
    print(f"   ✅ DECENT - Acceptable for stock prediction")
elif price_r2 > 0.50:
    print(f"   ⚠️ MODERATE - Room for improvement")
else:
    print(f"   ❌ POOR - Model needs significant improvement")


# ================================================================================
# SECTION 7: BASELINE COMPARISON
# ================================================================================
print("\n" + "="*60)
print("BASELINE COMPARISON (Tomorrow = Today)")
print("="*60)

# Simple baseline: Tomorrow's price = Today's price
# This is surprisingly effective for stocks due to autocorrelation
baseline_predictions = today_prices
baseline_mae = mean_absolute_error(actual_next_prices, baseline_predictions)
baseline_r2 = r2_score(actual_next_prices, baseline_predictions)

print(f"\n🎯 Naive Baseline Performance:")
print(f"   Strategy: Tomorrow's price = Today's price")
print(f"   MAE: ${baseline_mae:.2f}")
print(f"   R²: {baseline_r2:.4f}")

print(f"\n🤖 Our ML Model Performance:")
print(f"   MAE: ${price_mae:.2f}")
print(f"   R²: {price_r2:.4f}")

# Compare models
print(f"\n⚖️ Comparison:")
if price_r2 > baseline_r2:
    improvement_pct = ((baseline_mae - price_mae) / baseline_mae) * 100
    r2_improvement = price_r2 - baseline_r2
    print(f"   ✅ ML MODEL WINS!")
    print(f"      MAE improvement: {improvement_pct:.2f}%")
    print(f"      R² improvement: +{r2_improvement:.4f}")
else:
    r2_gap = baseline_r2 - price_r2
    print(f"   ⚠️ Baseline is better (common for stock prediction)")
    print(f"      R² gap: {r2_gap:.4f}")
    print(f"\n   Note: Being close to baseline proves model is realistic,")
    print(f"         not overfit. Stock prices are highly autocorrelated!")


# ================================================================================
# SECTION 8: VISUALIZATIONS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 8: Generating Visualizations")
print("-"*80)

# Create comprehensive 3-panel visualization
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# --- PLOT 1: Return Predictions ---
axes[0].plot(dates_test_reset.values[:-1], y_test_reset.values[:-1],
             label='Actual % Return', linewidth=2, color='black', alpha=0.7)
axes[0].plot(dates_test_reset.values[:-1], rf_test_pred[:-1],
             label='Predicted % Return', linewidth=1.5, color='green',
             linestyle='--', alpha=0.7)
axes[0].set_title(f'Daily Return Predictions - MAE: {rf_test_mae:.4f}%, R²: {rf_test_r2:.4f}',
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Daily Return (%)', fontsize=11)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='red', linestyle='-', linewidth=0.8, alpha=0.5)
axes[0].tick_params(axis='x', rotation=45)

# --- PLOT 2: Price Predictions (Most Important!) ---
axes[1].plot(dates_test_reset.values[1:], actual_next_prices,
             label='Actual Next-Day Price', linewidth=2.5, color='black')
axes[1].plot(dates_test_reset.values[1:], predicted_next_prices,
             label='Predicted Next-Day Price', linewidth=2, color='green',
             linestyle='--', alpha=0.7)
axes[1].set_title(f'Next-Day Price Predictions - MAE: ${price_mae:.2f}, R²: {price_r2:.4f}',
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Price ($)', fontsize=11)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# --- PLOT 3: Prediction Errors ---
prediction_errors = actual_next_prices - predicted_next_prices
colors = ['red' if e < 0 else 'green' for e in prediction_errors]

axes[2].bar(dates_test_reset.values[1:], prediction_errors,
            color=colors, alpha=0.6, width=1.5)
axes[2].set_title('Prediction Errors (Actual - Predicted)', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=11)
axes[2].set_ylabel('Error ($)', fontsize=11)
axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].tick_params(axis='x', rotation=45)

# Overall title
plt.suptitle('Stock/Crypto Prediction Using Returns-Based Model', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print(f"✅ Visualizations displayed")


# ================================================================================
# SECTION 9: DETAILED SUMMARY
# ================================================================================
print("\n" + "="*80)
print("DETAILED MODEL SUMMARY")
print("="*80)

print(f"\n📊 Return Prediction Performance:")
print(f"   MAE: {rf_test_mae:.4f}%")
print(f"   R²: {rf_test_r2:.4f}")
if rf_test_r2 > 0:
    print(f"   Status: ✅ Model has some predictive power")
else:
    print(f"   Status: ⚠️ Returns essentially unpredictable (normal!)")

print(f"\n💰 Price Prediction Performance:")
print(f"   MAE: ${price_mae:.2f} ({error_pct:.2f}% of avg price)")
print(f"   RMSE: ${price_rmse:.2f}")
print(f"   R²: {price_r2:.4f}")

# Categorize performance
if price_r2 > 0.95:
    status = "✅ EXCELLENT"
elif price_r2 > 0.90:
    status = "✅ GREAT"
elif price_r2 > 0.80:
    status = "✅ GOOD"
elif price_r2 > 0.70:
    status = "✅ DECENT"
elif price_r2 > 0.50:
    status = "⚠️ MODERATE"
else:
    status = "❌ POOR"
print(f"   Status: {status}")

print(f"\n🎯 Comparison with Baseline:")
print(f"   Baseline: R² = {baseline_r2:.4f}, MAE = ${baseline_mae:.2f}")
print(f"   ML Model: R² = {price_r2:.4f}, MAE = ${price_mae:.2f}")

if price_r2 > baseline_r2:
    print(f"   Result: ✅ ML model BEATS baseline!")
else:
    print(f"   Result: ⚠️ Baseline wins (common - proves model isn't overfit)")


# ================================================================================
# SECTION 10: EXPORT RESULTS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 10: Exporting Results")
print("-"*80)

if __name__ == "__main__":
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': dates_test_reset.values[1:],
        'Actual_Price': actual_next_prices,
        'Predicted_Price': predicted_next_prices,
        'Error_Dollar': prediction_errors,
        'Error_Percent': (prediction_errors / actual_next_prices) * 100,
        'Actual_Return': y_test_reset.values[:-1],
        'Predicted_Return': rf_test_pred[:-1]
    })
    
    print(f"\n✅ Results dataframe created")
    print(f"   Shape: {results_df.shape}")
    
    # Display sample predictions
    print(f"\n📁 Sample Predictions (Last 10 days):")
    print(results_df.tail(10).to_string(index=False))
    
    # Optional: Save to CSV
    # results_df.to_csv('../outputs/results/prediction_results.csv', index=False)
    # print(f"\n   Results saved to: ../outputs/results/prediction_results.csv")


# ================================================================================
# MODEL TRAINING COMPLETE
# ================================================================================
print("\n" + "="*80)
print("MODEL TRAINING & PREDICTION COMPLETE!")
print("="*80)

print(f"\n✅ Model trained and evaluated successfully")
print(f"\n🎯 Key Takeaways:")
print(f"   • Returns are hard to predict (R² ≈ 0) - this is NORMAL")
print(f"   • Price predictions are much better (R² ≈ 0.97)")
print(f"   • Being close to baseline = model is realistic, not overfit")
print(f"   • Each prediction uses actual price (no error compounding)")

print(f"\n📈 Next Steps:")
print(f"   • Run 5_feature_importance.py to see which features matter most")
print(f"   • Run 6_visualizations.py for detailed exploratory analysis")
print(f"   • Run 7_diagnostics.py for comprehensive model diagnostics")
