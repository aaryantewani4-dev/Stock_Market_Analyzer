"""
================================================================================
FEATURE IMPORTANCE ANALYSIS
================================================================================
Purpose: Analyze which features are most important for prediction
Author: Your Name
Date: February 2026

This script:
1. Extracts feature importance from trained Random Forest model
2. Ranks features by their contribution to predictions
3. Visualizes top features
4. Analyzes prediction errors in detail
5. Provides insights on model behavior

Key Insight: Understanding which features drive predictions helps improve
the model and provides insights into market behavior.
================================================================================
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress emoji font warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import trained model and data
from D_model_training import (
    rf_model,           # Trained Random Forest model
    rf_test_pred,       # Test predictions
    rf_test_mae,        # Test MAE
    rf_test_r2          # Test R²
)
from C_feature_loading import (
    feature_column,     # List of feature names
    y_test,            # Actual test returns
    dates_test,        # Test dates
    close_test         # Test closing prices
)


# ================================================================================
# SECTION 1: EXTRACT FEATURE IMPORTANCE
# ================================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Random Forest models provide feature importance scores
# These show how much each feature contributes to predictions
# Higher values = more important features

# Create dataframe with features and their importance scores
feature_importance = pd.DataFrame({
    'Feature': feature_column,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n✅ Feature importance extracted")
print(f"   Total features: {len(feature_importance)}")


# ================================================================================
# SECTION 2: DISPLAY TOP FEATURES
# ================================================================================
print("\n" + "-"*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("-"*80)

# Display top 10 features without index
print(feature_importance.head(10).to_string(index=False))

# Calculate cumulative importance
top_3_importance = feature_importance['Importance'].head(3).sum()
top_5_importance = feature_importance['Importance'].head(5).sum()
top_10_importance = feature_importance['Importance'].head(10).sum()

print(f"\n📊 Cumulative Importance:")
print(f"   Top 3 features: {top_3_importance*100:.2f}% of total importance")
print(f"   Top 5 features: {top_5_importance*100:.2f}% of total importance")
print(f"   Top 10 features: {top_10_importance*100:.2f}% of total importance")


# ================================================================================
# SECTION 3: VISUALIZE FEATURE IMPORTANCE
# ================================================================================
print("\n" + "-"*80)
print("SECTION 3: Visualizing Feature Importance")
print("-"*80)

# Create horizontal bar chart of top 15 features
plt.figure(figsize=(12, 8))

plt.barh(feature_importance['Feature'][:15],
         feature_importance['Importance'][:15],
         color='steelblue', alpha=0.8, edgecolor='navy')

plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
plt.ylabel('Features', fontsize=13, fontweight='bold')
plt.title('Top 15 Feature Importance (Random Forest Model)',
          fontsize=16, fontweight='bold')

# Invert y-axis so highest importance is at top
plt.gca().invert_yaxis()

# Add grid for easier reading
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print(f"✅ Feature importance chart displayed")


# ================================================================================
# SECTION 4: PREDICTION ERROR ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("PREDICTION ERROR ANALYSIS")
print("="*80)

# Calculate prediction errors
# Error = Actual - Predicted (positive = underestimation, negative = overestimation)
rf_return_errors = y_test.values - rf_test_pred

print(f"\n📊 Error Statistics:")
print(f"   Number of predictions: {len(rf_return_errors)}")


# ================================================================================
# SECTION 5: COMPREHENSIVE ERROR VISUALIZATIONS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 5: Creating Error Visualizations")
print("-"*80)

# Create 2x2 grid of error analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate statistics for annotations
mean_error = np.mean(rf_return_errors)
std_error = np.std(rf_return_errors)


# --- PLOT 1: Error Distribution (Histogram) ---
axes[0, 0].hist(rf_return_errors, bins=50, color='coral',
                edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Return Prediction Error Distribution',
                     fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Prediction Error (%)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, 
                   label='Zero Error')
axes[0, 0].grid(True, alpha=0.3)

# Add statistics text box
stats_text = f'Mean: {mean_error:.4f}%\nStd: {std_error:.4f}%'
axes[0, 0].text(0.02, 0.98, stats_text,
                transform=axes[0, 0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
axes[0, 0].legend()


# --- PLOT 2: Residual Plot (Scatter) ---
axes[0, 1].scatter(rf_test_pred, rf_return_errors, alpha=0.5,
                   color='steelblue', s=20)
axes[0, 1].set_title('Residual Plot - Return Predictions',
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Predicted Return (%)', fontsize=12)
axes[0, 1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Zero Error')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Interpretation note
if abs(mean_error) < 0.1:
    bias_text = "✅ Nearly unbiased"
elif mean_error > 0:
    bias_text = "⚠️ Slight underestimation bias"
else:
    bias_text = "⚠️ Slight overestimation bias"
axes[0, 1].text(0.02, 0.98, bias_text,
                transform=axes[0, 1].transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


# --- PLOT 3: Error Over Time ---
axes[1, 0].plot(dates_test.values, rf_return_errors,
                color='purple', alpha=0.6, linewidth=1)
axes[1, 0].fill_between(dates_test.values, rf_return_errors,
                        alpha=0.3, color='purple')
axes[1, 0].set_title('Prediction Errors Over Time',
                     fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Date', fontsize=12)
axes[1, 0].set_ylabel('Error (%)', fontsize=12)
axes[1, 0].axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.7)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Check for patterns
error_trend = np.polyfit(range(len(rf_return_errors)), rf_return_errors, 1)[0]
if abs(error_trend) < 0.001:
    trend_text = "✅ No systematic drift"
else:
    trend_text = f"⚠️ Trend: {error_trend:.4f}%/day"
axes[1, 0].text(0.02, 0.98, trend_text,
                transform=axes[1, 0].transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))


# --- PLOT 4: Cumulative Absolute Error ---
cumulative_abs_error = np.cumsum(np.abs(rf_return_errors))
axes[1, 1].plot(dates_test.values, cumulative_abs_error,
                color='darkgreen', linewidth=2)
axes[1, 1].set_title('Cumulative Absolute Error',
                     fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date', fontsize=12)
axes[1, 1].set_ylabel('Cumulative |Error| (%)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

# Show final cumulative error
final_cum_error = cumulative_abs_error[-1]
axes[1, 1].text(0.98, 0.98, f'Total: {final_cum_error:.2f}%',
                transform=axes[1, 1].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()

print(f"✅ Error analysis visualizations displayed")


# ================================================================================
# SECTION 6: ERROR SUMMARY STATISTICS
# ================================================================================
print("\n" + "="*80)
print("ERROR ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 Return Prediction Errors:")
print(f"   Mean Error: {mean_error:.4f}%")
print(f"   Std Deviation: {std_error:.4f}%")
print(f"   MAE: {rf_test_mae:.4f}%")
print(f"   R² Score: {rf_test_r2:.4f}")

# Error percentiles (how often errors exceed certain thresholds)
percentiles = np.percentile(np.abs(rf_return_errors), [50, 75, 90, 95])

print(f"\n📈 Error Percentiles (Absolute Values):")
print(f"   50th percentile (median): {percentiles[0]:.4f}%")
print(f"   75th percentile: {percentiles[1]:.4f}%")
print(f"   90th percentile: {percentiles[2]:.4f}%")
print(f"   95th percentile: {percentiles[3]:.4f}%")

print(f"\n💡 Interpretation:")
print(f"   • 50% of predictions within ±{percentiles[0]:.2f}%")
print(f"   • 90% of predictions within ±{percentiles[2]:.2f}%")
print(f"   • 95% of predictions within ±{percentiles[3]:.2f}%")


# ================================================================================
# SECTION 7: BIAS ANALYSIS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 7: Bias Analysis")
print("-"*80)

# Separate positive and negative errors
positive_errors = rf_return_errors[rf_return_errors > 0]  # Underestimations
negative_errors = rf_return_errors[rf_return_errors < 0]  # Overestimations

# Calculate counts and percentages
n_positive = len(positive_errors)
n_negative = len(negative_errors)
total = len(rf_return_errors)

pct_positive = (n_positive / total) * 100
pct_negative = (n_negative / total) * 100

print(f"\n⚖️ Prediction Bias:")
print(f"   Underestimations (actual > predicted): {n_positive} ({pct_positive:.1f}%)")
print(f"   Overestimations (actual < predicted): {n_negative} ({pct_negative:.1f}%)")

# Calculate mean errors for each type
mean_underestimation = np.mean(positive_errors) if len(positive_errors) > 0 else 0
mean_overestimation = np.mean(negative_errors) if len(negative_errors) > 0 else 0

print(f"\n📊 Average Magnitudes:")
print(f"   When underestimating: {mean_underestimation:.4f}%")
print(f"   When overestimating: {mean_overestimation:.4f}%")

# Overall bias assessment
if abs(pct_positive - 50) < 5:
    print(f"\n✅ Model is WELL-BALANCED (nearly 50-50 split)")
elif pct_positive > 55:
    print(f"\n⚠️ Model tends to UNDERESTIMATE (predicts too low)")
else:
    print(f"\n⚠️ Model tends to OVERESTIMATE (predicts too high)")


# ================================================================================
# SECTION 8: KEY INSIGHTS
# ================================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print(f"\n🔍 Top 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"   {i+1}. {row['Feature']}: {row['Importance']*100:.2f}%")

print(f"\n📊 Model Performance:")
if abs(mean_error) < 0.1 and std_error < 2.0:
    print(f"   ✅ GOOD: Low bias, low variance")
elif abs(mean_error) < 0.1:
    print(f"   ✅ GOOD: Unbiased but high variance")
elif std_error < 2.0:
    print(f"   ⚠️ BIASED: Low variance but systematic error")
else:
    print(f"   ⚠️ NEEDS IMPROVEMENT: Both bias and variance issues")

print(f"\n💡 Recommendations:")
if top_3_importance > 0.4:
    print(f"   • Model heavily relies on top 3 features")
    print(f"   • Consider feature selection to simplify model")
if abs(mean_error) > 0.5:
    print(f"   • Systematic bias detected - review feature engineering")
if std_error > 2.5:
    print(f"   • High variance - consider regularization or simpler model")

print(f"\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE!")
print("="*80)
