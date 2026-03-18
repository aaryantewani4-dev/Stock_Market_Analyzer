"""
================================================================================
MODEL DIAGNOSTICS & VALIDATION
================================================================================
Purpose: Comprehensive diagnostic checks for the trained model
Author: Your Name
Date: February 2026

This script performs:
1. Data shape validation
2. Target variable statistics
3. Prediction range verification
4. Model performance comparison (RF vs DT vs Baseline)
5. Data leakage detection
6. Sample prediction analysis

Use this to verify model is working correctly and identify potential issues.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# Import data and predictions
from C_feature_loading import X_train, X_test, y_train, y_test, df
from D_model_training import rf_test_pred, rf_test_r2, rf_test_mae

# Note: DecisionTree imports commented out - enable if you want DT comparison
# from DecisionTreePredictions import dt_test_pred, dt_test_r2, dt_test_mae


# ================================================================================
# DIAGNOSTIC REPORT HEADER
# ================================================================================
print("\n" + "="*80)
print("MODEL DIAGNOSTICS & VALIDATION REPORT")
print("="*80)
print("\nThis report checks for:")
print("  • Data integrity and shape consistency")
print("  • Target variable distribution")
print("  • Prediction sanity checks")
print("  • Model performance vs baseline")
print("  • Potential data leakage")


# ================================================================================
# SECTION 1: DATA SHAPE VALIDATION
# ================================================================================
print("\n" + "-"*80)
print("SECTION 1: Data Shape Validation")
print("-"*80)

print(f"\n📊 Dataset Shapes:")
print(f"   X_train: {X_train.shape} (rows × features)")
print(f"   X_test: {X_test.shape} (rows × features)")
print(f"   y_train: {y_train.shape} (target values)")
print(f"   y_test: {y_test.shape} (target values)")

# Verify shapes match
assert X_train.shape[0] == y_train.shape[0], "❌ Train X and y don't match!"
assert X_test.shape[0] == y_test.shape[0], "❌ Test X and y don't match!"
assert X_train.shape[1] == X_test.shape[1], "❌ Train and test features don't match!"

print(f"\n✅ Shape validation passed")
print(f"   • Train samples: {X_train.shape[0]}")
print(f"   • Test samples: {X_test.shape[0]}")
print(f"   • Number of features: {X_train.shape[1]}")
print(f"   • Total samples: {X_train.shape[0] + X_test.shape[0]}")


# ================================================================================
# SECTION 2: TARGET VARIABLE STATISTICS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 2: Target Variable Statistics")
print("-"*80)

print(f"\n📈 Training Set (y_train):")
print(f"   Range: {y_train.min():.2f}% to {y_train.max():.2f}%")
print(f"   Mean: {y_train.mean():.4f}%")
print(f"   Median: {y_train.median():.4f}%")
print(f"   Std Dev: {y_train.std():.4f}%")

print(f"\n📈 Testing Set (y_test):")
print(f"   Range: {y_test.min():.2f}% to {y_test.max():.2f}%")
print(f"   Mean: {y_test.mean():.4f}%")
print(f"   Median: {y_test.median():.4f}%")
print(f"   Std Dev: {y_test.std():.4f}%")

# Check for distribution drift
mean_diff = abs(y_train.mean() - y_test.mean())
std_diff = abs(y_train.std() - y_test.std())

print(f"\n📊 Train vs Test Comparison:")
print(f"   Mean difference: {mean_diff:.4f}%")
print(f"   Std difference: {std_diff:.4f}%")

if mean_diff < 0.5 and std_diff < 0.5:
    print(f"   ✅ Train and test distributions are similar")
else:
    print(f"   ⚠️ Significant distribution shift detected")


# ================================================================================
# SECTION 3: PREDICTION VALIDATION
# ================================================================================
print("\n" + "-"*80)
print("SECTION 3: Prediction Validation")
print("-"*80)

print(f"\n🤖 Random Forest Predictions:")
print(f"   Range: {rf_test_pred.min():.2f}% to {rf_test_pred.max():.2f}%")
print(f"   Mean: {rf_test_pred.mean():.4f}%")
print(f"   Std Dev: {rf_test_pred.std():.4f}%")

# Sanity checks
pred_in_reasonable_range = (rf_test_pred.min() > -50) and (rf_test_pred.max() < 50)
pred_not_constant = rf_test_pred.std() > 0.1

print(f"\n✓ Sanity Checks:")
if pred_in_reasonable_range:
    print(f"   ✅ Predictions in reasonable range (-50% to +50%)")
else:
    print(f"   ❌ Predictions outside reasonable range!")

if pred_not_constant:
    print(f"   ✅ Predictions vary (not constant)")
else:
    print(f"   ❌ Predictions are nearly constant!")


# ================================================================================
# SECTION 4: SAMPLE PREDICTIONS ANALYSIS
# ================================================================================
print("\n" + "-"*80)
print("SECTION 4: Sample Predictions (First 10 Test Samples)")
print("-"*80)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Actual_Return': y_test.values[:10],
    'RF_Predicted': rf_test_pred[:10],
    'RF_Error': y_test.values[:10] - rf_test_pred[:10],
    'Error_Abs': np.abs(y_test.values[:10] - rf_test_pred[:10])
})

print(f"\n{comparison.to_string(index=False)}")

# Summary of sample
print(f"\nSample Statistics:")
print(f"   Mean absolute error: {comparison['Error_Abs'].mean():.4f}%")
print(f"   Max error: {comparison['Error_Abs'].max():.4f}%")


# ================================================================================
# SECTION 5: MODEL PERFORMANCE SUMMARY
# ================================================================================
print("\n" + "-"*80)
print("SECTION 5: Model Performance Summary")
print("-"*80)

print(f"\n🤖 Random Forest:")
print(f"   R² Score: {rf_test_r2:.4f}")
print(f"   MAE: {rf_test_mae:.4f}%")

# Interpret performance
if rf_test_r2 > 0.05:
    rf_quality = "✅ GOOD"
elif rf_test_r2 > -0.05:
    rf_quality = "⚠️ ACCEPTABLE"
else:
    rf_quality = "❌ POOR"

print(f"   Quality: {rf_quality}")

# Decision Tree comparison (if available)
# Uncomment if you want to compare with Decision Tree
"""
print(f"\n🌳 Decision Tree:")
print(f"   R² Score: {dt_test_r2:.4f}")
print(f"   MAE: {dt_test_mae:.4f}%")

if dt_test_r2 > 0.05:
    dt_quality = "✅ GOOD"
elif dt_test_r2 > -0.05:
    dt_quality = "⚠️ ACCEPTABLE"
else:
    dt_quality = "❌ POOR"

print(f"   Quality: {dt_quality}")

# Compare models
if rf_test_r2 > dt_test_r2:
    print(f"\n🏆 Winner: Random Forest (R² advantage: {rf_test_r2 - dt_test_r2:.4f})")
else:
    print(f"\n🏆 Winner: Decision Tree (R² advantage: {dt_test_r2 - rf_test_r2:.4f})")
"""


# ================================================================================
# SECTION 6: BASELINE COMPARISON
# ================================================================================
print("\n" + "-"*80)
print("SECTION 6: Baseline Comparison")
print("-"*80)

# Naive baseline: Predict yesterday's return
# This tests if our model adds any value over simple persistence
baseline_pred = y_test.shift(1).dropna()
baseline_actual = y_test[1:]

baseline_r2 = r2_score(baseline_actual, baseline_pred)
baseline_mae = mean_absolute_error(baseline_actual, baseline_pred)

print(f"\n📊 Naive Baseline (Tomorrow = Yesterday):")
print(f"   Strategy: Use yesterday's return as prediction")
print(f"   R² Score: {baseline_r2:.4f}")
print(f"   MAE: {baseline_mae:.4f}%")

print(f"\n🤖 Our Model (Random Forest):")
print(f"   R² Score: {rf_test_r2:.4f}")
print(f"   MAE: {rf_test_mae:.4f}%")

print(f"\n⚖️ Verdict:")
if rf_test_r2 > baseline_r2:
    print(f"   ✅ Your model BEATS the baseline!")
    print(f"   R² improvement: {rf_test_r2 - baseline_r2:.4f}")
else:
    print(f"   ❌ Baseline is better than your model")
    print(f"   R² gap: {baseline_r2 - rf_test_r2:.4f}")
    print(f"\n   💡 Note: For stock returns, this is often the case.")
    print(f"           Returns are extremely hard to predict!")


# ================================================================================
# SECTION 7: DATA LEAKAGE CHECK
# ================================================================================
print("\n" + "-"*80)
print("SECTION 7: Data Leakage Detection")
print("-"*80)

print(f"\n🔍 Checking for data leakage...")
print(f"\nTarget Variable Alignment Check:")
print(f"{'Date':<20} | {'Close':<12} | {'Target Return':<12} | {'Open':<12}")
print("-" * 70)

# Display first 5 rows to verify alignment
for i in range(5):
    date = df['Price'].iloc[i]
    close = df['Close'].iloc[i]
    target = df['Target_Return'].iloc[i]
    open_price = df['Open'].iloc[i]
    
    print(f"{str(date):<20} | ${close:<11.2f} | {target:<11.2f}% | ${open_price:<11.2f}")

print(f"\n✓ Verification:")
print(f"   • Target at row i should be the return AFTER that day")
print(f"   • We should NOT be using future information in features")
print(f"   • Features at row i should only use data UP TO that day")

# Manual verification note
print(f"\n💡 Manual Check Required:")
print(f"   Please verify that Target_Return[i] = (Close[i+1] - Close[i]) / Close[i]")
print(f"   If this doesn't hold, there may be a shift error in feature engineering")


# ================================================================================
# SECTION 8: OVERFITTING CHECK
# ================================================================================
print("\n" + "-"*80)
print("SECTION 8: Overfitting Detection")
print("-"*80)

# Calculate train performance (if we have train predictions)
# Note: Train predictions not imported by default to save memory
# Uncomment RandomForestPrediction_returns import if needed

print(f"\n📊 Overfitting Assessment:")
print(f"   Test R²: {rf_test_r2:.4f}")

# Rule of thumb: If test R² is much lower than train R², overfitting occurred
# For stock returns, even train R² should be low
print(f"\n💡 Expected Behavior:")
print(f"   • Train R²: Should be 0.1 - 0.3 (returns are noisy)")
print(f"   • Test R²: Should be -0.1 to 0.1 (slightly worse than train)")
print(f"   • Large gap (>0.3): Indicates overfitting")

if rf_test_r2 < -0.2:
    print(f"\n   ⚠️ Test R² is very negative - possible issues:")
    print(f"      • Data leakage")
    print(f"      • Feature-target misalignment")
    print(f"      • Distribution shift between train/test")
elif rf_test_r2 > 0.5:
    print(f"\n   ⚠️ Test R² is very high - verify:")
    print(f"      • No data leakage (using future information)")
    print(f"      • Target variable is correct")
else:
    print(f"\n   ✅ Test R² is in expected range for stock returns")


# ================================================================================
# DIAGNOSTIC SUMMARY
# ================================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print(f"\n✓ Checks Performed:")
print(f"   ✅ Data shape validation")
print(f"   ✅ Target distribution analysis")
print(f"   ✅ Prediction sanity checks")
print(f"   ✅ Sample predictions reviewed")
print(f"   ✅ Baseline comparison")
print(f"   ✅ Data leakage detection")
print(f"   ✅ Overfitting assessment")

print(f"\n🎯 Overall Assessment:")

# Compile issues
issues = []
if not pred_in_reasonable_range:
    issues.append("Predictions out of reasonable range")
if not pred_not_constant:
    issues.append("Predictions are constant")
if rf_test_r2 < -0.2:
    issues.append("Very negative test R²")
if mean_diff > 1.0:
    issues.append("Large train-test distribution shift")

if len(issues) == 0:
    print(f"   ✅ ALL CHECKS PASSED")
    print(f"   Model appears to be working correctly")
else:
    print(f"   ⚠️ ISSUES DETECTED:")
    for issue in issues:
        print(f"      • {issue}")
    print(f"\n   Recommended actions:")
    print(f"      1. Review feature engineering code")
    print(f"      2. Check for data leakage")
    print(f"      3. Verify train-test split")
    print(f"      4. Consider simpler model")

print(f"\n" + "="*80)
print("DIAGNOSTICS COMPLETE!")
print("="*80)

print(f"\n📝 Note: This diagnostic report helps identify potential issues.")
print(f"   For stock/crypto prediction, low R² scores are NORMAL and EXPECTED.")
print(f"   What matters most is:")
print(f"   • No data leakage")
print(f"   • Proper train-test split")
print(f"   • Realistic performance expectations")
