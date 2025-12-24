"""
Quick test script for feature engineering
Tests basic functionality without pytest
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer

print("="*70)
print("FEATURE ENGINEERING - QUICK TEST")
print("="*70)

# Create test data
print("\n1. Creating sample transaction data...")
np.random.seed(42)
n = 100

data = {
    'Time': np.sort(np.random.randint(0, 172800, n)),
    'Amount': np.random.exponential(100, n)
}

for i in range(1, 29):
    data[f'V{i}'] = np.random.randn(n)

df = pd.DataFrame(data)
print(f"   ✅ Created {len(df)} transactions")
print(f"   Columns: {df.shape[1]}")

# Initialize engineer
print("\n2. Initializing FeatureEngineer...")
engineer = FeatureEngineer(cache_enabled=True)
print(f"   ✅ Initialized")
print(f"   Engineered feature count: {engineer.get_feature_count()}")

# Generate features
print("\n3. Generating features (batch mode)...")
df_features = engineer.generate_all_features(df, is_batch=True)
print(f"   ✅ Generated features")
print(f"   Input columns: {df.shape[1]}")
print(f"   Output columns: {df_features.shape[1]}")
print(f"   Features added: {df_features.shape[1] - df.shape[1]}")

# Test specific features
print("\n4. Validating feature generation...")
tests_passed = 0
tests_total = 0

# Test 1: Time features
tests_total += 1
if 'time_hour' in df_features.columns:
    if (df_features['time_hour'] >= 0).all() and (df_features['time_hour'] <= 23).all():
        print("   ✅ Time features: PASS")
        tests_passed += 1
    else:
        print("   ❌ Time features: FAIL (invalid hour values)")
else:
    print("   ❌ Time features: FAIL (column missing)")

# Test 2: Amount features
tests_total += 1
if 'amount_log' in df_features.columns and 'amount_zscore' in df_features.columns:
    if (df_features['amount_log'] >= 0).all():
        print("   ✅ Amount features: PASS")
        tests_passed += 1
    else:
        print("   ❌ Amount features: FAIL (invalid values)")
else:
    print("   ❌ Amount features: FAIL (columns missing)")

# Test 3: Velocity features
tests_total += 1
if 'txn_count_1hr' in df_features.columns:
    if (df_features['txn_count_1hr'] >= 0).all():
        print("   ✅ Velocity features: PASS")
        tests_passed += 1
    else:
        print("   ❌ Velocity features: FAIL (invalid counts)")
else:
    print("   ❌ Velocity features: FAIL (columns missing)")

# Test 4: V-feature analysis
tests_total += 1
if 'v_mean' in df_features.columns and 'v_norm' in df_features.columns:
    if (df_features['v_norm'] >= 0).all():
        print("   ✅ V-feature analysis: PASS")
        tests_passed += 1
    else:
        print("   ❌ V-feature analysis: FAIL (invalid values)")
else:
    print("   ❌ V-feature analysis: FAIL (columns missing)")

# Test 5: Behavioral features
tests_total += 1
if 'anomaly_score' in df_features.columns and 'risk_indicator' in df_features.columns:
    print("   ✅ Behavioral features: PASS")
    tests_passed += 1
else:
    print("   ❌ Behavioral features: FAIL (columns missing)")

# Test API extraction
print("\n5. Testing API feature extraction...")
try:
    features = engineer.extract_features_for_api(
        amount=250.75,
        time=7200,
        v_features=[0.5] * 28
    )
    
    if isinstance(features, dict) and 'amount_log' in features:
        print(f"   ✅ API extraction: PASS")
        print(f"   Returned {len(features)} features")
        tests_passed += 1
        tests_total += 1
    else:
        print("   ❌ API extraction: FAIL (invalid output)")
        tests_total += 1
except Exception as e:
    print(f"   ❌ API extraction: FAIL ({e})")
    tests_total += 1

# Summary
print(f"\n{'='*70}")
print(f"TEST SUMMARY")
print(f"{'='*70}")
print(f"Tests Passed: {tests_passed}/{tests_total}")
print(f"Success Rate: {tests_passed/tests_total*100:.1f}%")

if tests_passed == tests_total:
    print(f"\n🎉 ALL TESTS PASSED!")
    exit(0)
else:
    print(f"\n⚠️  Some tests failed")
    exit(1)
