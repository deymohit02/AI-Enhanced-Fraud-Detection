"""
Test the model with various transactions to check predictions
"""
import pandas as pd
import pickle
import numpy as np

# Load model and scaler
with open('models/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv('data/creditcard.csv')

print("Testing model predictions...")
print("=" * 70)

# Test 1: Real non-fraud transaction
print("\n1. REAL NON-FRAUD TRANSACTION:")
non_fraud = df[df['Class'] == 0].iloc[0]
X_test = non_fraud.drop('Class').values.reshape(1, -1)
X_scaled = scaler.transform(X_test)
prob = model.predict_proba(X_scaled)[0][1]
print(f"   Amount: ${non_fraud['Amount']:.2f}")
print(f"   Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"   Prediction: {'FRAUD' if prob > 0.5 else 'LEGITIMATE'}")

# Test 2: Real fraud transaction
print("\n2. REAL FRAUD TRANSACTION:")
fraud = df[df['Class'] == 1].iloc[0]
X_test = fraud.drop('Class').values.reshape(1, -1)
X_scaled = scaler.transform(X_test)
prob = model.predict_proba(X_scaled)[0][1]
print(f"   Amount: ${fraud['Amount']:.2f}")
print(f"   Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"   Prediction: {'FRAUD' if prob > 0.5 else 'LEGITIMATE'}")

# Test 3: Simulated feature from web UI
print("\n3. WEB UI SIMULATION (Amount: 150, Time: 3000):")
sampleVFeatures = [
    -0.05, -0.01, -0.04, -0.04, -0.01, 0.02, 0.03, -0.01,
    0.00, -0.02, 0.01, 0.00, -0.01, 0.01, -0.01, 0.00,
    0.00, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00
]
amount = 150
time_val = 3000
noise = (amount / 100) * 0.1
v_features = [val + (np.random.random() - 0.5) * noise for val in sampleVFeatures]
features = [time_val] + v_features + [amount]

X_test = np.array(features).reshape(1, -1)
X_scaled = scaler.transform(X_test)
prob = model.predict_proba(X_scaled)[0][1]
print(f"   Amount: ${amount:.2f}")
print(f"   Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"   Prediction: {'FRAUD' if prob > 0.5 else 'LEGITIMATE'}")

# Test 4: High amount transaction
print("\n4. HIGH AMOUNT TRANSACTION (Amount: 5000, Time: 10000):")
amount = 5000
time_val = 10000
noise = (amount / 100) * 0.1
v_features = [val + (np.random.random() - 0.5) * noise for val in sampleVFeatures]
features = [time_val] + v_features + [amount]

X_test = np.array(features).reshape(1, -1)
X_scaled = scaler.transform(X_test)
prob = model.predict_proba(X_scaled)[0][1]
print(f"   Amount: ${amount:.2f}")
print(f"   Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"   Prediction: {'FRAUD' if prob > 0.5 else 'LEGITIMATE'}")

print("\n" + "=" * 70)
print("Analysis:")
print("- If web UI always shows 0.0%, the V-features might be too similar")
print("- Model may be very conservative (low false positive rate)")
print("- Could adjust web UI to use more varied V-feature samples")
