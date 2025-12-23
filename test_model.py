import pickle
import pandas as pd
import numpy as np

# Load the model and scaler
with open('models/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Test case from test_api_local.py
print("=== Test from test_api_local.py ===")
amount = 150.00
time_val = 3000
row = [time_val] + [0.0]*28 + [amount]
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
df = pd.DataFrame([row], columns=columns)

print("\nInput features:")
print(f"Time: {time_val}")
print(f"Amount: {amount}")
print(f"V1-V28: all zeros")

# Scale
X_scaled = scaler.transform(df)
print("\nScaled features (first 5 and last 5):")
print(X_scaled[0][:5])
print("...")
print(X_scaled[0][-5:])

# Predict
proba = model.predict_proba(X_scaled)
print(f"\nPrediction probabilities:")
print(f"Non-fraud: {proba[0][0]:.4f}")
print(f"Fraud: {proba[0][1]:.4f}")
print(f"\nResult: {'FRAUD' if proba[0][1] > 0.5 else 'NOT FRAUD'}")

# Now test with actual dataset statistics
print("\n\n=== Testing with realistic values ===")
df_real = pd.read_csv('data/creditcard.csv')
# Get a typical non-fraud transaction
typical_non_fraud = df_real[df_real['Class']==0].sample(1, random_state=42)
print("\nTypical non-fraud transaction:")
X_test = typical_non_fraud.drop('Class', axis=1)
X_test_scaled = scaler.transform(X_test)
proba_real = model.predict_proba(X_test_scaled)
print(f"Amount: {X_test['Amount'].values[0]}")
print(f"Time: {X_test['Time'].values[0]}")
print(f"Fraud probability: {proba_real[0][1]:.4f}")

# Now test with zeros only for V features but realistic time/amount
print("\n\n=== Testing with zeros for V features ===")
amount2 = 50.00
time_val2 = 1000
row2 = [time_val2] + [0.0]*28 + [amount2]
df2 = pd.DataFrame([row2], columns=columns)
X_scaled2 = scaler.transform(df2)
proba2 = model.predict_proba(X_scaled2)
print(f"Time: {time_val2}, Amount: {amount2}")
print(f"Fraud probability: {proba2[0][1]:.4f}")
print(f"Result: {'FRAUD' if proba2[0][1] > 0.5 else 'NOT FRAUD'}")
