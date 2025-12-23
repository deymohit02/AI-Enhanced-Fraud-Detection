import pandas as pd
import pickle
import numpy as np
import os
import sys

# Load model and scaler
try:
    with open('models/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
        
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Models loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Load dataset to get a real fraud example
try:
    df = pd.read_csv('data/creditcard.csv')
    fraud_case = df[df['Class'] == 1].iloc[0]
    fraud_v_features = fraud_case.drop(['Time', 'Amount', 'Class']).values.tolist()
    
    normal_case = df[df['Class'] == 0].iloc[0]
    normal_v_features = normal_case.drop(['Time', 'Amount', 'Class']).values.tolist()
    print("Dataset loaded.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Fallback to hardcoded approximate fraud features (V14 is usually very negative)
    fraud_v_features = [0] * 28
    fraud_v_features[13] = -10.0 # V14
    fraud_v_features[3] = 5.0 # V4
    normal_v_features = [0] * 28

def predict(amount, time_val, v_features, label):
    features = [time_val] + v_features + [amount]
    X_test = np.array(features).reshape(1, -1)
    # Reconstruct DF for scaler consistent with app.py
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_test = pd.DataFrame(features).T
    df_test.columns = columns
    
    X_scaled = scaler.transform(df_test)
    prob = model.predict_proba(X_scaled)[0][1]
    print(f"\nScenario: {label}")
    print(f"   Amount: ${amount:,.2f}")
    print(f"   Time: {time_val}")
    print(f"   Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")

# 1. User's Case: High Amount, Normal V-Features (simulated by Web UI typically)
# If Web UI sends 0s or small randoms for V-features
predict(5_000_000_000, 0, normal_v_features, "User Case: 5 Billion, Normal V-Features")

# 2. Fraud V-Features, Normal Amount
predict(100, 0, fraud_v_features, "Real Fraud V-Features, Small Amount")

# 3. Fraud V-Features, High Amount
predict(5_000_000_000, 0, fraud_v_features, "Real Fraud V-Features, 5 Billion Amount")

# 4. Normal V-Features, Normal Amount
predict(100, 0, normal_v_features, "Normal V-Features, Small Amount")
